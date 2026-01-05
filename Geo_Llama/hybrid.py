import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .core.layers import GeoLlamaState, SpecializedLiftingLayer
from .core.gca_pytorch import GeoAttentionBias

class GeoLlamaHybrid:
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct", device="cpu", is_mock=False):
        self.device = device
        self.is_mock = is_mock
        self.target_geo_heads = 64 # README Spec
        self.d_model = 2048 # Standard for 1B
        
        if not is_mock:
            print(f"Loading {model_id}...")
            # 1. Load the Lexical System (T-Stream)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto" if device != "cpu" else None
            )
            self.d_model = self.model.config.hidden_size
            self.llama_num_heads = self.model.config.num_attention_heads
            self.head_dim = self.d_model // self.llama_num_heads
        else:
            print("Running in MOCK mode (Geometric logic testing)...")
            self.model = None
            self.tokenizer = None
            self.llama_num_heads = 32
            self.head_dim = 64

        # 2. Setup the Structural System (G-Stream)
        # We use SpecializedLiftingLayer which returns (Batch, Seq, Heads, 32) Rotors directly
        self.lifter = SpecializedLiftingLayer(d_model=self.d_model, num_heads=self.target_geo_heads).to(device)
        self.state = GeoLlamaState(num_heads=self.target_geo_heads, d_model=self.d_model, device=device)
        
        # 3. Setup the GCA Engine
        self.gca_engine = GeoAttentionBias(num_heads=self.target_geo_heads, lambda_val=0.1).to(device)
        self.gca_engine.eval()
        
        if self.model: self.model.eval()
        self.lifter.eval()
        
        # 4. Head Projector for GCA (d_head -> 32)
        # Initializes near identity to preserve structure initially
        self.head_lifter = torch.nn.Linear(self.head_dim, 32, bias=False).to(device)
        with torch.no_grad():
            self.head_lifter.weight.fill_(0.0)
            # Fill diagonal roughly
            n = min(self.head_dim, 32)
            for i in range(n):
                self.head_lifter.weight[i, i] = 0.1
                
        # 5. Manifold Mixing Layer (Section 9.2)
        from .core.layers import ManifoldMixingLayer
        self.mixing_layer = ManifoldMixingLayer(num_heads=self.target_geo_heads).to(device)
        
    def gca_bias(self, Q, K):
        """
        Calculates the geometry-conditioned attention bias.
        Formula: lambda * <PSI, Q ^ K>
        """
        # Q, K (standard): (batch, n_heads_llama, seq_len, d_head)
        # We need to map Llama Heads -> Geo Heads
        # Strategy: Lift Q/K to 32 and then broadcast/tile to 64 or 
        # project to 64 directly if we had a specific lifter.
        # Since we have a mismatch (32 vs 64), we'll do the GCA on the *available* Llama heads
        # derived from the Geometric Stream.
        
        # Wait, the README structure implies the G-Stream runs parallel.
        # We will compute the bias in the 64-dim space and then project down to Llama space.
        
        # 1. Use the Hidden State Rotors (Already computed in forward pass ideally)
        # But here we are inside the attention layer.
        # We will lift Q and K using our lightweight head_lifter.
        
        Q_lifted = self.head_lifter(Q) # (B, 32, S, 32_cga)
        K_lifted = self.head_lifter(K)
        
        # Tile to match 64 geometric heads if needed, or just use 32 subset?
        # The PSI state has 64 heads.
        # Let's repeat Q/K to match 64.
        # (B, 32, S, 32) -> (B, 64, S, 32)
        if self.llama_num_heads != self.target_geo_heads:
            ratio = self.target_geo_heads // self.llama_num_heads
            Q_lifted = Q_lifted.repeat_interleave(ratio, dim=1)
            K_lifted = K_lifted.repeat_interleave(ratio, dim=1)
            
        # 2. Compute the bias using the current G-Stream state PSI
        psi_torch = self.state.psi.to(self.device) # (64, 32)
        
        # Bias: (B, 64, S, S)
        bias_64 = self.gca_engine(psi_torch, Q_lifted, K_lifted)
        
        # 3. Project back to Llama Heads
        # Average the bias across the groups
        if self.llama_num_heads != self.target_geo_heads:
            ratio = self.target_geo_heads // self.llama_num_heads
            bias_32 = bias_64.view(bias_64.shape[0], self.llama_num_heads, ratio, bias_64.shape[2], bias_64.shape[3])
            bias = bias_32.mean(dim=2)
        else:
            bias = bias_64
            
        return bias

    def hook_attention(self):
        """
        Injects GCA Bias into all Llama attention layers via attention_mask modification.
        """
        if self.is_mock or not self.model:
            print("Skipping hooks (Mock mode active).")
            return

        print(f"Hooking onto {len(self.model.model.layers)} Llama attention layers...")
        
        hybrid_self = self 

        def create_custom_forward(original_forward, layer_idx):
            def custom_forward(hidden_states, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=False, use_cache=False, **kwargs):
                # We need Q and K to compute GCA. 
                # Standard LlamaAttention forward computes them internally.
                # To access them without rewriting the whole forward, we have to rely on
                # the fact that we can't easily.
                
                # ALTERNATIVE: We can compute the GCA bias from the *Input Hidden States* 
                # and inject it into the attention_mask.
                
                # 1. Project Hidden States to Q_proxy, K_proxy for GCA
                # We'll use the Llama linear layers if we can access them, or our lifter.
                # Let's use our head_lifter on the reshaped hidden states.
                
                # hidden_states: (B, S, D)
                bsz, q_len, _ = hidden_states.size()
                
                # Reshape to heads: (B, S, H, D_h) -> (B, H, S, D_h)
                # We approximate Q and K as the hidden state itself for the *Topological Check*
                # (checking if the token concepts themselves are compatible).
                h_reshaped = hidden_states.view(bsz, q_len, hybrid_self.llama_num_heads, hybrid_self.head_dim).transpose(1, 2)
                
                # Compute Bias
                # (B, H, S, S)
                geo_bias = hybrid_self.gca_bias(h_reshaped, h_reshaped) 
                
                # Inject into attention_mask
                # attention_mask is usually (B, 1, Q, K) or (B, 1, 1, K)
                # We add our bias.
                if attention_mask is not None:
                    # diff dims might require careful broadcasting
                    # mask: (B, 1, S, S) usually for causal
                    # geo_bias: (B, 32, S, S)
                    # We rely on broadcasting.
                    combined_mask = attention_mask + geo_bias
                else:
                    # If no mask (e.g. inference), we create one from bias? 
                    # Usually Llama handles mask generation internally if None.
                    # We can't easily inject if None.
                    # But for 'generate', mask is usually provided.
                    combined_mask = geo_bias

                # Call original with modified mask
                return original_forward(
                    hidden_states, 
                    attention_mask=combined_mask, 
                    position_ids=position_ids, 
                    past_key_values=past_key_values, 
                    output_attentions=output_attentions, 
                    use_cache=use_cache, 
                    **kwargs
                )
            return custom_forward

        # Iterate through layers and patch
        for i, layer in enumerate(self.model.model.layers):
            # Bind the original method
            orig_method = layer.self_attn.forward
            layer.self_attn.forward = create_custom_forward(orig_method, i)

        print("Llama architecture successfully augmented with Geometric Logic via Attention Mask Injection.")

    def generate_with_geometry(self, prompt, max_new_tokens=50):
        if not self.is_mock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 1. RESET STATE FOR NEW GENERATION (The Fix)
            # Get batch size from input
            bsz = inputs['input_ids'].shape[0]
            self.state.reset_state(batch_size=bsz)
            
            # Pre-Generation G-Stream Sync
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
        else:
            # Mock logic...
            bsz = 1
            self.state.reset_state(batch_size=bsz)
            inputs = {}
            print(f"Mocking Forward Pass for prompt: '{prompt}'")
            last_hidden_state = torch.randn(1, 5, self.d_model).to(self.device)
            
        # Update the Geometric State (Right Brain)
        rotors = self.lifter(last_hidden_state) # (Batch, Seq, Heads, 32)
        
        # Recursive Context Update
        seq_len = rotors.shape[1]
        
        # We must update time-step by time-step to respect causality
        for t in range(seq_len):
            # rotors[:, t] is (Batch, Heads, 32)
            self.state.update(rotors[:, t], mixing_layer=self.mixing_layer)
            
        print(f"Structural Context PSI synchronized across {self.target_geo_heads} manifolds.")
        
        if not self.is_mock:
            # Generate
            # The 'hook' is active, so generate() will call our custom forward
            # and inject the bias derived from self.state.PSI
            generated = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True, # Allow vivid generation
                temperature=0.7
            )
            return self.tokenizer.decode(generated[0], skip_special_tokens=False)
        else:
            return "[MOCK OUTPUT] Relationship between circle and sphere: A sphere is the set of points in 3D equidistant from a center."
