import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
import sys
import time
import math
import warnings
import datetime
import copy

# =================================================================
# 0. CONFIGURATION & UTILITIES
# =================================================================

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def print_header():
    print(f"\n{'='*80}")
    print(f"BENCHMARK: OPTIMIZED ROTOR SSM (O(1) RECURRENCE)")
    print(f"{'='*80}")
    print(f"Date:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware:     {GPU_NAME}")
    print(f"Mode:         Bitwise Ops + Recurrent State")
    print(f"{'='*80}\n")

def matthews_corrcoef(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0: return 0.0
    return numerator / denominator

def conformal_projection(grid: torch.Tensor):
    """
    Project Binary Grid (0/1) -> Conformal Vectors (Point / Infinity).
    """
    b, seq_len = grid.shape
    device = grid.device 
    # n_o = 0.5(e4-e3), n_inf = e4+e3 
    n_o = torch.zeros(32, device=device); n_o[16], n_o[8] = 0.5, -0.5
    n_inf = torch.zeros(32, device=device); n_inf[16], n_inf[8] = 1.0, 1.0
    
    out = torch.zeros((b, seq_len, 32), device=device)
    out += (grid == 1).unsqueeze(-1) * n_o 
    out += (grid == 0).unsqueeze(-1) * n_inf
    return out

def generate_dataset(size: int, n_samples: int, d_vectors: int = 4):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Data] Generating {n_samples} samples ({size}x{size})...", end=" ", flush=True)
    target_dev = torch.device('cpu') if size >= 64 else DEVICE
    
    X_list, Y_list = [], []
    for _ in range(n_samples):
        grid = np.zeros((size, size), dtype=np.float32)
        curr = (0,0); grid[0,0] = 1; path_coords = [curr]
        while curr != (size-1, size-1):
            cx, cy = curr
            moves = []
            if cx < size-1: moves.append((cx+1, cy))
            if cy < size-1: moves.append((cx, cy+1))
            if not moves: break
            curr = moves[np.random.randint(len(moves))]
            path_coords.append(curr); grid[curr] = 1
            
        is_solvable = (np.random.rand() > 0.5)
        if is_solvable: label = 1
        else:
            label = 0
            if len(path_coords) > 5:
                cut_idx = np.random.randint(2, len(path_coords)-2)
                grid[path_coords[cut_idx]] = 0 

        X_list.append(grid.flatten()); Y_list.append(label)
        
    X_t = torch.tensor(np.array(X_list), device=target_dev)
    Y_t = torch.tensor(np.array(Y_list), device=target_dev)
    X_proj = conformal_projection(X_t).unsqueeze(2).repeat(1, 1, d_vectors, 1)
    print("Done.")
    return X_proj, Y_t

# =================================================================
# 1. PHASE 1: BITWISE KERNEL (THE "SHADER")
# =================================================================

def popcount(n):
    return bin(n).count('1')

def compute_gp_sign_bitwise(a, b):
    """
    Computes the sign of e_a * e_b in Cl(4,1) using pure bitwise logic.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    
    swaps_v = torch.zeros_like(a, dtype=torch.long)
    for i in range(5):
        b_has_i = (b >> i) & 1
        mask_gt = (~((1 << (i + 1)) - 1)) & 31
        val = a & mask_gt
        # 5-bit popcount:
        pc = (val & 1) + ((val >> 1) & 1) + ((val >> 2) & 1) + ((val >> 3) & 1) + ((val >> 4) & 1)
        swaps_v += b_has_i * pc
        
    commutation_sign = torch.pow(-1.0, swaps_v.float())
    metric_sq = (a & 16) & (b & 16)
    metric_sign = torch.where(metric_sq > 0, -1.0, 1.0)
    
    return commutation_sign * metric_sign

def get_gp_map_reference(device):
    table = torch.zeros((32, 32, 32), device=device)
    for a in range(32):
        for b in range(32):
            sign, a_bits = 1.0, a
            for i in range(5):
                if (b >> i) & 1:
                    for j in range(i + 1, 5):
                        if (a_bits >> j) & 1: sign *= -1.0
                    if (a_bits >> i) & 1:
                        if i == 4: sign *= -1.0 
                        a_bits &= ~(1 << i)
                    else: a_bits |= (1 << i)
            table[a, b, a_bits] = sign
    return table

def verify_bitwise_logic():
    print(f"[Self-Check] Verifying Bitwise Algebra Logic...", end=" ")
    a = torch.arange(32).unsqueeze(1).repeat(1, 32).flatten()
    b = torch.arange(32).repeat(32)
    
    pred_res = a ^ b
    pred_sign = compute_gp_sign_bitwise(a, b)
    ref_table = get_gp_map_reference('cpu')
    
    errors = 0
    for idx in range(1024):
        ai, bi = a[idx].item(), b[idx].item()
        ri = pred_res[idx].item()
        s = pred_sign[idx].item()
        ref_s = ref_table[ai, bi, ri].item()
        if not math.isclose(s, ref_s):
            errors += 1
            
    if errors == 0:
        print("PASS ✓")
    else:
        print(f"FAIL ✗ ({errors} mismatches)")
        sys.exit(1)

# =================================================================
# 2. PHASE 2: ROTOR SSM (O(1) RECURRENCE)
# =================================================================

class FastGeometricProduct(nn.Module):
    """
    Computes Geometric Product W * x WITHOUT expanding a full 32x32x32 table.
    Uses indices permutations and bitwise signs.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        std = 0.5 / (in_features * 32)**0.5
        self.weight.data.normal_(0, std)
        
        self.register_buffer('perm_indices', torch.zeros(32, 32, dtype=torch.long))
        self.register_buffer('perm_signs', torch.zeros(32, 32))
        
        indices = torch.arange(32)
        for k in range(32):
            l_idx = indices ^ k 
            self.perm_indices[k] = l_idx
            self.perm_signs[k] = compute_gp_sign_bitwise(indices, l_idx)

    def forward(self, x):
        # x: (B, In, 32)
        B, In = x.shape[0], x.shape[1]
        x_flat = x
        w_flat = self.weight
        
        outputs = []
        for k in range(32):
            p_idx = self.perm_indices[k] 
            s_val = self.perm_signs[k]
            
            x_k_perm = x_flat[..., p_idx] * s_val # (B, In, 32)
            
            # einsum: bim, oim -> bo (sum over Input and Basis_m)
            val_k = torch.einsum('bim, oim -> bo', x_k_perm, w_flat)
            outputs.append(val_k)
            
        return torch.stack(outputs, dim=-1)

class RotorSSM_Cell(nn.Module):
    """
    Psi_{t+1} = Norm( R_t Psi_t ~R_t )
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Generates Rotor from Input
        self.to_rotor = nn.Linear(d_model * 32, 32) 
        
        # Precompute tables
        self.register_buffer('rev_signs', torch.ones(32))
        self.register_buffer('metric', torch.ones(32))
        self.register_buffer('perm_indices', torch.zeros(32, 32, dtype=torch.long))
        self.register_buffer('perm_signs', torch.zeros(32, 32))
                                   
        indices = torch.arange(32)
        for i in range(32):
            # Reversion
            g = bin(i).count('1')
            if (g * (g - 1) // 2) % 2 == 1: self.rev_signs[i] = -1.0
            
            # Metric
            m = 1.0
            if (i >> 4) & 1: m *= -1.0 # e4^2 = -1
            if (g * (g - 1) // 2) % 2 == 1: m *= -1.0
            self.metric[i] = m

            # Product Tables for reuse
            l = indices ^ i
            self.perm_indices[i] = l
            self.perm_signs[i] = compute_gp_sign_bitwise(indices, l)
            
        self.proj_out = FastGeometricProduct(d_model, d_model)

    def manifold_norm(self, A):
        sig = self.metric.to(A.device)
        norm_sq = torch.sum(A * A * sig, dim=-1)
        eps = 1e-6
        denom = torch.max(
            torch.sqrt(torch.abs(norm_sq) + eps).unsqueeze(-1),
            torch.norm(A, p=2, dim=-1, keepdim=True) / 4.0 + eps
        ).clamp(min=1.0)
        return A / denom

    def gp_broadcast(self, a, b):
        # a: (..., 32_m), b: (..., 32_l)
        # result_k = sum_m (a_m * b_{m^k} * s(m, m^k))
        # Optimized with buffers
        perm = self.perm_indices
        sign = self.perm_signs
        outputs = []
        for k in range(32):
            idx = perm[k]
            s = sign[k].to(a.device)
            b_perm = b[..., idx]
            term = a * b_perm * s 
            outputs.append(term.sum(dim=-1))
        return torch.stack(outputs, dim=-1)

    def forward_step(self, psi, x):
        # x: (B, D, 32)
        b_size = x.shape[0]
        r = self.to_rotor(x.view(b_size, -1)) # (B, 32)
        r = self.manifold_norm(r) # Ensure R is on manifold
        
        # 2. Reversion ~R
        r_rev = r * self.rev_signs
        
        # 3. Geometric Sandwich: Psi_new = R * Psi * ~R
        # Broadcast R (B, 32) against Psi (B, D, 32)
        # R * Psi
        lhs = self.gp_broadcast(r.unsqueeze(1), psi)
        # (R * Psi) * ~R
        # lhs: (B, D, 32), r_rev: (B, 32) -> unsqueeze to match
        psi_new = self.gp_broadcast(lhs, r_rev.unsqueeze(1))
        
        # 4. Integrate Input?
        # The recurrence formula implies rotation of state. 
        # But for 'Broken Snake', we need to check continuity.
        # If the rotor encodes 'movement', Psi tracks position.
        # We assume X generates the rotor.
        
        psi_out = self.manifold_norm(psi_new)
        out = self.proj_out(psi_out)
        return psi_out, out

class Rotor_Graph_Transformer(nn.Module):
    def __init__(self, d_vectors=4):
        super().__init__()
        self.cell = RotorSSM_Cell(d_vectors)
        self.psi_0 = nn.Parameter(torch.randn(1, d_vectors, 32) * 0.02)
        self.pool = nn.Linear(d_vectors*32, d_vectors*32)
        self.head = nn.Linear(d_vectors*32, 2)

    def forward(self, x):
        # x: (B, S, D, 32)
        b, s, d, _ = x.shape
        psi = self.psi_0.repeat(b, 1, 1).to(x.device)
        
        # Sequential Processing (O(N))
        # This loop proves it's O(1) in memory per step
        for t in range(s):
            xt = x[:, t]
            psi, _ = self.cell.forward_step(psi, xt)
            
        pooled = torch.tanh(self.pool(psi.view(b, -1)))
        return self.head(pooled)

# =================================================================
# 3. TRAINING ENGINE
# =================================================================

def run_cycle_rotor(model, size, d_vec, epochs=10):
    n_samples = 2000 # Reduced for specialized benchmark speed
    X, Y = generate_dataset(size, n_samples, d_vec)
    dataset = TensorDataset(X, Y)
    train_len = int(0.9 * len(dataset))
    train, val = random_split(dataset, [train_len, len(dataset)-train_len])
    
    batch_size = 16 if size < 32 else 8
    accum = 1
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    
    model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Train] Size: {size}x{size} | Epochs: {epochs}")
    
    history = []
    for ep in range(epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                o = model(xb)
                preds.extend(o.argmax(1).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        mcc = matthews_corrcoef(trues, preds)
        history.append(mcc)
        print(f"    Ep {ep+1}/{epochs} | MCC: {mcc:.3f}")
        
        if mcc > 0.98:
            print("    -> Converged.")
            break
            
    return history, history[-1]

# =================================================================
# 4. MAIN
# =================================================================

if __name__ == "__main__":
    print_header()
    verify_bitwise_logic()
    
    # Run Benchmark on 8x8 and 16x16 to prove concept
    SIZES = [8, 16] 
    results = {}
    
    print("\n>>> TESTING ROTOR SSM ARCHITECTURE")
    for size in SIZES:
        print(f"\n--- GRID {size}x{size} ---")
        model = Rotor_Graph_Transformer(d_vectors=4)
        hist, mcc = run_cycle_rotor(model, size, 4, epochs=15)
        results[size] = mcc
        
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    for s, m in results.items():
        print(f"Grid {s}x{s}: MCC = {m:.3f}")
    
    # Save graph
    plt.figure()
    plt.bar([str(s) for s in results.keys()], results.values())
    plt.title("Rotor SSM Performance (O(1) Recurrence)")
    plt.ylim(0, 1)
    plt.ylabel("MCC Score")
    plt.savefig('rotor_ssm_benchmark.png')
    print("\n[Output] Saved rotor_ssm_benchmark.png")

    # =================================================================
    # PART 2: MLX vs PYTORCH SPEED TEST
    # =================================================================
    print("\n" + "="*80)
    print("PART 2: MLX (METAL) vs PYTORCH (CUDA/MPS) SPEED TEST")
    print("="*80)
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import mlx.core as mx
        import kernel
        
        print("[MLX DETECTED] Running Hardware Comparison...")
        
        # Config
        BATCH_SIZE = 128
        SEQ_LEN = 128
        DIM = 32
        N_RUNS = 100
        
        # 1. PyTorch Benchmark
        print(f"\n[PyTorch] Benchmarking RotorSSM_Cell ({DEVICE})...")
        pt_model = RotorSSM_Cell(d_model=1).to(DEVICE) # d_model=1 means 1 multivector per token? 
        # Wait, RotorSSM_Cell takes d_model (feature dim). 
        # In this benchmark context, x is (B, D, 32). Let's use D=4 to match earlier.
        D_VEC = 4
        pt_model = RotorSSM_Cell(d_model=D_VEC).to(DEVICE)
        
        # Inputs
        pt_psi = torch.randn(BATCH_SIZE, D_VEC, 32, device=DEVICE)
        pt_x = torch.randn(BATCH_SIZE, SEQ_LEN, D_VEC, 32, device=DEVICE)
        
        # Warmup
        for _ in range(10):
            psi_curr = pt_psi.clone()
            for t in range(SEQ_LEN):
                psi_curr, _ = pt_model.forward_step(psi_curr, pt_x[:, t])
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        for _ in range(N_RUNS):
            psi_curr = pt_psi.clone()
            for t in range(SEQ_LEN):
                psi_curr, _ = pt_model.forward_step(psi_curr, pt_x[:, t])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt_pt = time.time() - t0
        tok_per_sec_pt = (BATCH_SIZE * SEQ_LEN * N_RUNS) / dt_pt
        print(f"    -> Time: {dt_pt:.3f}s | Throughput: {tok_per_sec_pt:,.0f} Tokens/s")

        # 2. MLX Benchmark
        print(f"\n[MLX] Benchmarking GAPU Kernel (Metal)...")
        # MLX Kernel works on (B, D, 32) states?
        # kernel.compiled_rotor_scan loops over dim 0 of inputs.
        # inputs needs to be (SEQ_LEN, B, D, 32) if we want to loop over time.
        # The kernel.py loop: for i in range(inputs.shape[0]).
        
        mx_psi = mx.array(pt_psi.cpu().numpy()) # (B, D, 32)
        # Transpose inputs to (SEQ_LEN, B, D, 32) for the scan loop
        mx_x = mx.array(pt_x.cpu().numpy()).transpose(1, 0, 2, 3) 
        
        # Warmup
        _ = kernel.compiled_rotor_scan(mx_psi, mx_x)
        mx.eval(_)
        
        t0 = time.time()
        for _ in range(N_RUNS):
            out = kernel.compiled_rotor_scan(mx_psi, mx_x)
            mx.eval(out)
        dt_mx = time.time() - t0
        tok_per_sec_mx = (BATCH_SIZE * SEQ_LEN * N_RUNS) / dt_mx
        print(f"    -> Time: {dt_mx:.3f}s | Throughput: {tok_per_sec_mx:,.0f} Tokens/s")
        
        print(f"\n>>> SPEEDUP (MLX vs PyTorch): {tok_per_sec_mx / tok_per_sec_pt:.2f}x")
        
    except ImportError:
        print("[SKIP] MLX not installed. Install via `pip install mlx` to run Metal benchmarks.")
    except Exception as e:
        print(f"[ERROR] Failed to run MLX benchmark: {e}")
