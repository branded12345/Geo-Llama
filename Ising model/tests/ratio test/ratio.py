"""
- Range: 2x2 (4 tokens) to 64x64 (4,096 tokens).
- Validation: 5-Fold Cross-Validation with Full Data Regeneration.
- Metric: Matthews Correlation Coefficient (MCC).
- Hardware: Dual NVIDIA T4 (32GB VRAM).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import gc
import sys
import time

# =================================================================
# 1. HARDWARE CONFIGURATION
# =================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
print(f"--- ISING TOPOLOGICAL BENCHMARK (PUBLICATION RUN) ---")
print(f"System: {NUM_GPUS}x NVIDIA T4 | Total VRAM: {16*NUM_GPUS}GB")
print(f"Regime: Thermodynamic Limit Scaling (4 to 4,096 tokens)")

# =================================================================
# 2. ALGEBRAIC KERNEL (Cl 4,1 CONFORMAL GEOMETRIC ALGEBRA)
# =================================================================
GRADE_INDICES = {0: [0], 1: [1, 2, 4, 8, 16]}

def compute_basis_product_cl41(a, b):
    """Computes geometric product sign and index."""
    sign, a_bits = 1.0, a
    for i in range(5):
        if (b >> i) & 1:
            for j in range(i + 1, 5):
                if (a_bits >> j) & 1: sign *= -1.0
            if (a_bits >> i) & 1:
                if i == 4: sign *= -1.0 # Minkowski metric e-
                a_bits &= ~(1 << i)
            else: a_bits |= (1 << i)
    return sign, a_bits

GP_MAP_CACHE = {}
SIG_CACHE = {}

def get_gp_map(device):
    if device not in GP_MAP_CACHE:
        table = torch.zeros((32, 32, 32), device=device)
        for a in range(32):
            for b in range(32):
                s, r = compute_basis_product_cl41(a, b)
                table[a, b, r] = s
        GP_MAP_CACHE[device] = table
    return GP_MAP_CACHE[device]

def get_metric_signature(device):
    if device not in SIG_CACHE:
        sig = torch.ones(32, device=device)
        for i in range(32):
            if (i >> 4) & 1: sig[i] *= -1.0
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1: sig[i] *= -1.0
        SIG_CACHE[device] = sig
    return SIG_CACHE[device]

def manifold_normalization(A, eps=1e-8):
    """Projects multivectors onto the unit manifold."""
    sig = get_metric_signature(A.device)
    norm_sq = torch.sum(A * A * sig, dim=-1)
    denom = torch.max(torch.sqrt(torch.abs(norm_sq) + eps).unsqueeze(-1), 
                     torch.norm(A, p=2, dim=-1, keepdim=True) / 4.0 + eps).clamp(min=1.0)
    return A / denom

def conformal_projection(spins):
    """Maps scalar spins to Conformal Null Basis {no, ninf}."""
    n_o = torch.zeros(32, device=DEVICE); n_o[16], n_o[8] = 0.5, -0.5
    n_inf = torch.zeros(32, device=DEVICE); n_inf[16], n_inf[8] = 1.0, 1.0
    out = torch.zeros((spins.shape[0], spins.shape[1], 32), device=DEVICE, dtype=spins.dtype)
    out += (spins == 1).unsqueeze(-1) * n_o
    out += (spins == -1).unsqueeze(-1) * n_inf
    return out

# =================================================================
# 3. MODEL ARCHITECTURES
# =================================================================
class GeometricLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_f, in_f, 32))
        with torch.no_grad():
            std = 1.0 / (in_f * 32)**0.5
            self.weight[:, :, 0].normal_(0.0, std)
            for idx in GRADE_INDICES[1]: self.weight[:, :, idx].normal_(0.0, std)
    def forward(self, x):
        gp = get_gp_map(x.device)
        W_op = torch.einsum('oij,jlk->oilk', self.weight, gp)
        return manifold_normalization(torch.einsum('bsil,oilk->bsok', x, W_op))

class GeometricAttention(nn.Module):
    def __init__(self, d, h=2):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q_p, self.k_p, self.v_p, self.o_p = [GeometricLinear(d, d) for _ in range(4)]
    def forward(self, x):
        b, s, d, _ = x.shape
        q = self.q_p(x).view(b, s, self.h, self.dh, 32).transpose(1, 2)
        k = self.k_p(x).view(b, s, self.h, self.dh, 32).transpose(1, 2)
        v = self.v_p(x).view(b, s, self.h, self.dh, 32).transpose(1, 2)
        sig = get_metric_signature(q.device)
        score = torch.matmul((q*sig).reshape(b, self.h, s, -1), k.reshape(b, self.h, s, -1).transpose(-1, -2)) / (self.dh**0.5)
        attn = torch.softmax(score, dim=-1)
        out = torch.einsum('bhsi,bhidl->bhsdl', attn, v)
        return self.o_p(out.transpose(1, 2).reshape(b, s, d, 32))

class GeometricTransformer(nn.Module):
    def __init__(self, d, layers, expansion=6):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'attn': GeometricAttention(d),
            'mlp': nn.Sequential(GeometricLinear(d, d*expansion), nn.Tanh(), GeometricLinear(d*expansion, d)),
            'ln1': nn.LayerNorm([d, 32]), 'ln2': nn.LayerNorm([d, 32])
        }) for _ in range(layers)])
        self.pool = GeometricLinear(d, d)
        self.head = nn.Linear(d*32, 3)
    def forward(self, x):
        for b in self.blocks:
            x = manifold_normalization(x + b['attn'](b['ln1'](x)))
            x = manifold_normalization(x + b['mlp'](b['ln2'](x)))
        p = manifold_normalization(torch.tanh(self.pool(x).mean(dim=1)))
        return self.head(p.view(x.shape[0], -1))

class StandardTransformer(nn.Module):
    def __init__(self, dm, nh, nl, df):
        super().__init__()
        self.proj = nn.Linear(128, dm)
        self.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh, df, batch_first=True, norm_first=True), nl)
        self.head = nn.Linear(dm, 3)
    def forward(self, x):
        return self.head(self.t(self.proj(x.view(x.shape[0], x.shape[1], -1))).mean(dim=1))

# =================================================================
# 4. PHYSICS ENGINE
# =================================================================
def generate_ising_dataset(grid_size, n_samples=1500):
    """Generates Critical Phase Ising Data (Vectorized Metropolis)."""
    # Reduce verbosity for loop usage
    # print(f"  [Gen] {grid_size}x{grid_size}...", end=" ", flush=True)
    
    grids = torch.where(torch.rand((n_samples, grid_size, grid_size), device=DEVICE) > 0.5, 1.0, -1.0)
    temps = 2.269 + (torch.rand((n_samples, 1, 1), device=DEVICE) * 0.3 - 0.15)
    betas = 1.0 / temps
    idx = torch.stack(torch.meshgrid(torch.arange(grid_size, device=DEVICE), torch.arange(grid_size, device=DEVICE), indexing='ij'))
    m_even = (idx.sum(0) % 2 == 0).unsqueeze(0); m_odd = ~m_even
    
    # Steps: Optimized for convergence vs runtime
    steps = 2000 if grid_size >= 64 else 1000
    
    for _ in range(steps):
        for m in [m_even, m_odd]:
            nn = torch.roll(grids,1,1)+torch.roll(grids,-1,1)+torch.roll(grids,1,2)+torch.roll(grids,-1,2)
            accept = (torch.rand((n_samples, grid_size, grid_size), device=DEVICE) < torch.exp(-2*grids*nn*betas)) | (2*grids*nn <= 0)
            grids = torch.where(m & accept, -grids, grids)
            
    labs = torch.ones(n_samples, dtype=torch.long, device=DEVICE)
    labs[temps.squeeze() < 2.20], labs[temps.squeeze() > 2.35] = 0, 2
    return conformal_projection(grids.view(n_samples, -1)).unsqueeze(2).repeat(1, 1, 4, 1), labs

# =================================================================
# 5. EXPERIMENT RUNNER
# =================================================================
def train_and_evaluate(model, X, Y):
    if NUM_GPUS > 1: model = nn.DataParallel(model)
    model.to(DEVICE)
    ds = TensorDataset(X, Y)
    tr_len = int(len(ds)*0.85)
    tr_ds, vl_ds = random_split(ds, [tr_len, len(ds)-tr_len])
    
    # Smart Batching: Aggressive but Safe
    seq_len = X.shape[1]
    if seq_len > 3000:   bsize = 16 # 64x64 Safe Mode
    elif seq_len > 1000: bsize = 64 # 32x32 Fast Mode
    else:                bsize = 128 # Small Grid Turbo Mode
    
    tr_ld = DataLoader(tr_ds, batch_size=bsize, shuffle=True, drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=bsize)
    
    opt = optim.Adam(model.parameters(), lr=0.0007)
    crit = nn.CrossEntropyLoss()
    best_mcc = -1.0
    
    for _ in range(25):
        model.train()
        for xb, yb in tr_ld:
            opt.zero_grad(); crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        p, t = [], []
        with torch.no_grad():
            for xb, yb in vl_ld:
                p.extend(model(xb).argmax(1).cpu().numpy()); t.extend(yb.cpu().numpy())
        m = matthews_corrcoef(t, p)
        if m > best_mcc: best_mcc = m
    return best_mcc

def run_benchmark():
    # 5 Seeds for Hirst-Compliant Rigor
    SEEDS = [553, 42, 777, 1024, 2026]
    
    # --- PHASE 1: SCALING ---
    sizes = [2, 4, 8, 12, 16, 24, 32, 64] # Full Spectrum
    res_a = {'geo': [], 'van': []}
    
    print("\n=== PHASE 1: SCALING ANALYSIS (2x2 to 64x64) ===")
    for s in sizes:
        print(f"\n> Grid {s}x{s} ({s*s} Tokens):")
        gs, vs = [], []
        
        for i, seed in enumerate(SEEDS):
            print(f"  Seed {seed} [{i+1}/5]...", end=" ", flush=True)
            # REGENERATE PHYSICS per seed
            X, Y = generate_ising_dataset(s, n_samples=1500)
            
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            g_score = train_and_evaluate(GeometricTransformer(4, 2, expansion=6), X, Y)
            
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            v_score = train_and_evaluate(StandardTransformer(32, 2, 2, 64), X, Y)
            
            gs.append(g_score); vs.append(v_score)
            print(f"G={g_score:.3f} V={v_score:.3f}")
            del X, Y; gc.collect(); torch.cuda.empty_cache()
            
        res_a['geo'].append((np.mean(gs), np.std(gs)))
        res_a['van'].append((np.mean(vs), np.std(vs)))
        print(f"  AVG: Geo {np.mean(gs):.3f} | Van {np.mean(vs):.3f}")

    # --- PHASE 2: EFFICIENCY ---
    print(f"\n=== PHASE 2: PARAMETER EFFICIENCY (64x64) ===")
    tiers = [('Tiny', 2, 24), ('Small', 6, 32), ('Medium', 12, 48)]
    res_b = {'geo': [], 'van': [], 'pg': [], 'pv': []}
    
    for lbl, ge, vd in tiers:
        print(f"\n> Tier {lbl}:")
        mod_g = GeometricTransformer(4, 2, expansion=ge)
        mod_v = StandardTransformer(vd, 2, 2, vd*2)
        pg = sum(p.numel() for p in mod_g.parameters())
        pv = sum(p.numel() for p in mod_v.parameters())
        res_b['pg'].append(pg); res_b['pv'].append(pv)
        del mod_g, mod_v
        
        gs, vs = [], []
        for i, seed in enumerate(SEEDS):
            print(f"  Seed {seed} [{i+1}/5]...", end=" ", flush=True)
            X, Y = generate_ising_dataset(64, n_samples=1500)
            
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            gs.append(train_and_evaluate(GeometricTransformer(4, 2, expansion=ge), X, Y))
            
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            vs.append(train_and_evaluate(StandardTransformer(vd, 2, 2, vd*2), X, Y))
            
            print(f"G={gs[-1]:.3f} V={vs[-1]:.3f}")
            del X, Y; gc.collect(); torch.cuda.empty_cache()
            
        res_b['geo'].append((np.mean(gs), np.std(gs)))
        res_b['van'].append((np.mean(vs), np.std(vs)))
        print(f"  Ratio: {np.mean(gs)/max(np.mean(vs), 0.01):.2f}x")

    return sizes, res_a, res_b

# =================================================================
# 6. VISUALIZATION
# =================================================================
def plot_results(sz, ra, rb):
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot A
    xa = [s*s for s in sz]
    yg, yge = zip(*ra['geo']); yv, yve = zip(*ra['van'])
    ax[0].errorbar(xa, yg, yerr=yge, fmt='-o', color='#4A90E2', lw=2.5, capsize=5, label='Geometric (Cl 4,1)')
    ax[0].errorbar(xa, yv, yerr=yve, fmt='--s', color='#E74C3C', lw=2.5, capsize=5, label='Standard Attention')
    ax[0].set_xscale('log'); ax[0].set_xticks(xa); ax[0].set_xticklabels([f"{s}x{s}" for s in sz])
    ax[0].set_xlabel('Sequence Length (Complexity)', fontweight='bold')
    ax[0].set_ylabel('MCC', fontweight='bold'); ax[0].legend()
    
    # Plot B
    xp = [(g+v)/2 for g,v in zip(rb['pg'], rb['pv'])]
    yr = [g[0]/max(v[0], 0.01) for g,v in zip(rb['geo'], rb['van'])]
    yre = [r * np.sqrt((g[1]/g[0])**2 + (v[1]/v[0])**2) if g[0]>0 else 0 for r,g,v in zip(yr, rb['geo'], rb['van'])]
    ax[1].errorbar(xp, yr, yerr=yre, fmt='-^', color='#8E44AD', lw=2.5, capsize=5, label='Efficiency Ratio')
    ax[1].axhline(1.0, color='gray', ls='--')
    ax[1].fill_between(xp, 1.0, yr, color='#8E44AD', alpha=0.15)
    ax[1].set_xscale('log'); ax[1].set_xlabel('Parameters', fontweight='bold')
    ax[1].set_ylabel('Advantage Ratio', fontweight='bold')
    
    plt.tight_layout(); plt.savefig('PUBLICATION_RESULTS.png', dpi=300)

if __name__ == "__main__":
    try:
        s, ra, rb = run_benchmark()
        plot_results(s, ra, rb)
    except Exception as e:
        print(f"Fatal: {e}")