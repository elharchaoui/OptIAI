"""
Scientific evaluation of the ideas from idea-discussion.md:
  1. Low-rank (SVD) approximation of W
  2. Binary gate as a stand-in for nonlinear activation (your core insight)
  3. SVD + binary gate combined
  4. Error accumulation across L stacked layers

Protocol
--------
- Ground truth: Y = relu(X @ W.T)
- All comparisons at fixed seed for reproducibility
- Metrics: relative L2 error, cosine similarity, wall-clock time
- Sweep: rank ∈ {2,4,8,16,32,64}, gate threshold percentile ∈ {0,25,50,75}
- Layer-stacking: L ∈ {1,2,4,8,12} layers
"""

import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)

# ── constants ────────────────────────────────────────────────────────────────
DIM      = 128      # matrix size (n×n)
N_SAMPLE = 512      # input samples
N_RUNS   = 200      # timing repetitions
RANKS    = [2, 4, 8, 16, 32, 64, 128]
PCTS     = [0, 25, 50, 75]   # gate threshold percentile (0 = plain relu)


# ── helpers ──────────────────────────────────────────────────────────────────
def make_layer():
    return torch.randn(DIM, DIM) / DIM**0.5

def low_rank(W, rank):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank, :]

def forward_true(X, W):
    return torch.relu(X @ W.T)

def forward_lowrank(X, W_lr):
    return torch.relu(X @ W_lr.T)

def forward_gate(X, W, pct):
    Z = X @ W.T
    if pct == 0:
        return torch.relu(Z)                          # baseline
    thresh = torch.quantile(Z, pct / 100.0)
    return Z * (Z > thresh).float()                   # binary gate

def forward_svd_gate(X, W_lr, pct):
    Z = X @ W_lr.T
    if pct == 0:
        return torch.relu(Z)
    thresh = torch.quantile(Z, pct / 100.0)
    return Z * (Z > thresh).float()

def metrics(Y_true, Y_approx):
    rel_err  = (torch.norm(Y_true - Y_approx) / torch.norm(Y_true)).item()
    cos_sim  = torch.nn.functional.cosine_similarity(
        Y_true.flatten().unsqueeze(0),
        Y_approx.flatten().unsqueeze(0)
    ).item()
    sparsity = (Y_approx == 0).float().mean().item()
    return rel_err, cos_sim, sparsity

def timeit(fn, n=N_RUNS):
    for _ in range(20):   # warmup
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


# ── setup ────────────────────────────────────────────────────────────────────
W = make_layer()
X = torch.randn(N_SAMPLE, DIM)
Y_true = forward_true(X, W)

# precompute low-rank approximations
W_lrs = {r: low_rank(W, r) for r in RANKS}

t_baseline = timeit(lambda: forward_true(X, W))
print(f"Baseline time: {t_baseline*1e6:.1f} µs\n")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Low-rank only
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXP 1 — Low-rank approximation (relu stays)")
print("=" * 60)
lr_results = []
for r in RANKS:
    Y_a = forward_lowrank(X, W_lrs[r])
    re, cs, sp = metrics(Y_true, Y_a)
    t = timeit(lambda W_lr=W_lrs[r]: forward_lowrank(X, W_lr))
    speedup = t_baseline / t
    lr_results.append(dict(rank=r, rel_err=re, cos_sim=cs,
                           sparsity=sp, time_us=t*1e6, speedup=speedup))
    print(f"  rank={r:3d}  rel_err={re:.4f}  cos={cs:.4f}  "
          f"sparsity={sp:.3f}  speedup={speedup:.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Full W + binary gate (your core idea)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 2 — Binary gate replacing relu (full W)")
print("=" * 60)
gate_results = []
for pct in PCTS:
    Y_a = forward_gate(X, W, pct)
    re, cs, sp = metrics(Y_true, Y_a)
    t = timeit(lambda p=pct: forward_gate(X, W, p))
    speedup = t_baseline / t
    gate_results.append(dict(pct=pct, rel_err=re, cos_sim=cs,
                             sparsity=sp, time_us=t*1e6, speedup=speedup))
    label = "relu (baseline)" if pct == 0 else f"gate@p{pct}"
    print(f"  {label:20s}  rel_err={re:.4f}  cos={cs:.4f}  "
          f"sparsity={sp:.3f}  speedup={speedup:.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: SVD + binary gate (combined)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 3 — SVD + binary gate (combined)")
print("=" * 60)
combined_results = []
for r in RANKS:
    for pct in PCTS:
        Y_a = forward_svd_gate(X, W_lrs[r], pct)
        re, cs, sp = metrics(Y_true, Y_a)
        t = timeit(lambda W_lr=W_lrs[r], p=pct: forward_svd_gate(X, W_lr, p))
        speedup = t_baseline / t
        combined_results.append(dict(rank=r, pct=pct, rel_err=re,
                                     cos_sim=cs, sparsity=sp,
                                     time_us=t*1e6, speedup=speedup))
        if pct == 0:   # only print the relu baseline per rank
            print(f"  rank={r:3d}  pct={pct}  rel_err={re:.4f}  "
                  f"cos={cs:.4f}  speedup={speedup:.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Error accumulation across L stacked layers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 4 — Error accumulation (12 stacked layers)")
print("=" * 60)
L_vals = [1, 2, 4, 8, 12]
stack_ranks = [16, 32, 64]

def stack_true(X_in, layers):
    out = X_in
    for W in layers:
        out = forward_true(out, W)
    return out

def stack_approx(X_in, layers_lr, pct=0):
    out = X_in
    for W_lr in layers_lr:
        out = forward_svd_gate(out, W_lr, pct)
    return out

stack_results = []
for L in L_vals:
    layers    = [make_layer() for _ in range(L)]
    Y_stack_t = stack_true(X, layers)
    for r in stack_ranks:
        layers_lr = [low_rank(W_l, r) for W_l in layers]
        Y_stack_a = stack_approx(X, layers_lr, pct=0)
        re, cs, sp = metrics(Y_stack_t, Y_stack_a)
        stack_results.append(dict(L=L, rank=r, rel_err=re, cos_sim=cs))
        print(f"  L={L:2d}  rank={r:3d}  rel_err={re:.4f}  cos={cs:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Optimization Idea — Scientific Evaluation", fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: Low-rank Pareto frontier ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
errs   = [r['rel_err']  for r in lr_results]
speeds = [r['speedup']  for r in lr_results]
ax1.scatter(speeds, errs, c=RANKS, cmap='plasma', s=80, zorder=3)
for r, s, e in zip(RANKS, speeds, errs):
    ax1.annotate(f"r={r}", (s, e), textcoords="offset points",
                 xytext=(4, 4), fontsize=8)
ax1.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='err=0.10 threshold')
ax1.set_xlabel("Speedup vs baseline")
ax1.set_ylabel("Relative L2 error")
ax1.set_title("Exp 1: Low-rank Pareto frontier")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── Plot 2: Binary gate — sparsity vs error ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for pct, res in zip(PCTS, gate_results):
    label = "relu" if pct == 0 else f"gate@p{pct}"
    ax2.scatter(res['sparsity'], res['rel_err'], s=120,
                label=label, zorder=3)
ax2.set_xlabel("Sparsity (fraction zeros)")
ax2.set_ylabel("Relative L2 error")
ax2.set_title("Exp 2: Binary gate — sparsity vs error")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Plot 3: Combined — all (rank, pct) on one Pareto plot ────────────────────
ax3 = fig.add_subplot(gs[1, 0])
markers = {0: 'o', 25: 's', 50: '^', 75: 'D'}
colors  = plt.cm.tab10(np.linspace(0, 1, len(RANKS)))
for pct in PCTS:
    xs = [r['speedup']  for r in combined_results if r['pct'] == pct]
    ys = [r['rel_err']  for r in combined_results if r['pct'] == pct]
    label = "relu" if pct == 0 else f"gate@p{pct}"
    ax3.plot(xs, ys, marker=markers[pct], label=label, linewidth=1.2)
ax3.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='err=0.10')
ax3.set_xlabel("Speedup vs baseline")
ax3.set_ylabel("Relative L2 error")
ax3.set_title("Exp 3: SVD + gate combined Pareto")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Plot 4: Error accumulation across layers ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for r in stack_ranks:
    Ls  = [x['L']       for x in stack_results if x['rank'] == r]
    res = [x['rel_err'] for x in stack_results if x['rank'] == r]
    ax4.plot(Ls, res, marker='o', label=f"rank={r}")
ax4.axhline(0.1, color='red', linestyle='--', alpha=0.6, label='err=0.10')
ax4.set_xlabel("Number of stacked layers")
ax4.set_ylabel("Relative L2 error (cumulative)")
ax4.set_title("Exp 4: Error accumulation across L layers")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.savefig("results.png", dpi=150, bbox_inches='tight')
print("\nSaved results.png")

# ── Summary verdict ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best_lr = min(lr_results, key=lambda r: r['rel_err'] + (1 - r['speedup']/max(x['speedup'] for x in lr_results)))
print(f"\nBest low-rank:  rank={best_lr['rank']}  "
      f"rel_err={best_lr['rel_err']:.4f}  speedup={best_lr['speedup']:.2f}x")

best_comb = min((r for r in combined_results if r['rel_err'] < 0.1),
                key=lambda r: -r['speedup'], default=None)
if best_comb:
    print(f"Best combined:  rank={best_comb['rank']}  pct={best_comb['pct']}  "
          f"rel_err={best_comb['rel_err']:.4f}  speedup={best_comb['speedup']:.2f}x")
else:
    print("No combined config achieved rel_err < 0.10")

last_stack = [x for x in stack_results if x['L'] == 12]
ok = [x for x in last_stack if x['rel_err'] < 0.5]
print(f"\nAt 12 layers:  {len(ok)}/{len(last_stack)} rank configs "
      f"stay below rel_err=0.50")
