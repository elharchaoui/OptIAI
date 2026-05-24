"""
Monarch Projection on GPT-2 small
==================================
Key question: Does block-level rank-1 decomposition (Monarch structure)
preserve perplexity better than global SVD/ASVD at the same parameter budget?

New idea: Activation-Aware Monarch
  Scale W columns by ASVD-style input statistics, apply Monarch projection,
  then de-scale. Combines ASVD's input-awareness with Monarch's parallel structure.
  Not proposed in any of the papers — original combination.

Monarch projection algorithm (from Monarch paper, Algorithm 1):
  For W (out × in), choose n_blocks dividing both dims:
    p = out // n_blocks,  q = in // n_blocks
    Reshape W → W_4d of shape (n_blocks, p, n_blocks, q)
    W_4d[l, j, k, i] = W[l*p + j, k*q + i]

    For each (j, k) in [p] × [n_blocks]:
      W_4d[:, j, k, :] is a (n_blocks × q) matrix → rank-1 SVD → u_{jk}, v_{jk}

    L_3d[l, j, k] = σ_{jk} · u_{jk}[l]   (L's l-th block is a (p × n_blocks) matrix)
    R_3d[k, j, i] = v_{jk}[i]             (R's k-th block is a (p × q) matrix)

    W_approx[l*p+j, k*q+i] = L_3d[l, j, k] · R_3d[k, j, i]

Parallelism at inference:
  Step 1: y[b, k, j] = Σ_i R_3d[k, j, i] · x[b, k, i]   — all k blocks in parallel
  Step 2: z[b, l, j] = Σ_k L_3d[l, j, k] · y[b, k, j]   — all l blocks in parallel
  Only 2 sequential stages vs 1 dense matmul — but each stage is fully parallel.

Parameter count comparison (for W: out × in, n = n_blocks):
  Dense:   out × in
  SVD r:   r × (out + in)
  Monarch: out × n_blocks + out × (in/n_blocks) = out × (n_blocks + in/n_blocks)
           minimum at n_blocks = √in → 2 × out × √in  (Monarch achieves O(n^1.5) instead of O(n²))
"""

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).eval().to(DEVICE)
N_LAYERS = len(model.transformer.h)

CALIB_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths and stories of "
    "artificial beings endowed with intelligence by craftsmen. The seeds of modern AI were "
    "planted by philosophers who described human thinking as mechanical symbol manipulation.",
    "Machine learning uses statistical techniques to give systems the ability to learn from "
    "data without being explicitly programmed. The name was coined in 1959 by Arthur Samuel.",
    "The transformer architecture was introduced in Attention Is All You Need by Vaswani et al "
    "in 2017. It has become the dominant architecture for large language models.",
    "Deep learning is part of a family of machine learning methods based on artificial neural "
    "networks. Learning can be supervised, semi-supervised, or unsupervised.",
    "Natural language processing focuses on interactions between computers and human language, "
    "how to program computers to process and analyze large amounts of natural language data.",
    "The development of large language models has been driven by advances in hardware, "
    "specifically graphics processing units designed for rendering video games.",
]
calib_inputs = [
    tokenizer(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(DEVICE)
    for t in CALIB_TEXTS
]

EVAL_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by master "
    "craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe "
    "the process of human thinking as the mechanical manipulation of symbols. This work "
    "culminated in the invention of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning. This device and the ideas behind "
    "it inspired a handful of scientists to begin seriously discussing the possibility of "
    "building an electronic brain. The field of AI research was founded at a workshop held "
    "on the campus of Dartmouth College during the summer of 1956."
)
eval_ids = tokenizer(EVAL_TEXT, return_tensors="pt",
                     truncation=True, max_length=512)["input_ids"].to(DEVICE)


# ── Core helpers ───────────────────────────────────────────────────────────────
def compute_ppl(mdl):
    with torch.no_grad():
        out = mdl(eval_ids, labels=eval_ids)
    return torch.exp(out.loss).item()


def param_count(out_dim, in_dim, n_blocks):
    """Parameters used by Monarch vs SVD for a single weight matrix."""
    monarch = out_dim * n_blocks + out_dim * (in_dim // n_blocks)
    return monarch


def equiv_svd_rank(out_dim, in_dim, n_blocks):
    """SVD rank with same parameter count as Monarch with n_blocks."""
    return param_count(out_dim, in_dim, n_blocks) // (out_dim + in_dim)


def monarch_proj(W_std, n_blocks):
    """
    Project W_std (out, in) onto the set of Monarch matrices.
    Returns W_approx (out, in) with the same shape.

    The projection is optimal in Frobenius norm (analogous to Eckart-Young for SVD)
    but at the block level: each cross-block slice W_4d[:, j, k, :] is approximated rank-1.
    """
    out_dim, in_dim = W_std.shape
    assert out_dim % n_blocks == 0 and in_dim % n_blocks == 0, \
        f"n_blocks={n_blocks} must divide both out={out_dim} and in={in_dim}"
    p = out_dim // n_blocks   # block height (output side)
    q = in_dim  // n_blocks   # block width  (input side)

    # Reshape: W_4d[l, j, k, i] = W_std[l*p+j, k*q+i]
    W_4d = W_std.float().reshape(n_blocks, p, n_blocks, q)

    # Batch all (j, k) pairs → shape (p*n_blocks, n_blocks, q)
    # After permute(1, 2, 0, 3): shape (p, n_blocks, n_blocks, q)
    # → reshape (p*n_blocks, n_blocks, q): batch_idx = j*n_blocks + k
    W_bat = W_4d.permute(1, 2, 0, 3).reshape(p * n_blocks, n_blocks, q)

    # Batch rank-1 SVD (full_matrices=False for efficiency)
    U, S, Vh = torch.linalg.svd(W_bat, full_matrices=False)
    # U:  (p*n_blocks, n_blocks, min(n_blocks,q))
    # S:  (p*n_blocks, min(n_blocks,q))
    # Vh: (p*n_blocks, min(n_blocks,q), q)

    # Rank-1 approximation per batch: absorb σ into u
    u = U[:, :, 0] * S[:, 0:1]   # (p*n_blocks, n_blocks)  — u_{jk} scaled by σ_{jk}
    v = Vh[:, 0, :]               # (p*n_blocks, q)          — v_{jk}

    # Assemble factor tensors
    u_3d = u.reshape(p, n_blocks, n_blocks)    # u_3d[j, k, l] = σ_{jk}·ũ_{jk}[l]
    v_3d = v.reshape(p, n_blocks, q)           # v_3d[j, k, i] = ṽ_{jk}[i]

    L_3d = u_3d.permute(2, 0, 1)              # L_3d[l, j, k] = u_3d[j, k, l]  (n_blocks, p, n_blocks)
    R_3d = v_3d.permute(1, 0, 2)              # R_3d[k, j, i] = v_3d[j, k, i]  (n_blocks, p, q)

    # Reconstruct: W_4d_approx[l, j, k, i] = L_3d[l, j, k] · R_3d[k, j, i]
    W_4d_approx = torch.einsum('ljk,kji->ljki', L_3d, R_3d)   # (n_blocks, p, n_blocks, q)
    return W_4d_approx.reshape(out_dim, in_dim).to(W_std.dtype)


def asvd_monarch(W_std, act_flat, n_blocks, alpha=0.5):
    """
    Activation-Aware Monarch: scale W columns by input statistics,
    apply Monarch projection, then de-scale. Novel combination not in any paper.
    """
    s = act_flat.float().abs().mean(0).pow(alpha).clamp(min=1e-8)  # (in,)
    W_scaled = W_std.float() * s.unsqueeze(0)
    W_approx_scaled = monarch_proj(W_scaled, n_blocks)
    return (W_approx_scaled / s.unsqueeze(0)).to(W_std.dtype)


def naive_svd(W_std, rank):
    U, S, Vh = torch.linalg.svd(W_std.float(), full_matrices=False)
    k = min(rank, len(S))
    return ((U[:, :k] * S[:k]) @ Vh[:k, :]).to(W_std.dtype)


def asvd(W_std, act_flat, rank, alpha=0.5):
    s = act_flat.float().abs().mean(0).pow(alpha).clamp(min=1e-8)
    U, S, Vh = torch.linalg.svd(W_std.float() * s.unsqueeze(0), full_matrices=False)
    k = min(rank, len(S))
    return (((U[:, :k] * S[:k]) @ Vh[:k, :]) / s.unsqueeze(0)).to(W_std.dtype)


# ── Collect calibration activations ───────────────────────────────────────────
print("\nCollecting calibration activations...")
calib_acts = {i: {"c_fc": [], "c_proj": []} for i in range(N_LAYERS)}
hooks = []
for i, block in enumerate(model.transformer.h):
    def _fc(idx):
        def h(m, inp, out): calib_acts[idx]["c_fc"].append(inp[0].detach().cpu().float())
        return h
    def _proj(idx):
        def h(m, inp, out): calib_acts[idx]["c_proj"].append(inp[0].detach().cpu().float())
        return h
    hooks += [block.mlp.c_fc.register_forward_hook(_fc(i)),
              block.mlp.c_proj.register_forward_hook(_proj(i))]
with torch.no_grad():
    for inp in calib_inputs:
        model(inp)
for h in hooks:
    h.remove()
for i in range(N_LAYERS):
    for k in ("c_fc", "c_proj"):
        calib_acts[i][k] = torch.cat([a.flatten(0, 1) for a in calib_acts[i][k]], dim=0)

baseline_ppl = compute_ppl(model)
print(f"Baseline perplexity: {baseline_ppl:.2f}")

# ── Print parameter budget comparison ─────────────────────────────────────────
print("\nParameter budget comparison (c_fc layer: 3072×768):")
out_d, in_d = 3072, 768
print(f"  {'Method':30s}  {'Params':>10s}  {'% of dense':>10s}  {'Equiv SVD rank':>14s}")
print(f"  {'Dense':30s}  {out_d*in_d:>10,d}  {'100.0%':>10s}  {'full':>14s}")
for nb in [8, 16, 32]:
    mp = param_count(out_d, in_d, nb)
    er = equiv_svd_rank(out_d, in_d, nb)
    print(f"  {f'Monarch n_blocks={nb}':30s}  {mp:>10,d}  {mp/(out_d*in_d)*100:>9.1f}%  {er:>14d}")
for r in [64, 128, 256, 512]:
    rp = r * (out_d + in_d)
    print(f"  {f'SVD/ASVD rank={r}':30s}  {rp:>10,d}  {rp/(out_d*in_d)*100:>9.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Monarch vs ASVD vs Naive SVD — full model compression
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 1 — Monarch projection vs ASVD vs Naive SVD (all MLP layers)")
print("=" * 60)

N_BLOCKS_LIST = [8, 16, 32]

# Compute equivalent SVD ranks for fair comparison
# (using c_fc shape 3072×768 as representative)
equiv_ranks = [equiv_svd_rank(3072, 768, nb) for nb in N_BLOCKS_LIST]
print(f"\n  Monarch n_blocks  : {N_BLOCKS_LIST}")
print(f"  Equiv. SVD ranks  : {equiv_ranks}")

monarch_ppls      = []
aa_monarch_ppls   = []
naive_ppls_equiv  = []
asvd_ppls_equiv   = []

for nb, eq_rank in zip(N_BLOCKS_LIST, equiv_ranks):
    # ── Monarch ───────────────────────────────────────────────────────────────
    m = copy.deepcopy(model)
    for li, block in enumerate(m.transformer.h):
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            W_lr  = monarch_proj(W_std, nb)
            conv.weight.data = W_lr.T.cpu().contiguous()
    m.to(DEVICE)
    mp = compute_ppl(m)
    monarch_ppls.append(mp)
    del m; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── Activation-Aware Monarch ───────────────────────────────────────────────
    m = copy.deepcopy(model)
    for li, block in enumerate(m.transformer.h):
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            acts  = calib_acts[li][attr].to(DEVICE)
            W_lr  = asvd_monarch(W_std, acts, nb)
            conv.weight.data = W_lr.T.cpu().contiguous()
    m.to(DEVICE)
    aap = compute_ppl(m)
    aa_monarch_ppls.append(aap)
    del m; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── Naive SVD at equivalent rank ───────────────────────────────────────────
    m = copy.deepcopy(model)
    for block in m.transformer.h:
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            r = min(eq_rank, min(W_std.shape))
            conv.weight.data = naive_svd(W_std, r).T.cpu().contiguous()
    m.to(DEVICE)
    np_ = compute_ppl(m)
    naive_ppls_equiv.append(np_)
    del m; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── ASVD at equivalent rank ────────────────────────────────────────────────
    m = copy.deepcopy(model)
    for li, block in enumerate(m.transformer.h):
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            acts  = calib_acts[li][attr].to(DEVICE)
            r = min(eq_rank, min(W_std.shape))
            conv.weight.data = asvd(W_std, acts, r).T.cpu().contiguous()
    m.to(DEVICE)
    ap = compute_ppl(m)
    asvd_ppls_equiv.append(ap)
    del m; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    print(f"\n  n_blocks={nb}  (equiv SVD rank≈{eq_rank})")
    print(f"    Naive SVD:          ppl={np_:.2f}  ({np_/baseline_ppl:.1f}× baseline)")
    print(f"    ASVD:               ppl={ap:.2f}  ({ap/baseline_ppl:.1f}× baseline)")
    print(f"    Monarch:            ppl={mp:.2f}  ({mp/baseline_ppl:.1f}× baseline)")
    print(f"    AA-Monarch (new):   ppl={aap:.2f}  ({aap/baseline_ppl:.1f}× baseline)")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Per-layer sensitivity — Monarch n_blocks=16 vs ASVD rank=32
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 2 — Per-layer sensitivity: Monarch(n=16) vs ASVD(rank=32)")
print("=" * 60)

layer_monarch   = []
layer_aa_mon    = []

for li in range(N_LAYERS):
    for method, results_list in [("monarch", layer_monarch), ("aa_monarch", layer_aa_mon)]:
        m = copy.deepcopy(model)
        block = m.transformer.h[li]
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            if method == "monarch":
                W_lr = monarch_proj(W_std, n_blocks=16)
            else:
                acts = calib_acts[li][attr].to(DEVICE)
                W_lr = asvd_monarch(W_std, acts, n_blocks=16)
            conv.weight.data = W_lr.T.cpu().contiguous()
        m.to(DEVICE)
        ppl = compute_ppl(m)
        results_list.append(ppl)
        del m; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    print(f"  Layer {li:2d}:  monarch={layer_monarch[li]:.2f}  "
          f"aa_monarch={layer_aa_mon[li]:.2f}  "
          f"Δ(AA-Monarch)={layer_aa_mon[li]-layer_monarch[li]:+.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Reconstruction quality — weight error vs perplexity
# (do weight errors predict perplexity rank ordering?)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 3 — Weight reconstruction error comparison (layer 6, c_fc)")
print("=" * 60)

block6 = model.transformer.h[6]
W_fc6  = block6.mlp.c_fc.weight.detach().T.to(DEVICE)   # (3072, 768)
acts6  = calib_acts[6]["c_fc"].to(DEVICE)

def rel_err(W, W_a):
    return (torch.norm(W.float() - W_a.float()) / torch.norm(W.float())).item()

for nb in N_BLOCKS_LIST:
    eq_r = equiv_svd_rank(3072, 768, nb)
    e_svd    = rel_err(W_fc6, naive_svd(W_fc6, eq_r))
    e_asvd   = rel_err(W_fc6, asvd(W_fc6, acts6, eq_r))
    e_mon    = rel_err(W_fc6, monarch_proj(W_fc6, nb))
    e_aa_mon = rel_err(W_fc6, asvd_monarch(W_fc6, acts6, nb))
    print(f"  n_blocks={nb} (≈rank {eq_r}):  "
          f"svd={e_svd:.4f}  asvd={e_asvd:.4f}  "
          f"monarch={e_mon:.4f}  aa-monarch={e_aa_mon:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Monarch Projection on GPT-2 Small — Novel Activation-Aware Variant",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: PPL comparison at each n_blocks ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(N_BLOCKS_LIST))
w = 0.2
ax1.bar(x - 1.5*w, naive_ppls_equiv,   width=w, label="Naive SVD (equiv rank)", color="tomato",    alpha=0.85)
ax1.bar(x - 0.5*w, asvd_ppls_equiv,    width=w, label="ASVD (equiv rank)",       color="orange",    alpha=0.85)
ax1.bar(x + 0.5*w, monarch_ppls,        width=w, label="Monarch",                 color="steelblue", alpha=0.85)
ax1.bar(x + 1.5*w, aa_monarch_ppls,    width=w, label="AA-Monarch (new)",         color="green",     alpha=0.85)
ax1.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_ppl:.1f})")
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels([f"n_blocks={nb}\n(≈rank {er})" for nb, er in zip(N_BLOCKS_LIST, equiv_ranks)])
ax1.set_ylabel("Perplexity [log scale]")
ax1.set_title("Exp 1: All 4 methods at matching parameter budget")
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3, which="both")

# ── Plot 2: Ratio to baseline ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ratios_svd  = [p / baseline_ppl for p in naive_ppls_equiv]
ratios_asvd = [p / baseline_ppl for p in asvd_ppls_equiv]
ratios_mon  = [p / baseline_ppl for p in monarch_ppls]
ratios_aa   = [p / baseline_ppl for p in aa_monarch_ppls]
ax2.plot(N_BLOCKS_LIST, ratios_svd,  marker="o", color="tomato",    linewidth=2, label="Naive SVD (equiv rank)")
ax2.plot(N_BLOCKS_LIST, ratios_asvd, marker="s", color="orange",    linewidth=2, label="ASVD (equiv rank)")
ax2.plot(N_BLOCKS_LIST, ratios_mon,  marker="^", color="steelblue", linewidth=2, label="Monarch")
ax2.plot(N_BLOCKS_LIST, ratios_aa,   marker="D", color="green",     linewidth=2, label="AA-Monarch (new)")
ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Baseline (1×)")
ax2.axhline(2.0, color="gray",  linestyle=":",  alpha=0.6, label="2× threshold")
ax2.set_xlabel("n_blocks")
ax2.set_ylabel("PPL ratio vs baseline")
ax2.set_title("Exp 1: PPL degradation ratio (lower = better)")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3)

# ── Plot 3: Per-layer sensitivity ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
layers_x = list(range(N_LAYERS))
ax3.plot(layers_x, layer_monarch,   marker="^", color="steelblue", linewidth=2, label="Monarch n=16")
ax3.plot(layers_x, layer_aa_mon,    marker="D", color="green",     linewidth=2, label="AA-Monarch n=16")
ax3.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax3.set_xlabel("Layer index (0=first, 11=last)")
ax3.set_ylabel("Perplexity (one layer compressed)")
ax3.set_title("Exp 2: Per-layer sensitivity — Monarch n_blocks=16")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Plot 4: Weight reconstruction error ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
err_svd  = [rel_err(W_fc6, naive_svd(W_fc6, equiv_svd_rank(3072, 768, nb))) for nb in N_BLOCKS_LIST]
err_asvd = [rel_err(W_fc6, asvd(W_fc6, acts6, equiv_svd_rank(3072, 768, nb))) for nb in N_BLOCKS_LIST]
err_mon  = [rel_err(W_fc6, monarch_proj(W_fc6, nb)) for nb in N_BLOCKS_LIST]
err_aa   = [rel_err(W_fc6, asvd_monarch(W_fc6, acts6, nb)) for nb in N_BLOCKS_LIST]
ax4.plot(N_BLOCKS_LIST, err_svd,  marker="o", color="tomato",    linewidth=2, label="Naive SVD")
ax4.plot(N_BLOCKS_LIST, err_asvd, marker="s", color="orange",    linewidth=2, label="ASVD")
ax4.plot(N_BLOCKS_LIST, err_mon,  marker="^", color="steelblue", linewidth=2, label="Monarch")
ax4.plot(N_BLOCKS_LIST, err_aa,   marker="D", color="green",     linewidth=2, label="AA-Monarch")
ax4.set_xlabel("n_blocks")
ax4.set_ylabel("Relative weight reconstruction error")
ax4.set_title("Exp 3: Weight error (layer 6, c_fc, 3072×768)")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.savefig("results_monarch.png", dpi=150, bbox_inches="tight")
print("\nSaved results_monarch.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline: {baseline_ppl:.2f}")

best_method_per_nb = []
for i, nb in enumerate(N_BLOCKS_LIST):
    scores = {
        "Naive SVD":  naive_ppls_equiv[i],
        "ASVD":       asvd_ppls_equiv[i],
        "Monarch":    monarch_ppls[i],
        "AA-Monarch": aa_monarch_ppls[i],
    }
    best = min(scores, key=scores.get)
    best_method_per_nb.append((nb, best, scores[best]))
    print(f"  n_blocks={nb}: best={best} ppl={scores[best]:.2f} "
          f"({scores[best]/baseline_ppl:.1f}× baseline)")

print(f"\nAA-Monarch improvement over plain Monarch:")
for i, nb in enumerate(N_BLOCKS_LIST):
    delta = monarch_ppls[i] - aa_monarch_ppls[i]
    print(f"  n_blocks={nb}: Δ={delta:+.2f} ({'better' if delta > 0 else 'worse'})")

print(f"\nAA-Monarch improvement over ASVD (same parameter budget):")
for i, nb in enumerate(N_BLOCKS_LIST):
    delta = asvd_ppls_equiv[i] - aa_monarch_ppls[i]
    print(f"  n_blocks={nb}: Δ={delta:+.2f} ({'better' if delta > 0 else 'worse'})")
