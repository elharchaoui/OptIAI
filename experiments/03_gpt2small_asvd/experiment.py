"""
ASVD: Activation-aware SVD on GPT-2 small
==========================================
Implements the key insight from the ASVD paper (arXiv:2312.05821):
  Instead of minimizing ||W - W_k||  (weight error),
  minimize ||W·X - W_k·X|| (output error on real data).

Method for each weight W (out×in):
  1. Collect calibration inputs X — shape (N, in) via forward hooks
  2. s_j = mean_i(|X_ij|)^alpha  — per-channel importance scaling
  3. W_scaled = W @ diag(s)      — scale columns by channel importance
  4. Truncated SVD: W_scaled ≈ U_k Σ_k V_k^T
  5. W_approx = U_k Σ_k V_k^T @ diag(1/s)  — de-scale

This concentrates singular values on directions that matter for real inputs,
allowing more aggressive rank reduction without perplexity explosion.

Experiments:
  1. Naive SVD vs ASVD perplexity across ranks
  2. Alpha (scaling exponent) sensitivity
  3. Per-layer sensitivity — which layers tolerate compression best?
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
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
model.eval().to(DEVICE)
N_LAYERS = len(model.transformer.h)

# ── Calibration corpus ─────────────────────────────────────────────────────────
# Diverse texts give better activation statistics than a single passage
CALIB_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths and stories "
    "of artificial beings endowed with intelligence by craftsmen. The seeds of modern AI "
    "were planted by philosophers who described human thinking as mechanical manipulation "
    "of symbols.",

    "Machine learning is a branch of artificial intelligence that uses statistical "
    "techniques to give computer systems the ability to learn from data without being "
    "explicitly programmed. The name machine learning was coined in 1959 by Arthur Samuel.",

    "Natural language processing is a subfield of linguistics and computer science that "
    "focuses on the interactions between computers and human language, in particular how "
    "to program computers to process and analyze large amounts of natural language data.",

    "The transformer architecture was introduced in the paper Attention Is All You Need "
    "by Vaswani et al in 2017. It has become the dominant architecture for large language "
    "models and has enabled significant advances in natural language understanding.",

    "Deep learning is part of a broader family of machine learning methods based on "
    "artificial neural networks with representation learning. Learning can be supervised, "
    "semi-supervised, or unsupervised. Deep learning architectures have been applied to "
    "fields including computer vision, speech recognition, and natural language processing.",

    "The development of large language models has been driven by advances in hardware, "
    "specifically graphics processing units originally designed for rendering video games. "
    "These parallel processors proved well suited for the matrix operations that underlie "
    "neural network training.",
]

calib_inputs = [
    tokenizer(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(DEVICE)
    for t in CALIB_TEXTS
]

# Evaluation text — same as experiment_gpt2.py for comparability
EVAL_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by master "
    "craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe "
    "the process of human thinking as the mechanical manipulation of symbols. This work "
    "culminated in the invention of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning. This device and the ideas behind "
    "it inspired a handful of scientists to begin seriously discussing the possibility of "
    "building an electronic brain. The field of AI research was founded at a workshop held "
    "on the campus of Dartmouth College during the summer of 1956. Those who attended would "
    "become the leaders of AI research for decades. Many of them predicted that a machine as "
    "intelligent as a human being would exist in no more than a generation."
)
eval_ids = tokenizer(EVAL_TEXT, return_tensors="pt",
                     truncation=True, max_length=512)["input_ids"].to(DEVICE)


# ── Core helpers ───────────────────────────────────────────────────────────────
def compute_perplexity(mdl):
    with torch.no_grad():
        out = mdl(eval_ids, labels=eval_ids)
    return torch.exp(out.loss).item()


def naive_svd(W_std, rank):
    """Truncated SVD with no activation awareness. W_std is (out, in)."""
    U, S, Vh = torch.linalg.svd(W_std.float(), full_matrices=False)
    k = min(rank, len(S))
    return (U[:, :k] * S[:k]) @ Vh[:k, :]


def asvd(W_std, act_flat, rank, alpha=0.5):
    """
    Activation-aware SVD.
    W_std:    (out, in)  — weight in standard convention
    act_flat: (N,  in)  — calibration activations, N = batch * seq_len
    rank:     target rank
    alpha:    scaling exponent (0.5 recommended by ASVD paper)
    """
    s = act_flat.float().abs().mean(dim=0).pow(alpha).clamp(min=1e-8)  # (in,)
    W_scaled = W_std.float() * s.unsqueeze(0)                          # (out, in)
    U, Sv, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
    k = min(rank, len(Sv))
    # Reconstruct and de-scale (divide each column j by s_j)
    return (U[:, :k] * Sv[:k]) @ Vh[:k, :] / s.unsqueeze(0)


# ── Collect activation statistics via forward hooks ────────────────────────────
print("Collecting calibration activations...")
calib_acts = {i: {"c_fc": [], "c_proj": []} for i in range(N_LAYERS)}
hooks = []

for i, block in enumerate(model.transformer.h):
    def _fc_hook(idx):
        def hook(module, inp, out):
            calib_acts[idx]["c_fc"].append(inp[0].detach().cpu().float())
        return hook

    def _proj_hook(idx):
        def hook(module, inp, out):
            calib_acts[idx]["c_proj"].append(inp[0].detach().cpu().float())
        return hook

    hooks.append(block.mlp.c_fc.register_forward_hook(_fc_hook(i)))
    hooks.append(block.mlp.c_proj.register_forward_hook(_proj_hook(i)))

with torch.no_grad():
    for inp in calib_inputs:
        model(inp)

for h in hooks:
    h.remove()

# Flatten (batch, seq, features) → (N, features)
for i in range(N_LAYERS):
    for key in ("c_fc", "c_proj"):
        # Each entry is (1, seq_len, features) with varying seq_len — flatten first
        calib_acts[i][key] = torch.cat(
            [a.flatten(0, 1) for a in calib_acts[i][key]], dim=0
        )

print(f"  c_fc  activations: {calib_acts[0]['c_fc'].shape}  "
      f"(layer 0, {len(CALIB_TEXTS)} texts)")
print(f"  c_proj activations: {calib_acts[0]['c_proj'].shape}")

# ── Baseline ───────────────────────────────────────────────────────────────────
baseline_ppl = compute_perplexity(model)
print(f"\nBaseline perplexity: {baseline_ppl:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Naive SVD vs ASVD — perplexity across ranks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 1 — Naive SVD vs ASVD perplexity (all 12 MLP layers)")
print("=" * 60)
print(f"{'rank':>6}  {'naive_ppl':>10}  {'asvd_ppl':>10}  {'Δppl':>8}")

RANKS = [32, 64, 96, 128, 192, 256, 384, 512]
naive_ppls = []
asvd_ppls  = []

for rank in RANKS:
    # Naive SVD: replace all MLP weights with truncated SVD
    m_naive = copy.deepcopy(model)
    for block in m_naive.transformer.h:
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)      # (out, in)
            W_lr  = naive_svd(W_std, min(rank, min(W_std.shape)))
            conv.weight.data = W_lr.T.cpu().contiguous()
    m_naive.to(DEVICE)
    nppl = compute_perplexity(m_naive)
    naive_ppls.append(nppl)
    del m_naive
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ASVD: same rank budget, but activation-aware
    m_asvd = copy.deepcopy(model)
    for li, block in enumerate(m_asvd.transformer.h):
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std  = conv.weight.detach().T.to(DEVICE)
            acts   = calib_acts[li][attr].to(DEVICE)
            W_lr   = asvd(W_std, acts, min(rank, min(W_std.shape)))
            conv.weight.data = W_lr.T.cpu().contiguous()
    m_asvd.to(DEVICE)
    appl = compute_perplexity(m_asvd)
    asvd_ppls.append(appl)
    del m_asvd
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  {rank:4d}  {nppl:10.2f}  {appl:10.2f}  {appl - nppl:+8.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Alpha (scaling exponent) sensitivity at rank=128
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 2 — Alpha sensitivity at rank=128")
print("=" * 60)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
alpha_ppls = []

for alpha in ALPHAS:
    m_test = copy.deepcopy(model)
    for li, block in enumerate(m_test.transformer.h):
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            r = min(128, min(W_std.shape))
            if alpha == 0.0:
                W_lr = naive_svd(W_std, r)
            else:
                acts = calib_acts[li][attr].to(DEVICE)
                W_lr = asvd(W_std, acts, r, alpha=alpha)
            conv.weight.data = W_lr.T.cpu().contiguous()
    m_test.to(DEVICE)
    ppl = compute_perplexity(m_test)
    alpha_ppls.append(ppl)
    del m_test
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  alpha={alpha:.2f}  ppl={ppl:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Per-layer sensitivity (compress one layer at a time, rank=128)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 3 — Per-layer sensitivity at rank=128 (one layer at a time)")
print("=" * 60)
print(f"{'layer':>6}  {'naive_ppl':>10}  {'asvd_ppl':>10}  {'sensitivity_naive':>18}")

layer_naive = []
layer_asvd  = []

for li in range(N_LAYERS):
    for method, results_list in [("naive", layer_naive), ("asvd", layer_asvd)]:
        m_test = copy.deepcopy(model)
        block  = m_test.transformer.h[li]
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_std = conv.weight.detach().T.to(DEVICE)
            r = min(128, min(W_std.shape))
            if method == "naive":
                W_lr = naive_svd(W_std, r)
            else:
                acts = calib_acts[li][attr].to(DEVICE)
                W_lr = asvd(W_std, acts, r)
            conv.weight.data = W_lr.T.cpu().contiguous()
        m_test.to(DEVICE)
        ppl = compute_perplexity(m_test)
        results_list.append(ppl)
        del m_test
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print(f"  {li:4d}  {layer_naive[li]:10.2f}  {layer_asvd[li]:10.2f}  "
          f"  Δ(naive-baseline)={layer_naive[li]-baseline_ppl:+.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle("ASVD: Activation-Aware SVD on GPT-2 Small",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Perplexity vs rank (main result) ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(RANKS, naive_ppls, marker="o", color="tomato",
             linewidth=2, label="Naive SVD")
ax1.semilogy(RANKS, asvd_ppls, marker="s", color="steelblue",
             linewidth=2, label="ASVD (α=0.5)")
ax1.axhline(baseline_ppl, color="green", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax1.set_xlabel("SVD rank (all 12 MLP layers)")
ax1.set_ylabel("Perplexity  [log scale]")
ax1.set_title("Exp 1: Naive SVD vs ASVD — all MLP layers")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, which="both")

# ── Plot 2: PPL improvement at each rank ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
improvements = [n - a for n, a in zip(naive_ppls, asvd_ppls)]
colors = ["steelblue" if v >= 0 else "tomato" for v in improvements]
ax2.bar(range(len(RANKS)), improvements, color=colors, alpha=0.85)
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.set_xticks(range(len(RANKS)))
ax2.set_xticklabels([str(r) for r in RANKS])
ax2.set_xlabel("SVD rank")
ax2.set_ylabel("PPL reduction  (naive − ASVD)")
ax2.set_title("Exp 1: ASVD advantage over naive SVD")
ax2.grid(alpha=0.3, axis="y")

# ── Plot 3: Alpha sensitivity ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(ALPHAS, alpha_ppls, marker="o", color="purple", linewidth=2)
ax3.axhline(baseline_ppl, color="green", linestyle="--",
            label=f"Baseline ({baseline_ppl:.1f})")
best_a = ALPHAS[int(np.argmin(alpha_ppls))]
ax3.axvline(best_a, color="purple", linestyle=":", alpha=0.6,
            label=f"Best α={best_a:.2f}")
ax3.set_xlabel("Alpha (scaling exponent)")
ax3.set_ylabel("Perplexity")
ax3.set_title("Exp 2: Alpha sensitivity  (rank=128, all MLP layers)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Plot 4: Per-layer sensitivity ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
layers_x = list(range(N_LAYERS))
ax4.plot(layers_x, layer_naive, marker="o", color="tomato",
         linewidth=2, label="Naive SVD")
ax4.plot(layers_x, layer_asvd,  marker="s", color="steelblue",
         linewidth=2, label="ASVD")
ax4.axhline(baseline_ppl, color="green", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax4.set_xlabel("Layer index (0=first, 11=last)")
ax4.set_ylabel("Perplexity  (only that layer compressed)")
ax4.set_title("Exp 3: Per-layer sensitivity  (rank=128)")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.savefig("results_asvd.png", dpi=150, bbox_inches="tight")
print("\nSaved results_asvd.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline perplexity: {baseline_ppl:.2f}")

print("\nExp 1 — Perplexity at rank=128:")
idx128 = RANKS.index(128)
print(f"  Naive SVD:  {naive_ppls[idx128]:.2f}  "
      f"({naive_ppls[idx128]/baseline_ppl:.1f}× baseline)")
print(f"  ASVD:       {asvd_ppls[idx128]:.2f}  "
      f"({asvd_ppls[idx128]/baseline_ppl:.1f}× baseline)")

asvd_ok = [(r, p) for r, p in zip(RANKS, asvd_ppls) if p < baseline_ppl * 1.5]
naive_ok = [(r, p) for r, p in zip(RANKS, naive_ppls) if p < baseline_ppl * 1.5]
if asvd_ok:
    print(f"\nASVD within 1.5× baseline: rank ≥ {asvd_ok[0][0]}  "
          f"(ppl={asvd_ok[0][1]:.2f})")
else:
    print("\nASVD does not reach 1.5× baseline at tested ranks")
if naive_ok:
    print(f"Naive SVD within 1.5× baseline: rank ≥ {naive_ok[0][0]}  "
          f"(ppl={naive_ok[0][1]:.2f})")
else:
    print("Naive SVD does not reach 1.5× baseline at tested ranks")

print(f"\nExp 2 — Best alpha: {best_a:.2f}  "
      f"(ppl={alpha_ppls[int(np.argmin(alpha_ppls))]:.2f})")

most_sensitive = int(np.argmax(layer_naive))
least_sensitive = int(np.argmin(layer_naive))
print(f"\nExp 3 — Most sensitive layer:   {most_sensitive}  "
      f"(naive ppl={layer_naive[most_sensitive]:.2f})")
print(f"        Least sensitive layer: {least_sensitive}  "
      f"(naive ppl={layer_naive[least_sensitive]:.2f})")
asvd_layer_improvement = [n - a for n, a in zip(layer_naive, layer_asvd)]
print(f"        Max ASVD improvement:  layer {int(np.argmax(asvd_layer_improvement))}  "
      f"(Δ={max(asvd_layer_improvement):.2f})")
