"""
ASVD on GPT-2 XL (1.5B, 1600-dim hidden)
==========================================
Memory-efficient version:
  - Model lives on CPU; moves to GPU only for perplexity eval
  - SVD done layer-by-layer on GPU (one weight at a time) → no OOM
  - Peak VRAM: ~3.5 GB (one model evaluation)

GPT-2 XL geometry:
  - 48 layers, hidden dim 1600
  - c_fc: 6400×1600,  c_proj: 1600×6400
  - Break-even speedup: rank < 800

Experiments:
  1. Naive SVD vs ASVD — rank sweep [512, 640, 768, 1024, 1280]
  2. Per-layer sensitivity at rank=640 (sample every 4th layer for speed)
"""

import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

print("Loading GPT-2 XL (float16, CPU)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl", torch_dtype=torch.float16)
model.eval()
# ← model stays on CPU; we move it to GPU only during eval
N_LAYERS = len(model.transformer.h)
in_dim  = model.transformer.h[0].mlp.c_fc.weight.shape[0]   # 1600
out_dim = model.transformer.h[0].mlp.c_fc.weight.shape[1]   # 6400
print(f"  {N_LAYERS} layers | hidden={in_dim} | c_fc={out_dim}×{in_dim}")
print(f"  Break-even speedup rank: < {in_dim // 2}")

# ── Calibration corpus ─────────────────────────────────────────────────────────
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
    "semi-supervised, or unsupervised.",
    "The development of large language models has been driven by advances in hardware, "
    "specifically graphics processing units originally designed for rendering video games. "
    "These parallel processors proved well suited for the matrix operations underlying "
    "neural network training.",
]

# Collect activations on GPU (model temporarily moved there)
print("\nCollecting calibration activations...")
model.to(DEVICE)
calib_inputs = [
    tokenizer(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(DEVICE)
    for t in CALIB_TEXTS
]
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

for i in range(N_LAYERS):
    for key in ("c_fc", "c_proj"):
        calib_acts[i][key] = torch.cat(
            [a.flatten(0, 1) for a in calib_acts[i][key]], dim=0
        )
print(f"  c_fc  activations: {calib_acts[0]['c_fc'].shape}")
print(f"  c_proj activations: {calib_acts[0]['c_proj'].shape}")
print(f"  VRAM used after hooks: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Eval text ─────────────────────────────────────────────────────────────────
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

# Baseline eval (model already on GPU)
with torch.no_grad():
    out = model(eval_ids, labels=eval_ids)
baseline_ppl = torch.exp(out.loss).item()
print(f"\nBaseline perplexity: {baseline_ppl:.2f}")

# Move model back to CPU — it lives here between evals
model.cpu()
torch.cuda.empty_cache()
print(f"  VRAM after model→CPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_perplexity(mdl):
    """Move model to GPU, eval, return to CPU."""
    mdl.to(DEVICE)
    with torch.no_grad():
        out = mdl(eval_ids, labels=eval_ids)
    ppl = torch.exp(out.loss).item()
    mdl.cpu()
    torch.cuda.empty_cache()
    return ppl


def svd_compress(W_std_cpu, rank, acts_cpu=None, alpha=0.5):
    """
    W_std_cpu: (out, in) float16 on CPU
    acts_cpu:  (N, in)   float32 on CPU — if given, uses ASVD
    Returns:   (out, in) float16 on CPU
    """
    W = W_std_cpu.float().to(DEVICE)

    if acts_cpu is not None:
        s = acts_cpu.float().to(DEVICE).abs().mean(0).pow(alpha).clamp(min=1e-8)
        W = W * s.unsqueeze(0)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    W_lr = (U[:, :k] * S[:k]) @ Vh[:k, :]

    if acts_cpu is not None:
        W_lr = W_lr / s.unsqueeze(0)

    result = W_lr.to(torch.float16).cpu()
    del W, U, S, Vh, W_lr
    torch.cuda.empty_cache()
    return result


def compress_model(base_model, rank, use_asvd=False, layer_only=None):
    """
    Deep-copy base model and compress MLP weights.
    layer_only: if int, only compress that single layer.
    """
    m = copy.deepcopy(base_model)
    layers = [layer_only] if layer_only is not None else range(N_LAYERS)
    for li in layers:
        block = m.transformer.h[li]
        for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W_T = conv.weight.detach()          # stored as (in, out) in GPT-2
            W_std = W_T.T.contiguous()          # (out, in)
            acts = calib_acts[li][attr] if use_asvd else None
            W_lr = svd_compress(W_std, rank, acts_cpu=acts)
            conv.weight.data = W_lr.T.contiguous()
    return m


# Print param budget table
dense_params = out_dim * in_dim
print(f"\nParameter budget (c_fc {out_dim}×{in_dim}, break-even < {in_dim//2}):")
print(f"  {'rank':>6}  {'% dense':>8}  {'theory speedup':>16}")
for r in [512, 640, 768, 1024, 1280]:
    p = (out_dim + in_dim) * r
    spd = dense_params / p
    be = " ← break-even" if r == in_dim // 2 else ""
    print(f"  {r:6d}  {100*p/dense_params:7.1f}%  {spd:14.2f}×{be}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Rank sweep
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"EXP 1 — Naive SVD vs ASVD rank sweep (all {N_LAYERS} layers)")
print("=" * 60)
print(f"{'rank':>6}  {'naive':>10}  {'asvd':>10}  {'naive/base':>12}  {'asvd/base':>10}")

RANKS = [512, 640, 768, 1024, 1280]
naive_ppls, asvd_ppls = [], []

for rank in RANKS:
    t0 = time.time()
    m = compress_model(model, rank, use_asvd=False)
    nppl = compute_perplexity(m)
    del m
    naive_ppls.append(nppl)

    m = compress_model(model, rank, use_asvd=True)
    appl = compute_perplexity(m)
    del m
    asvd_ppls.append(appl)

    elapsed = time.time() - t0
    print(f"  {rank:4d}  {nppl:10.2f}  {appl:10.2f}  "
          f"{nppl/baseline_ppl:10.2f}×  {appl/baseline_ppl:8.2f}×  "
          f"[{elapsed:.0f}s]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Per-layer sensitivity (sampled layers, rank=640)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
SENS_RANK = 640
sampled = list(range(0, N_LAYERS, 4)) + ([N_LAYERS - 1] if (N_LAYERS - 1) % 4 != 0 else [])
print(f"EXP 2 — Per-layer sensitivity at rank={SENS_RANK}  "
      f"(layers {sampled})")
print("=" * 60)
print(f"{'layer':>6}  {'naive':>10}  {'asvd':>10}  {'naive/base':>12}  {'asvd/base':>10}")

layer_naive, layer_asvd, layer_ids = [], [], []

for li in sampled:
    row = []
    for use_asvd in (False, True):
        m = compress_model(model, SENS_RANK, use_asvd=use_asvd, layer_only=li)
        ppl = compute_perplexity(m)
        del m
        row.append(ppl)
    layer_naive.append(row[0])
    layer_asvd.append(row[1])
    layer_ids.append(li)
    print(f"  {li:4d}  {row[0]:10.2f}  {row[1]:10.2f}  "
          f"{row[0]/baseline_ppl:10.2f}×  {row[1]/baseline_ppl:8.2f}×")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
fig.suptitle("ASVD on GPT-2 XL (1.5B, 1600-dim)",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(RANKS, naive_ppls, marker="o", color="tomato",
             linewidth=2, label="Naive SVD")
ax1.semilogy(RANKS, asvd_ppls, marker="s", color="steelblue",
             linewidth=2, label="ASVD (α=0.5)")
ax1.axhline(baseline_ppl, color="green", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax1.axvline(in_dim // 2, color="gray", linestyle=":", alpha=0.7,
            label=f"Break-even ({in_dim//2})")
ax1.set_xlabel(f"SVD rank (all {N_LAYERS} MLP layers)")
ax1.set_ylabel("Perplexity  [log scale]")
ax1.set_title("Exp 1: Rank sweep")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, which="both")

ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(layer_ids, layer_naive, marker="o", color="tomato",
             linewidth=1.5, markersize=5, label="Naive SVD")
ax2.semilogy(layer_ids, layer_asvd, marker="s", color="steelblue",
             linewidth=1.5, markersize=5, label="ASVD")
ax2.axhline(baseline_ppl, color="green", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax2.set_xlabel("Layer index  (sampled)")
ax2.set_ylabel("Perplexity  [log scale]")
ax2.set_title(f"Exp 2: Per-layer sensitivity (rank={SENS_RANK})")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, which="both")

plt.savefig("results_gpt2xl.png", dpi=150, bbox_inches="tight")
print("\nSaved results_gpt2xl.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: GPT-2 XL | {N_LAYERS} layers | {in_dim}-dim | baseline ppl={baseline_ppl:.2f}")
print(f"Break-even speedup rank: < {in_dim // 2}")

print(f"\nExp 1 — Rank sweep (all {N_LAYERS} MLP layers):")
for r, np_, ap in zip(RANKS, naive_ppls, asvd_ppls):
    pct = 100 * (out_dim + in_dim) * r / dense_params
    spd = dense_params / ((out_dim + in_dim) * r)
    print(f"  rank={r} ({pct:.0f}%, {spd:.2f}× theory):  "
          f"naive={np_:.2f} ({np_/baseline_ppl:.2f}×)  "
          f"asvd={ap:.2f} ({ap/baseline_ppl:.2f}×)")

asvd_good  = [(r, p) for r, p in zip(RANKS, asvd_ppls)  if p < baseline_ppl * 1.5]
naive_good = [(r, p) for r, p in zip(RANKS, naive_ppls) if p < baseline_ppl * 1.5]
print(f"\nASVD within 1.5× baseline:     ", end="")
print(f"rank ≥ {asvd_good[0][0]}  (ppl={asvd_good[0][1]:.2f})" if asvd_good else "not achieved")
print(f"Naive SVD within 1.5× baseline: ", end="")
print(f"rank ≥ {naive_good[0][0]}  (ppl={naive_good[0][1]:.2f})" if naive_good else "not achieved")

if asvd_good:
    r_ok = asvd_good[0][0]
    spd_ok = dense_params / ((out_dim + in_dim) * r_ok)
    be = r_ok < in_dim // 2
    print(f"\n→ ASVD breaks even at rank={r_ok}: {'YES' if be else 'NO'} "
          f"({'%.2f' % spd_ok}× speedup, break-even needs rank < {in_dim//2})")
