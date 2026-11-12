"""
ASVD on GPT-2 Large (774M, 1280-dim hidden)
============================================
Memory-efficient design:
  - device_map="cuda:0" → weights load directly to GPU, bypassing CPU RAM peak
  - In-place patching: patch weights → eval → restore (no deepcopy, no 2nd model)
  - Peak VRAM: ~2 GB (model + one SVD pass)
  - Peak CPU RAM: ~2.5 GB (imports + stored original weights + calibration activations)

GPT-2 Large geometry:
  - 36 layers, hidden dim 1280
  - c_fc: 5120×1280,  c_proj: 1280×5120
  - Break-even speedup: rank < 640
"""

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

# Load directly to GPU — never fully occupies CPU RAM
print("Loading GPT-2 Large directly to GPU (float16)...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained(
    "gpt2-large",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
model.eval()
N_LAYERS = len(model.transformer.h)
in_dim  = model.transformer.h[0].mlp.c_fc.weight.shape[0]   # 1600 (stored transposed)
out_dim = model.transformer.h[0].mlp.c_fc.weight.shape[1]   # 6400
print(f"  {N_LAYERS} layers | hidden={in_dim} | c_fc={out_dim}×{in_dim}")
print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
# Break-even (speedup > 1): rank < out·in/(out+in). in_dim//2 is the 1.6× target for 4:1 matrices.
BREAK_EVEN = out_dim * in_dim // (out_dim + in_dim)
TARGET_RANK = in_dim // 2
print(f"  Break-even speedup rank: < {BREAK_EVEN}  |  1.6× target rank: {TARGET_RANK}")

# ── Save original weights to CPU (allows in-place patch + restore) ─────────────
print("\nCaching original weights on CPU...")
orig_weights = {}
for li, block in enumerate(model.transformer.h):
    orig_weights[li] = {
        "c_fc":   block.mlp.c_fc.weight.data.cpu().clone(),
        "c_proj": block.mlp.c_proj.weight.data.cpu().clone(),
    }
print(f"  Cached {N_LAYERS * 2} weight tensors on CPU")

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
    "These parallel processors proved well suited for matrix operations underlying "
    "neural network training.",
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
    "on the campus of Dartmouth College during the summer of 1956. Those who attended would "
    "become the leaders of AI research for decades. Many of them predicted that a machine as "
    "intelligent as a human being would exist in no more than a generation."
)
eval_ids = tokenizer(EVAL_TEXT, return_tensors="pt",
                     truncation=True, max_length=512)["input_ids"].to(DEVICE)

# ── Collect calibration activations ───────────────────────────────────────────
print("\nCollecting calibration activations...")
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

# ── Baseline ───────────────────────────────────────────────────────────────────
with torch.no_grad():
    out = model(eval_ids, labels=eval_ids)
baseline_ppl = torch.exp(out.loss).item()
print(f"\nBaseline perplexity: {baseline_ppl:.2f}")
print(f"VRAM after setup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

dense_params = out_dim * in_dim
print(f"\nParameter budget (c_fc {out_dim}×{in_dim}, break-even < {BREAK_EVEN}):")
for r in [256, 384, 512, 640, 768, 1024]:
    p = (out_dim + in_dim) * r
    spd = dense_params / p
    be = " ← 1.6× target" if r == TARGET_RANK else ""
    print(f"  rank={r}: {100*p/dense_params:.0f}% of dense, {spd:.2f}× theory speedup{be}")


# ── Core: in-place patch → eval → restore ─────────────────────────────────────
def compute_perplexity():
    with torch.no_grad():
        out = model(eval_ids, labels=eval_ids)
    return torch.exp(out.loss).item()


def patch_layer(li, rank, use_asvd):
    block = model.transformer.h[li]
    for attr, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
        # GPT-2 Conv1D stores weights as (in, out); W_std = weight.T = (out, in)
        W_std = conv.weight.data.T.float()   # (out, in) on GPU

        if use_asvd:
            s = calib_acts[li][attr].float().to(DEVICE).abs().mean(0).pow(0.5).clamp(min=1e-8)
            W = W_std * s.unsqueeze(0)
        else:
            W = W_std

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, len(S))
        W_lr = (U[:, :k] * S[:k]) @ Vh[:k, :]

        if use_asvd:
            W_lr = W_lr / s.unsqueeze(0)

        conv.weight.data = W_lr.to(torch.float16).T.contiguous()
        del U, S, Vh, W_lr, W, W_std
        if use_asvd:
            del s
    torch.cuda.empty_cache()


def restore_layer(li):
    block = model.transformer.h[li]
    block.mlp.c_fc.weight.data   = orig_weights[li]["c_fc"].to(DEVICE)
    block.mlp.c_proj.weight.data = orig_weights[li]["c_proj"].to(DEVICE)


def run_experiment(rank, use_asvd, layer_only=None):
    """Patch model in-place, eval, restore. Returns perplexity."""
    layers = [layer_only] if layer_only is not None else range(N_LAYERS)
    for li in layers:
        patch_layer(li, rank, use_asvd)
    ppl = compute_perplexity()
    for li in layers:
        restore_layer(li)
    return ppl


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Rank sweep (all 48 MLP layers)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"EXP 1 — Naive SVD vs ASVD rank sweep (all {N_LAYERS} layers)")
print("=" * 60)
print(f"{'rank':>6}  {'naive':>10}  {'asvd':>10}  {'naive/base':>12}  {'asvd/base':>10}  {'time':>6}")

RANKS = [256, 384, 512, 640, 768, 1024]
naive_ppls, asvd_ppls = [], []

for rank in RANKS:
    t0 = time.time()
    nppl = run_experiment(rank, use_asvd=False)
    appl = run_experiment(rank, use_asvd=True)
    elapsed = time.time() - t0
    naive_ppls.append(nppl)
    asvd_ppls.append(appl)
    print(f"  {rank:4d}  {nppl:10.2f}  {appl:10.2f}  "
          f"{nppl/baseline_ppl:10.2f}×  {appl/baseline_ppl:8.2f}×  [{elapsed:.0f}s]")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Per-layer sensitivity (sampled, rank=640)
# ══════════════════════════════════════════════════════════════════════════════
SENS_RANK = 640
sampled = list(range(0, N_LAYERS, 4))
if (N_LAYERS - 1) not in sampled:
    sampled.append(N_LAYERS - 1)

print("\n" + "=" * 60)
print(f"EXP 2 — Per-layer sensitivity at rank={SENS_RANK}  (sampled layers)")
print("=" * 60)
print(f"{'layer':>6}  {'naive':>10}  {'asvd':>10}  {'naive/base':>12}  {'asvd/base':>10}")

layer_naive, layer_asvd, layer_ids = [], [], []

for li in sampled:
    nppl = run_experiment(SENS_RANK, use_asvd=False, layer_only=li)
    appl = run_experiment(SENS_RANK, use_asvd=True,  layer_only=li)
    layer_naive.append(nppl)
    layer_asvd.append(appl)
    layer_ids.append(li)
    print(f"  {li:4d}  {nppl:10.2f}  {appl:10.2f}  "
          f"{nppl/baseline_ppl:10.2f}×  {appl/baseline_ppl:8.2f}×")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
fig.suptitle("ASVD on GPT-2 Large (774M, 1280-dim)", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(RANKS, naive_ppls, marker="o", color="tomato", linewidth=2, label="Naive SVD")
ax1.semilogy(RANKS, asvd_ppls,  marker="s", color="steelblue", linewidth=2, label="ASVD (α=0.5)")
ax1.axhline(baseline_ppl, color="green", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax1.axvline(TARGET_RANK, color="gray", linestyle=":", alpha=0.7,
            label=f"1.6× target ({TARGET_RANK})")
ax1.axvline(BREAK_EVEN, color="black", linestyle=":", alpha=0.5,
            label=f"Break-even ({BREAK_EVEN})")
ax1.set_xlabel(f"SVD rank (all {N_LAYERS} MLP layers)")
ax1.set_ylabel("Perplexity  [log scale]")
ax1.set_title("Exp 1: Rank sweep — all MLP layers")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, which="both")

ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(layer_ids, layer_naive, marker="o", color="tomato",
             linewidth=1.5, markersize=5, label="Naive SVD")
ax2.semilogy(layer_ids, layer_asvd, marker="s", color="steelblue",
             linewidth=1.5, markersize=5, label="ASVD")
ax2.axhline(baseline_ppl, color="green", linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_ppl:.1f})")
ax2.set_xlabel("Layer index  (sampled every 4)")
ax2.set_ylabel("Perplexity  [log scale]")
ax2.set_title(f"Exp 2: Per-layer sensitivity  (rank={SENS_RANK})")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3, which="both")

plt.savefig("results_gpt2xl.png", dpi=150, bbox_inches="tight")
print("\nSaved results_gpt2xl.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"GPT-2 Large | {N_LAYERS} layers | {in_dim}-dim | baseline ppl={baseline_ppl:.2f}")
print(f"Break-even speedup rank: < {BREAK_EVEN}  |  1.6× target rank: {TARGET_RANK}\n")

for r, np_, ap in zip(RANKS, naive_ppls, asvd_ppls):
    spd = dense_params / ((out_dim + in_dim) * r)
    be = " ← 1.6× target" if r == TARGET_RANK else ""
    print(f"  rank={r} ({spd:.2f}× theory):  "
          f"naive={np_:.2f} ({np_/baseline_ppl:.2f}×)  "
          f"asvd={ap:.2f} ({ap/baseline_ppl:.2f}×){be}")

asvd_good  = [(r, p) for r, p in zip(RANKS, asvd_ppls)  if p < baseline_ppl * 1.5]
naive_good = [(r, p) for r, p in zip(RANKS, naive_ppls) if p < baseline_ppl * 1.5]
print(f"\nASVD within 1.5× baseline:     ", end="")
print(f"rank ≥ {asvd_good[0][0]}  (ppl={asvd_good[0][1]:.2f})" if asvd_good else "not achieved")
print(f"Naive SVD within 1.5× baseline: ", end="")
print(f"rank ≥ {naive_good[0][0]}  (ppl={naive_good[0][1]:.2f})" if naive_good else "not achieved")

if asvd_good:
    r_ok = asvd_good[0][0]
    spd_ok = dense_params / ((out_dim + in_dim) * r_ok)
    be = r_ok < BREAK_EVEN
    print(f"\n→ At rank={r_ok}: {spd_ok:.2f}× theory speedup  |  "
          f"speedup zone: {'YES' if be else 'NO'} (need rank < {BREAK_EVEN})")
