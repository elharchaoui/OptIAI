"""
Real-model experiment: GPT-2 small
===================================
Tests SVD low-rank approximation on actual trained weight matrices.

Key difference from toy experiment:
  - Real weights have fast-decaying singular values (low intrinsic rank)
  - GPU timing on RTX 3060 for realistic matrix sizes (768×3072)
  - End-to-end perplexity to measure downstream impact

Experiments:
  1. Singular value decay — how compressible are GPT-2's weight matrices?
  2. Per-layer SVD quality — error vs rank on real weights
  3. GPU timing — does low-rank actually speed up at real dimensions?
  4. Perplexity — what rank is needed before language quality degrades?
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ── load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2 small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
model.eval().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.1f}M")

# ── collect weight matrices ───────────────────────────────────────────────────
# GPT-2 uses Conv1D (transposed), so weights are stored as (in, out)
# We normalise to (out, in) = standard linear layer convention
weight_groups = {
    "attn_proj": [],   # attention output projection  768×768
    "mlp_fc":    [],   # FFN up-projection             768×3072
    "mlp_proj":  [],   # FFN down-projection          3072×768
}

for i, block in enumerate(model.transformer.h):
    weight_groups["attn_proj"].append(
        block.attn.c_proj.weight.detach().T.cpu()   # (768, 768)
    )
    weight_groups["mlp_fc"].append(
        block.mlp.c_fc.weight.detach().T.cpu()      # (3072, 768)
    )
    weight_groups["mlp_proj"].append(
        block.mlp.c_proj.weight.detach().T.cpu()    # (768, 3072)
    )

# representative matrices (layer 6 = middle of network)
W_attn = weight_groups["attn_proj"][6].to(DEVICE)
W_fc   = weight_groups["mlp_fc"][6].to(DEVICE)
W_proj = weight_groups["mlp_proj"][6].to(DEVICE)

print(f"\nMatrix shapes: attn={tuple(W_attn.shape)}  "
      f"mlp_fc={tuple(W_fc.shape)}  mlp_proj={tuple(W_proj.shape)}")

# ── helpers ───────────────────────────────────────────────────────────────────
def svd_approx(W, rank):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank, :]

def rel_err(W, W_approx):
    return (torch.norm(W - W_approx) / torch.norm(W)).item()

def timeit_gpu(fn, n=300):
    for _ in range(30):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

N_RUNS  = 200
RANKS   = [8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Singular value decay
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 1 — Singular value decay (how compressible are real weights?)")
print("=" * 60)

sv_data = {}
for name, W in [("attn_proj", W_attn), ("mlp_fc", W_fc), ("mlp_proj", W_proj)]:
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    S = S.cpu().numpy()
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    r90 = int(np.searchsorted(cumvar, 0.90)) + 1
    r99 = int(np.searchsorted(cumvar, 0.99)) + 1
    sv_data[name] = (S, cumvar)
    print(f"  {name:12s}  shape={tuple(W.shape)}  "
          f"rank@90%={r90}  rank@99%={r99}  "
          f"full_rank={W.shape[0]}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Weight approximation quality vs rank
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 2 — SVD weight reconstruction error vs rank")
print("=" * 60)

err_data = {name: [] for name in ["attn_proj", "mlp_fc", "mlp_proj"]}
for r in RANKS:
    for name, W in [("attn_proj", W_attn), ("mlp_fc", W_fc), ("mlp_proj", W_proj)]:
        max_r = min(W.shape)
        if r > max_r:
            err_data[name].append(None)
            continue
        W_lr = svd_approx(W, r)
        e = rel_err(W, W_lr)
        err_data[name].append(e)
    print(f"  rank={r:4d}  "
          f"attn={err_data['attn_proj'][-1]:.4f}  "
          f"mlp_fc={err_data['mlp_fc'][-1] or 'n/a'!s:.6}  "
          f"mlp_proj={err_data['mlp_proj'][-1] or 'n/a'!s:.6}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3: GPU timing — does low-rank win at real dimensions?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 3 — GPU timing: dense vs low-rank at real matrix dimensions")
print("=" * 60)

BATCH = 64   # realistic batch during inference
timing_data = {}

for name, W in [("attn_proj (768×768)", W_attn),
                ("mlp_fc   (3072×768)", W_fc)]:
    out_dim, in_dim = W.shape
    X_gpu = torch.randn(BATCH, in_dim, device=DEVICE)

    # precompute SVD factors for each rank
    U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)

    t_dense = timeit_gpu(lambda: X_gpu @ W.T)
    rows = []

    for r in [8, 16, 32, 64, 128, 256]:
        if r > min(W.shape):
            continue
        A = (U_full[:, :r] * S_full[:r]).contiguous()   # (out, r)
        B = Vh_full[:r, :].contiguous()                  # (r, in)
        # low-rank forward: X @ B.T @ A.T
        t_lr = timeit_gpu(lambda A=A, B=B: (X_gpu @ B.T) @ A.T)
        speedup = t_dense / t_lr
        W_lr = A @ B
        e = rel_err(W, W_lr)
        rows.append((r, t_dense*1e6, t_lr*1e6, speedup, e))
        print(f"  {name}  rank={r:4d}  "
              f"dense={t_dense*1e6:.1f}µs  lr={t_lr*1e6:.1f}µs  "
              f"speedup={speedup:.2f}x  weight_err={e:.4f}")
    timing_data[name] = (t_dense, rows)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4: End-to-end perplexity with approximated weights
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXP 4 — End-to-end perplexity (real language quality)")
print("=" * 60)

# sample text for perplexity (avoids HF dataset network issues)
text = (
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
    "intelligent as a human being would exist in no more than a generation, and they were "
    "given millions of dollars to make this vision come true. Eventually it became obvious "
    "that commercial developers and government researchers had grossly underestimated the "
    "difficulty of the project. In 1974, in response to the criticism of Sir James Lighthill "
    "and ongoing pressure from the US Congress to fund more productive projects, both the "
    "U.S. and British governments cut off exploratory research in AI. The next few years "
    "would later be called an AI winter. In the early 1980s, AI research was revived by the "
    "commercial success of expert systems, a form of AI program that simulated the knowledge "
    "and analytical skills of human experts. By 1985, the market for AI had reached over a "
    "billion dollars. At the same time, Japan's fifth generation computer project inspired "
    "the U.S and British governments to restore funding for academic research. However, "
    "beginning with the collapse of the Lisp Machine market in 1987, AI once again fell "
    "into disrepute, and a second, longer-lasting AI winter began. Deep learning, big data "
    "and artificial general intelligence in the 21st century led to a renaissance of neural "
    "network research and the current era of large language models trained on internet-scale "
    "corpora of text data."
)
enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
input_ids = enc["input_ids"].to(DEVICE)

def compute_perplexity(mdl):
    with torch.no_grad():
        out = mdl(input_ids, labels=input_ids)
    return torch.exp(out.loss).item()

baseline_ppl = compute_perplexity(model)
print(f"\n  Baseline perplexity: {baseline_ppl:.2f}")

ppl_results = []

for rank in [32, 64, 128, 192, 256, 384]:
    # clone and replace ALL mlp_fc and mlp_proj weights with low-rank versions
    import copy
    m_approx = copy.deepcopy(model)
    for block in m_approx.transformer.h:
        # mlp.c_fc: shape (in=768, out=3072) in Conv1D convention
        W_up   = block.mlp.c_fc.weight.detach().T    # (3072, 768)
        W_down = block.mlp.c_proj.weight.detach().T  # (768, 3072)

        r = min(rank, min(W_up.shape))
        W_up_lr   = svd_approx(W_up.to(DEVICE), r).cpu()
        W_down_lr = svd_approx(W_down.to(DEVICE), r).cpu()

        block.mlp.c_fc.weight.data   = W_up_lr.T.contiguous()
        block.mlp.c_proj.weight.data = W_down_lr.T.contiguous()

    m_approx.to(DEVICE)
    ppl = compute_perplexity(m_approx)
    delta = ppl - baseline_ppl
    ppl_results.append((rank, ppl, delta))
    print(f"  rank={rank:4d}  ppl={ppl:.2f}  Δppl={delta:+.2f}")
    del m_approx
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.suptitle("GPT-2 Small — SVD Low-Rank Analysis on Real Trained Weights",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── Plot 1: singular value spectra ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for name, (S, _) in sv_data.items():
    ax1.plot(S / S[0], label=name)     # normalise to 1
ax1.set_xlabel("Singular value index")
ax1.set_ylabel("Normalised singular value")
ax1.set_title("Exp 1: Singular value decay (real weights)")
ax1.set_yscale("log")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── Plot 2: cumulative energy captured by top-r singular values ──────────────
ax2 = fig.add_subplot(gs[0, 1])
for name, (_, cumvar) in sv_data.items():
    ax2.plot(cumvar, label=name)
ax2.axhline(0.90, color='orange', linestyle='--', alpha=0.7, label='90%')
ax2.axhline(0.99, color='red',    linestyle='--', alpha=0.7, label='99%')
ax2.set_xlabel("Rank (number of singular values kept)")
ax2.set_ylabel("Cumulative energy fraction")
ax2.set_title("Exp 2: Energy captured vs rank")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Plot 3: GPU speedup vs weight reconstruction error ───────────────────────
ax3 = fig.add_subplot(gs[1, 0])
markers = ['o', 's']
for (name, (_, rows)), mk in zip(timing_data.items(), markers):
    su = [r[3] for r in rows]
    er = [r[4] for r in rows]
    rs = [r[0] for r in rows]
    ax3.plot(su, er, marker=mk, label=name.strip())
    for s, e, r in zip(su, er, rs):
        ax3.annotate(f"r={r}", (s, e), textcoords="offset points",
                     xytext=(4, 4), fontsize=7)
ax3.axhline(0.05, color='green', linestyle='--', alpha=0.7, label='err<0.05')
ax3.axvline(1.0,  color='gray',  linestyle=':',  alpha=0.7, label='break-even')
ax3.set_xlabel("Speedup vs dense")
ax3.set_ylabel("Weight reconstruction error")
ax3.set_title("Exp 3: GPU speedup vs error (real dims)")
ax3.legend(fontsize=7)
ax3.grid(alpha=0.3)

# ── Plot 4: perplexity vs rank ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ranks_ppl = [r[0] for r in ppl_results]
ppls      = [r[1] for r in ppl_results]
ax4.plot(ranks_ppl, ppls, marker='o', color='steelblue', label='approx model')
ax4.axhline(baseline_ppl, color='red', linestyle='--', label=f'baseline ({baseline_ppl:.1f})')
ax4.fill_between(ranks_ppl,
                 [baseline_ppl] * len(ranks_ppl),
                 ppls,
                 alpha=0.15, color='red')
ax4.set_xlabel("SVD rank (all MLP layers)")
ax4.set_ylabel("Perplexity (lower = better)")
ax4.set_title("Exp 4: End-to-end perplexity vs rank")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.savefig("results_gpt2.png", dpi=150, bbox_inches='tight')
print("\nSaved results_gpt2.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for name, (S, cumvar) in sv_data.items():
    r90 = int(np.searchsorted(cumvar, 0.90)) + 1
    frac = r90 / len(S)
    print(f"  {name}: {r90} singular values capture 90% energy "
          f"({frac*100:.1f}% of full rank)")

best_rank_ppl = min(ppl_results, key=lambda x: x[1])
ok_ppl = [(r, p) for r, p, d in ppl_results if d < 5.0]
if ok_ppl:
    min_ok = ok_ppl[0]
    print(f"\n  Perplexity within +5 of baseline: rank >= {min_ok[0]} "
          f"(ppl={min_ok[1]:.2f})")
print(f"  Baseline perplexity: {baseline_ppl:.2f}")
