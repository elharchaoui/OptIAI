#!/usr/bin/env python3
"""
Experiment 07 — Closed-Form Per-Layer Recovery After ASVD

After ASVD rank=640, compute rank-r correction A·B for each weight matrix via
activation-weighted SVD of the residual (W_orig - W_approx). No training loop.

Key insight: SVD(W_orig * diag(s)) = U Σ Vᵀ
  · ASVD uses top-640 singular vectors → W_approx
  · Correction residual = next r singular vectors — free from the same SVD

Memory strategy: only store per-dimension activation scales s (1.5 MB),
not the full calibration activations (940 MB). This avoids OOM with full swap.

Compare to Exp 06: LoRA rank=16 → PPL=24.53 (0.89× baseline) in 55 min.
"""

import json
import time

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ASVD_RANK  = 640
ALPHA      = 0.5
SEQ_LEN    = 256
N_CALIB    = 2048
CORR_RANKS = [4, 8, 16, 32, 64]


@torch.no_grad()
def evaluate_ppl(model, tokenizer, text, seq_len=SEQ_LEN):
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    n      = (len(tokens) // seq_len) * seq_len
    chunks = tokens[:n].view(-1, seq_len).to(DEVICE)
    total_loss = total_tok = 0
    for chunk in chunks:
        loss = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0)).loss.item()
        total_loss += loss * (seq_len - 1)
        total_tok  += seq_len - 1
    return torch.exp(torch.tensor(total_loss / total_tok)).item()


def main():
    print(f"Device: {DEVICE}")
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model     = GPT2LMHeadModel.from_pretrained("gpt2-large", torch_dtype=torch.float16)
    model     = model.to(DEVICE).eval()

    ds        = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_txt = "\n\n".join(s for s in ds["train"]["text"] if s.strip())
    test_txt  = "\n\n".join(s for s in ds["test"]["text"]  if s.strip())
    calib_ids = tokenizer(train_txt, return_tensors="pt").input_ids[0][:N_CALIB]

    print("Baseline perplexity...")
    t0       = time.time()
    base_ppl = evaluate_ppl(model, tokenizer, test_txt)
    print(f"  PPL = {base_ppl:.2f}  ({time.time()-t0:.0f}s)")

    targets = {
        name: mod
        for name, mod in model.named_modules()
        if hasattr(mod, "weight")
        and any(name.endswith(s) for s in (".mlp.c_fc", ".mlp.c_proj"))
    }
    print(f"Target layers: {len(targets)}")

    # ── Collect per-dimension activation scales (not full activations) ─────────
    # Only store mean(|X|^alpha) per input dimension — ~1.5 MB total instead of ~940 MB.
    print(f"\nCollecting activation scales ({N_CALIB} tokens)...")
    act_scales = {}  # name → (in_dim,) float32 on CPU
    counts     = {}
    hooks      = []

    def make_hook(name):
        def h(mod, inp, out):
            x = inp[0].detach().cpu().float()
            x_flat = x.reshape(-1, x.shape[-1])                  # (n_tokens, in_dim)
            batch_scale = x_flat.abs().mean(0)                    # (in_dim,)
            n_new = x_flat.shape[0]
            if name not in act_scales:
                act_scales[name] = batch_scale
                counts[name]     = n_new
            else:
                n_old              = counts[name]
                act_scales[name]   = (act_scales[name] * n_old + batch_scale * n_new) / (n_old + n_new)
                counts[name]      += n_new
        return h

    for name, mod in targets.items():
        hooks.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for chunk in calib_ids.view(-1, SEQ_LEN).to(DEVICE):
            model(chunk.unsqueeze(0))

    for h in hooks:
        h.remove()

    scale_mem = sum(v.numel() * 4 for v in act_scales.values()) / 1e6
    print(f"  Scales collected: {len(act_scales)} layers  ({scale_mem:.1f} MB CPU)")

    # ── One-pass: ASVD + precompute all correction ranks ─────────────────────
    # Single SVD per layer covers both ASVD (top-640) and corrections (next r).
    print(f"\nApplying ASVD rank={ASVD_RANK} + precomputing corrections...")
    W_approx_store = {}                          # name → (out, in) float16 CPU
    corrections    = {r: {} for r in CORR_RANKS} # rank → name → (A, B) float16 CPU
    t_svd = time.time()

    for i, (name, mod) in enumerate(targets.items()):
        # SVD on GPU: uses VRAM (12.5 GB free) instead of RAM (swap-full)
        W = mod.weight.detach().T.float().to(DEVICE)   # (out_dim, in_dim), float32 on GPU
        s = act_scales[name].pow(ALPHA).clamp(min=1e-6).to(DEVICE)  # (in_dim,) on GPU

        U, sv, Vt = torch.linalg.svd(W * s[None, :], full_matrices=False)

        # ASVD: top-ASVD_RANK singular components
        W_approx = (U[:, :ASVD_RANK] * sv[:ASVD_RANK]) @ Vt[:ASVD_RANK, :] / s[None, :]
        mod.weight.data = W_approx.T.half().contiguous()   # stays on GPU
        W_approx_store[name] = W_approx.half().cpu()        # save to CPU float16

        # Corrections: next r singular components (residual in activation-aware space)
        max_sv = U.shape[1]
        for r in CORR_RANKS:
            start = ASVD_RANK
            end   = min(ASVD_RANK + r, max_sv)
            A = U[:, start:end].half().cpu()
            B = (sv[start:end, None] * Vt[start:end, :] / s[None, :]).half().cpu()
            corrections[r][name] = (A, B)

        del W, U, sv, Vt, W_approx
        torch.cuda.empty_cache()

        if (i + 1) % 20 == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  {i+1}/{len(targets)} layers  ({time.time()-t_svd:.0f}s)  VRAM: {vram:.1f} GB")

    print(f"  Done: {len(targets)} layers  ({time.time()-t_svd:.0f}s total)")

    # ── ASVD-only eval ────────────────────────────────────────────────────────
    print("ASVD-only perplexity...")
    t0       = time.time()
    asvd_ppl = evaluate_ppl(model, tokenizer, test_txt)
    print(f"  PPL = {asvd_ppl:.2f}  ({asvd_ppl/base_ppl:.2f}× baseline)  ({time.time()-t0:.0f}s)")

    # ── Correction sweep ──────────────────────────────────────────────────────
    results = {"baseline": base_ppl, "asvd_only": asvd_ppl, "correction_ppl": {}}

    print("\n=== Closed-form correction sweep ===")
    for r in CORR_RANKS:
        t0 = time.time()
        for name, mod in targets.items():
            A, B = corrections[r][name]
            W_corr = W_approx_store[name].float() + A.float() @ B.float()
            mod.weight.data = W_corr.T.half().to(DEVICE).contiguous()
            del W_corr
        ppl     = evaluate_ppl(model, tokenizer, test_txt)
        elapsed = time.time() - t0
        print(f"  rank={r:2d}: PPL={ppl:.2f}  ({ppl/base_ppl:.3f}×)  [{elapsed:.0f}s]")
        results["correction_ppl"][r] = ppl
        # Reset to ASVD weights
        for name, mod in targets.items():
            mod.weight.data = W_approx_store[name].T.to(DEVICE).contiguous()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("RESULTS SUMMARY")
    print("=" * 62)
    print(f"  {'Stage':<40} {'PPL':>7}  {'×baseline':>10}")
    print(f"  {'Baseline GPT-2 Large':<40} {base_ppl:>7.2f}  {'1.000×':>10}")
    print(f"  {f'ASVD rank={ASVD_RANK}':<40} {asvd_ppl:>7.2f}  {asvd_ppl/base_ppl:>9.3f}×")
    for r, ppl in results["correction_ppl"].items():
        print(f"  {f'+ closed-form rank={r}':<40} {ppl:>7.2f}  {ppl/base_ppl:>9.3f}×")
    print(f"  {'LoRA fine-tuned rank=16 (Exp 06)':<40} {'24.53':>7}  {'0.890×':>10}  [55 min T4]")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results.json")


if __name__ == "__main__":
    main()
