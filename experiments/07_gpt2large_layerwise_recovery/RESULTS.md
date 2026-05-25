# Experiment 07 — Closed-Form Per-Layer Recovery After ASVD

## Setup

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 Large (774M params, 36 layers, 1280 hidden) |
| Compression | ASVD rank=640 on `c_fc`, `c_proj` (72 layers total) |
| Correction | Activation-weighted SVD of residual W_orig − W_approx |
| Calibration | 2048 tokens from WikiText-2 train (activation scales only) |
| Evaluation | WikiText-2 test set (283K tokens, stride=256) |
| Hardware | RTX 3060 (12.5 GB VRAM), ~4.5 min total runtime |

## Method

For each compressed layer (W_approx ≈ ASVD(W_orig)):

1. Collect per-dimension activation scales during a calibration pass:
   `s_j = mean|X_j|^α`

2. Compute single activation-weighted SVD:
   `W_orig * diag(s) = U Σ Vᵀ`

3. **ASVD uses top-640 singular vectors** → W_approx

4. **Correction uses next-r singular vectors** (the residual in scaled space):
   `A = U[:, 640:640+r]`,  `B = (Σ_{640:640+r} · Vt_{640:640+r}) / s`

5. Apply: `W_corrected = W_approx + A · B`

No second SVD required — the correction is free from the same decomposition.
No training loop, no gradient computation.

## Results

| Stage | PPL (WikiText-2) | ×baseline |
|-------|-----------------|-----------|
| Baseline GPT-2 Large | 27.36 | 1.000× |
| ASVD rank=640 (no recovery) | 159.72 | 5.838× |
| + closed-form rank=4 | 150.07 | 5.485× |
| + closed-form rank=8 | 141.80 | 5.183× |
| + closed-form rank=16 | 124.20 | 4.539× |
| + closed-form rank=32 | 99.31 | 3.630× |
| + closed-form rank=64 | 70.89 | 2.591× |
| **LoRA fine-tuned rank=16** (Exp 06) | **24.53** | **0.890×** |

## Key Finding

**Closed-form correction at rank=16 achieves 4.54× baseline — vs 0.89× for LoRA fine-tuning at the same rank. A 5× quality gap. Even rank=64 (4× more parameters) is still 3× worse than LoRA rank=16.**

The hypothesis was wrong in a clean, scientifically useful way.

## Why It Fails

The correction optimizes each of the 72 layers **independently**:
- Layer 0's correction is computed assuming it sees ASVD'd inputs
- Layer 1's correction is computed assuming it sees ASVD'd layer 0 outputs
- Once layer 0 is corrected, layer 1 receives a slightly different input — its correction is now mismatched
- This mismatch compounds across all 36 transformer blocks

LoRA fine-tuning, in contrast, propagates gradients through the full network. Each adapter sees the cascaded effect of all other adapters — the optimization is inherently joint, not layer-local.

**The fundamental issue:** cross-layer error propagation cannot be fixed by per-layer independent correction. The 5.58× ASVD degradation is cumulative (confirmed in Exp 05), and reversing it also requires a cumulative (end-to-end) correction.

## Speedup Consideration

Closed-form correction (when merged into weights) produces a dense matrix — it destroys the 1.6× inference speedup. LoRA fine-tuning also loses the speedup if adapters are merged, but in LoRA the factored (compressed) weights do inference and the adapters are discarded after training.

If the closed-form correction were kept in factored form (W_approx + A·B at inference), we would trade some speedup for quality:
- W_approx: rank=640 factored (gives 1.6× speedup)
- A·B: rank=r additional matmul overhead

At rank=64, the overhead is small (64/640 = 10% extra compute), and the speedup would be roughly 1.45× — still worthwhile, but quality at 2.59× baseline is still unusable.

## Conclusion

Per-layer closed-form reconstruction is fast (~18 seconds for 72 SVDs on RTX 3060) but insufficient. End-to-end gradient-based recovery (LoRA fine-tuning) is 5× better at the same parameter budget because it accounts for cross-layer error cascade.

This experiment confirms that the 55-minute LoRA training in Exp 06 is doing genuinely necessary work — not reducible to a closed-form trick.
