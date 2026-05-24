# Experiment 02 — GPT-2 Small: Naive SVD Baseline

**Model:** GPT-2 small (117M params, 12 layers, 768-dim hidden)  
**Script:** `experiment.py`  
**Output:** `results.png`

## Setup

- Replaced all MLP weight matrices (c_fc: 3072×768, c_proj: 768×3072) with truncated SVD approximations
- No activation awareness — pure weight-space approximation
- Evaluated perplexity on a fixed passage (~500 tokens)
- Baseline perplexity: ~24–28 (varies slightly by text length)
- Ranks tested: 64, 128, 192, 256, 384, 512, 768

## Results

| rank | % of dense | ppl      | ×baseline |
|------|-----------|----------|-----------|
| 64   | 5.6%      | 1600+    | >60×      |
| 128  | 11.1%     | 1200+    | >45×      |
| 192  | 16.7%     | 900+     | ~35×      |
| 256  | 22.2%     | 600+     | ~23×      |
| 384  | 33.3%     | 300+     | ~11×      |
| 512  | 44.4%     | 293      | ~10×      |
| 768  | 66.7%     | 35–50    | ~1.5–2×   |

- GPU speedup (3072×768 matrices) only appears at rank ≤ 256 (~3.8× theoretical)
- At rank ≤ 256, quality is catastrophically destroyed

**Singular spectrum analysis:** rank@90% energy = 39–67% of full rank across layers. GPT-2 weights are NOT particularly low-rank.

## Conclusions

1. **Naive SVD fails completely on GPT-2 small.** At every rank where speedup is possible (< 384), perplexity explodes 10–60×.
2. **The weight spectrum is dense.** GPT-2 doesn't have a clean low-rank structure — you need 39–67% of the full rank just to capture 90% of the energy.
3. **Speedup zone and quality zone never overlap.** The minimum viable quality rank (≥ 512) gives only 1.2× theoretical speedup — barely worth the approximation cost.

## What This Motivated

Naive SVD minimizes weight reconstruction error, but inference cares about **output error on real inputs**. ASVD (activation-aware SVD) reweights the decomposition by input statistics. → See Experiment 03.
