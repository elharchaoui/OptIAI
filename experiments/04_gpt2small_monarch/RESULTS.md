# Experiment 04 — GPT-2 Small: Monarch Projection vs ASVD

**Model:** GPT-2 small (117M params, 12 layers, 768-dim hidden)  
**Script:** `experiment.py`  
**Output:** `results.png`

## Setup

**Monarch projection** (arXiv:2204.00595): approximate W ≈ L·P^T·R where L and R are block-diagonal matrices and P is a fixed permutation. Algorithm:
1. Reshape W into a 4D tensor indexed by block structure
2. Per-(j,k) pair: rank-1 SVD to find best rank-1 outer product
3. Assemble L and R block-diagonal factors

**Activation-Aware Monarch (novel, not in any paper):** apply ASVD-style column scaling before Monarch projection, then de-scale — combining both ideas.

Parameter budgets tested (c_fc layer: 3072×768 = 2,359,296 dense params):

| n_blocks | params  | % dense | equiv SVD rank |
|----------|---------|---------|----------------|
| 8        | 319,488 | 13.5%   | 83             |
| 16       | 196,608 | 8.3%    | 51             |
| 32       | 172,032 | 7.3%    | 44             |

Baseline perplexity: 28.03

## Results

### Exp 1 — All methods, all MLP layers

| n_blocks | equiv rank | Naive SVD | ASVD    | Monarch | AA-Monarch |
|----------|-----------|-----------|---------|---------|------------|
| 8        | 83        | 5789 (206×)| 7891 (281×)| **2406 (85×)**| 2507 (89×)|
| 16       | 51        | 6922 (247×)| 6655 (237×)| **2025 (72×)**| 2829 (101×)|
| 32       | 44        | 7864 (280×)| 5802 (207×)| **2164 (77×)**| 2313 (82×)|

- **Monarch consistently beats SVD and ASVD at equal parameter budget** (~3× better perplexity)
- **AA-Monarch is worse than plain Monarch** at every setting (see analysis below)
- All methods are far from usable — these parameter budgets (7–14%) are too aggressive for GPT-2

### Exp 2 — Per-layer sensitivity (Monarch n_blocks=16, ASVD rank=32)

| Layer | Monarch ppl | AA-Monarch ppl |
|-------|-------------|----------------|
| 0     | 289         | 468            |
| 1     | 30          | 33             |
| 2–9   | 28–30       | 28–31          |
| 10–11 | 31–33       | 31–33          |

- Layer 0 is still the most sensitive (consistent with Experiment 03)
- AA-Monarch makes layer 0 even worse (+179 ppl vs plain Monarch)
- Middle/late layers nearly unaffected for both methods

### Exp 3 — Weight reconstruction error (layer 6, c_fc)

| n_blocks | SVD err | ASVD err | Monarch err | AA-Monarch err |
|----------|---------|----------|-------------|----------------|
| 8        | 0.8087  | 0.8138   | 0.8829      | 0.8871         |
| 16       | 0.8630  | 0.8671   | 0.9171      | 0.9233         |
| 32       | 0.8761  | 0.8800   | 0.9259      | 0.9338         |

**Key finding:** Monarch has *higher* Frobenius reconstruction error than SVD but *lower* perplexity. Reconstruction error (weight space) is a poor proxy for output quality (activation space). Monarch captures task-relevant geometry better despite fitting weights less precisely.

## Why AA-Monarch Failed

ASVD's activation scaling helps when you have enough rank budget to preserve the important directions (it worked well at rank ≥ 384 on GPT-2 small, 44% of dense). At 7–14% parameter budget:
- Every singular vector/block matters — there's no slack
- Scaling distorts the singular value spectrum before decomposition
- The distortion costs more quality than the reweighting gains
- At very low budgets, the unbiased reconstruction (plain Monarch) is better

## Conclusions

1. **Monarch block structure is inherently better than global SVD** at equal parameter budgets — captures local structure that global rank misses.
2. **Reconstruction error is a misleading metric.** Optimize for output fidelity (perplexity), not weight Frobenius norm.
3. **Activation scaling (ASVD trick) only helps at high parameter budgets** (≥ 40% of dense). At low budgets, it hurts.
4. **GPT-2 small is still too small.** Even at n_blocks=8 (13.5% of dense, the best Monarch config), ppl is 85× baseline. The 3× Monarch advantage over SVD doesn't close the gap enough.

## What This Motivated

The fundamental bottleneck is model scale. GPT-2 (768-dim) lacks low-rank structure. Scaling to GPT-2 Large (1280-dim, 774M params) where matrices are larger and more redundant. → See Experiment 05.
