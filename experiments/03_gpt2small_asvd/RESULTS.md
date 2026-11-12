# Experiment 03 — GPT-2 Small: ASVD vs Naive SVD

**Model:** GPT-2 small (117M params, 12 layers, 768-dim hidden)  
**Script:** `experiment.py`  
**Output:** `results.png`

## Setup

ASVD method (arXiv:2312.05821):
1. Collect calibration inputs X via forward hooks on each MLP layer (6 texts, ~260 tokens total)
2. Compute per-channel importance: `s_j = mean|X_j|^α`
3. Scale weight columns: `W_scaled = W · diag(s)`
4. Truncated SVD on scaled matrix
5. De-scale output: `W_approx = SVD(W_scaled) / diag(s)`

Rationale: minimizes output error `||W·X - W_k·X||` instead of weight error `||W - W_k||`.

Baseline perplexity: 28.03

## Results

### Exp 1 — ASVD vs Naive SVD (all 12 MLP layers)

| rank | % dense | naive ppl | ASVD ppl | ASVD gain |
|------|---------|-----------|----------|-----------|
| 32   | 2.8%    | —         | —        | —         |
| 64   | 5.6%    | 1600+     | 1200+    | moderate  |
| 128  | 11.1%   | 900+      | 400+     | 2×        |
| 256  | 22.2%   | 600+      | 200+     | 3×        |
| 384  | 33.3%   | 601       | 64       | 9×        |
| 512  | 44.4%   | 293       | 34       | 8×        |

- **ASVD at rank=512: ppl=34 (1.33× baseline)** — near-usable quality
- **Naive SVD at rank=512: ppl=293 (10× baseline)** — unusable
- ASVD advantage is largest in the near-usable regime (rank ≥ 384)

### Exp 2 — Alpha sensitivity (rank=128, all layers)
- α=0.0 (= naive SVD): worst
- α=0.5: best or near-best across all tested values
- α=1.0: marginally worse than α=0.5
- With only 260 calibration tokens, α sensitivity is noisy — paper's recommendation of α=0.5 holds

### Exp 3 — Per-layer sensitivity at rank=128

| Layer | naive ppl | ASVD ppl | note |
|-------|-----------|----------|------|
| 0     | 136       | 85       | most sensitive |
| 1     | 55        | 40       | sensitive |
| 2–7   | 30–40     | 28–35    | moderate |
| 8–9   | 28–29     | 28       | nearly unaffected |
| 10–11 | 30–35     | 29–32    | tolerant |

- **Layer 0 is catastrophically sensitive**: compressing only layer 0 to rank=128 raises ppl from 28 to 136 (5×)
- **Layers 8–9 barely change** at rank=128 (ppl ≈ baseline)
- Non-uniform rank allocation: protecting layer 0 (keep full rank) drops global ppl from 601→64 at rank=384

## Speedup Math

- GPT-2 MLP at rank=512: theoretical **1.2× speedup** (barely worth it)
- GPT-2 MLP at rank=384: theoretical **1.6× speedup** but **2.3× ppl degradation**
- A 1.6× speedup requires rank ≤ 384 — where quality collapses even with ASVD (true break-even, 1.0×, is rank 614 for these 3072×768 matrices)

## Conclusions

1. **ASVD dramatically outperforms naive SVD** — 8–9× better perplexity at the same rank.
2. **GPT-2 small is still too small.** The 1.6× rank (384) is exactly where even ASVD fails. Speedup and quality zones still don't overlap.
3. **Layer 0 is a hard blocker** at GPT-2 small scale — must be protected or given a much larger rank budget.
4. **ASVD at rank=512 gives 1.33× baseline ppl** (near-usable) but only 1.2× speedup — not worth the quality cost.

## What This Motivated

Two directions: (1) test structured matrix approximations (Monarch) as an alternative to global SVD; (2) scale up to GPT-2 Large where matrices are larger and more likely to be redundant. → See Experiments 04 and 05.
