# Experiment 01 — Toy Matrix Approximations

**Model:** 128×128 random matrices (no real LLM)  
**Script:** `experiment.py`  
**Output:** `results.png`

## Setup

- W: random 128×128 matrix, X: 512 random input samples
- Ground truth: `Y = relu(W·X)`
- Methods tested: low-rank SVD, binary gate (replacing relu), SVD + binary gate combined, error accumulation across L stacked layers
- Metrics: relative L2 error, cosine similarity, wall-clock speedup

## Results

### Exp 1 — Low-rank SVD (relu preserved)
- Rank=64 (full): rel_err ≈ 0, speedup ≈ 1× (baseline)
- Rank=32: rel_err ≈ 0.25, speedup ≈ 1.2×
- Rank=16: rel_err ≈ 0.45, speedup ≈ 1.5×
- No configuration achieves rel_err < 0.10 with meaningful speedup at this scale

### Exp 2 — Binary gate replacing relu
- Gate at p50 (median threshold) ≡ ReLU mathematically: same active set → rel_err ≈ 0.0001
- Gate at p75 (top 25% pass): rel_err rises but sparsity increases
- The gate itself adds no speedup — cost is in the matmul, not the nonlinearity

### Exp 3 — SVD + binary gate combined
- No config beats SVD-alone at the same rank for quality/speed tradeoff
- Binary gate on top of low-rank W doesn't recover the quality lost by rank reduction

### Exp 4 — Error accumulation (12 stacked layers)
- At rank=64 (full rank), error stays near 0 across all layers
- At rank=32: rel_err ≈ 0.25 at L=1, grows to ≈ 0.90 at L=12 — catastrophic
- At rank=16: saturates near 1.0 after just 4 layers

## Conclusions

1. **Binary gate ≡ ReLU** at p50 threshold — the idea is mathematically valid but doesn't add any speedup because the activation is not the bottleneck.
2. **Error accumulation is severe**: a 0.25 error per layer becomes 0.9 after 12 layers. Any compression scheme must keep per-layer error very small (< 0.05) to be viable in a deep network.
3. **128×128 is too small for meaningful low-rank speedup**: the break-even rank (64) is the full rank, so no speedup is possible without quality loss.

## What This Motivated

Scaling up to a real LLM (GPT-2 small) to test whether learned weights have enough low-rank structure that the break-even point becomes achievable. → See Experiment 02.
