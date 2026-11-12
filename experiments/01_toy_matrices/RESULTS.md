# Experiment 01 — Toy Matrix Approximations

**Model:** 128×128 random matrices (no real LLM)  
**Script:** `experiment.py`  
**Output:** `results.png`

## Setup

- W: random 128×128 matrix, X: 512 random input samples
- Ground truth: `Y = relu(W·X)`
- Methods tested: low-rank SVD, binary gate (replacing relu), SVD + binary gate combined, error accumulation across L stacked layers
- Metrics: relative L2 error, cosine similarity, wall-clock speedup
- **Timing caveat:** `low_rank()` returns the dense 128×128 product `U·Σ·Vh`, so every rank runs the same dense matmul. The measured "speedup" is timing noise (0.86–1.09× across runs), not a low-rank effect.

## Results

### Exp 1 — Low-rank SVD (relu preserved)
| rank | rel_err | cos sim | theoretical speedup `128² / (256·k)` |
|------|---------|---------|------|
| 128 (full) | 0.000 | 1.000 | 0.5× |
| 64 | 0.310 | 0.951 | 1.0× (break-even) |
| 32 | 0.569 | 0.823 | 2.0× |
| 16 | 0.728 | 0.690 | 4.0× |
| 8 | 0.823 | 0.579 | 8.0× |

- Only full rank (128) reaches rel_err < 0.10, and full rank cannot give a speedup.
- A random Gaussian matrix has a flat singular spectrum, so it has no low-rank structure to exploit.

### Exp 2 — Binary gate replacing relu
- Gate at p50 (median threshold) ≡ ReLU mathematically: same active set → rel_err ≈ 0.0001
- Gate at p75 (top 25% pass): rel_err rises but sparsity increases
- The gate gives no speedup: the matmul is still dense. As implemented, it is about 100× slower than ReLU (0.01×), because `torch.quantile` is expensive.

### Exp 3 — SVD + binary gate combined
- No config beats SVD-alone at the same rank for quality/speed tradeoff
- Binary gate on top of low-rank W doesn't recover the quality lost by rank reduction

### Exp 4 — Error accumulation (12 stacked layers)
| rank | L=1 | L=2 | L=4 | L=8 | L=12 |
|------|-----|-----|-----|-----|------|
| 64 | 0.30 | 0.41 | 0.57 | 0.56 | 0.76 |
| 32 | 0.56 | 0.70 | 0.86 | 0.92 | 0.98 |
| 16 | 0.72 | 0.83 | 0.95 | 0.99 | 1.00 |

- At rank=64: rel_err 0.30 at L=1 grows to 0.76 at L=12
- At rank=32: 0.56 at L=1 grows to 0.98 at L=12 — catastrophic
- At rank=16: above 0.95 after just 4 layers

## Conclusions

1. **Binary gate ≡ ReLU** at p50 threshold — the idea is mathematically valid but doesn't add any speedup because the activation is not the bottleneck.
2. **Error accumulation is severe**: a 0.30 error per layer (rank 64) becomes 0.76 after 12 layers; 0.56 (rank 32) becomes 0.98. Any compression scheme must keep per-layer error very small to be viable in a deep network. (The < 0.05 target used later is a rule of thumb, not a measured threshold.)
3. **Random 128×128 matrices show no usable low-rank trade-off**: break-even is rank 64 (half of full rank 128), and rank 64 already gives 0.31 error. The cause is the flat spectrum of random matrices, not only the size. Real trained weights may have more structure, which Experiment 02 tests.

## What This Motivated

Scaling up to a real LLM (GPT-2 small) to test whether learned weights have enough low-rank structure to beat break-even with acceptable error. → See Experiment 02.
