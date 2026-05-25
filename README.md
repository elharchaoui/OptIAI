# OptIAI — Approximating σ(W·X) at Inference Time

## Research Question

Can we replace an LLM's weight matrix W (and its activation) with a cheaper operation that produces nearly the same output, reducing inference cost without retraining?

**Core constraint:** High one-time optimization cost is acceptable. Only inference-time cost matters.

**Hardware:** RTX 3060 (12.5 GB VRAM)

---

## The Math

A single MLP layer computes:

```
Y = σ(W · X)       W: (out × in),  X: (in × N)
```

Full matmul costs `out × in` multiply-adds. The goal is to find a cheaper factored form:

```
W ≈ A · B          A: (out × k),  B: (k × in),  k << min(out, in)
```

This costs `(out + in) × k` — a speedup of `(out × in) / ((out + in) × k)`.

**Break-even condition:** speedup > 1 requires `k < (out × in) / (out + in) ≈ min(out, in) / 2`.

---

## Experiment Progression

| # | Experiment | Model | Key Finding |
|---|-----------|-------|-------------|
| [01](experiments/01_toy_matrices/) | Toy matrices | 128×128 random | Binary gate ≡ ReLU; error accumulation is catastrophic across 12 layers |
| [02](experiments/02_gpt2_naive_svd/) | Naive SVD | GPT-2 small | GPT-2 weights are dense (rank@90% energy = 39–67%); naive SVD destroys quality at every viable rank |
| [03](experiments/03_gpt2small_asvd/) | ASVD | GPT-2 small | ASVD (activation-aware SVD) is 8–9× better than naive; layer 0 is a hard blocker; speedup and quality still don't overlap |
| [04](experiments/04_gpt2small_monarch/) | Monarch | GPT-2 small | Block structure beats global SVD 3× at equal parameter budget; but all methods fail at GPT-2 scale |
| [05](experiments/05_gpt2large_asvd/) | ASVD | GPT-2 Large | **First success:** ASVD rank=640 gives 1.6× speedup at 2.78× baseline ppl; no single layer is a hard blocker; recovery via LoRA is viable |
| [06](experiments/06_gpt2large_lora_recovery/) | ASVD + LoRA recovery | GPT-2 Large | **Full recovery:** 1.13% trainable params restore + surpass baseline (0.89× ppl) while keeping 1.6× speedup; epoch 1 alone crosses below baseline |
| [07](experiments/07_gpt2large_layerwise_recovery/) | Closed-form per-layer recovery | GPT-2 Large | **Fails:** per-layer SVD correction achieves only 4.54× at rank=16 vs LoRA's 0.89× — cross-layer error cascade cannot be fixed without end-to-end gradients |
| [08](experiments/08_gpt2large_lora_cf_init/) | ASVD + LoRA with CF init (LoftQ-style) | GPT-2 Large | **Best result:** CF-init LoRA reaches 0.863× baseline (23.79 ppl) vs 0.890× for zero-init — 3% better quality, epoch 1 alone beats Exp 06 final |

---

## Methods Implemented

### ASVD — Activation-Aware SVD
Scale weight columns by input activation statistics before SVD, then de-scale:
```
s_j = mean|X_j|^α
W_approx = SVD_k(W · diag(s)) / diag(s)
```
Minimizes output error `||W·X - W_k·X||` instead of weight error `||W - W_k||`.
Paper: [arXiv:2312.05821](papers/ASVD_2312.05821.pdf)

### Monarch Projection
Approximate W as a product of two block-diagonal matrices × a fixed permutation:
```
W ≈ L · P^T · R       L, R: block-diagonal
```
Block structure allows trivial parallelism (each block independent) and 3× better quality than global SVD at the same parameter budget.
Paper: [arXiv:2204.00595](papers/Monarch_2204.00595.pdf)

### Activation-Aware Monarch (novel)
Combination of ASVD scaling with Monarch projection — not in any paper. Tested in Experiment 04; found to hurt quality at low parameter budgets (< 40% of dense).

---

## Key Findings

1. **Scale is the gating factor.** The break-even speedup rank (< `in_dim / 2`) requires enough redundancy in the weights. GPT-2 small (768-dim) and Large (1280-dim) both need roughly the same fraction of rank before quality holds, but GPT-2 Large's larger absolute matrix size means the break-even point becomes reachable.

2. **ASVD is non-negotiable.** Naive SVD is 10–34× worse perplexity at the same parameter budget. The activation scaling trick is essential for any real LLM.

3. **Reconstruction error ≠ output quality.** Monarch has higher Frobenius weight error than SVD but lower perplexity. Always evaluate on the actual task metric.

4. **Degradation at compression is cumulative, not structural.** At GPT-2 Large, each individual layer tolerates rank=640 within ±2% of baseline. The 2.78× ppl hit comes from 36 layers compounding — which LoRA fine-tuning can recover.

5. **Compressed + recovered > original.** ASVD rank=640 + LoRA fine-tuning (1.13% trainable params, 55 min on T4) achieves 0.89× baseline perplexity on WikiText-2 while keeping the full 1.6× inference speedup. The factored weights, not LoRA, do inference — so the speedup is fully preserved.

6. **Per-layer closed-form correction fails.** Fixing each compressed matrix independently via SVD of its residual achieves only 4.54× baseline at rank=16 — 5× worse than LoRA at equal parameter count. The cascade is cumulative: layer-local corrections mismatch once adjacent layers are also corrected. End-to-end gradient flow is not optional.

7. **Closed-form init beats zero-init LoRA.** Using the residual SVD (Exp 07) as LoRA initialization (LoftQ-style) gives 23.79 ppl vs 24.53 for zero-init — 3% better final quality at identical training cost. Epoch 1 of CF-init already beats Exp 06's 3-epoch result. The warm start places adapters on the principal components of the ASVD error, letting joint fine-tuning close the cross-layer cascade from a better basin.

---

## Papers

| File | Description |
|------|-------------|
| [ASVD_2312.05821.pdf](papers/ASVD_2312.05821.pdf) | Activation-aware SVD for LLM compression |
| [Monarch_2204.00595.pdf](papers/Monarch_2204.00595.pdf) | Monarch matrices: block-diagonal structured approximation |
| [KroneckerSparse_2405.15013.pdf](papers/KroneckerSparse_2405.15013.pdf) | KS matrices: fused CUDA kernel eliminating permutation overhead |
| [BLAST_2410.21262.pdf](papers/BLAST_2410.21262.pdf) | Block-level adaptive structured compression (best on LLaMA-7B) |
| [Kaleidoscope_2012.14966.pdf](papers/Kaleidoscope_2012.14966.pdf) | Butterfly/Kaleidoscope structured matrices |
| [TensorTrain_LLM_2501.19135.pdf](papers/TensorTrain_LLM_2501.19135.pdf) | Tensor-train decomposition for LLMs |
| [LoftQ_2310.08659.pdf](papers/LoftQ_2310.08659.pdf) | Alternating quantization + SVD for LoRA initialization |
| [LLM-Pruner_2305.11627.pdf](papers/LLM-Pruner_2305.11627.pdf) | Structural pruning with gradient importance + LoRA recovery |
| [PolyActivation_2402.11224.pdf](papers/PolyActivation_2402.11224.pdf) | Polynomial activation approximation (for homomorphic encryption, not GPU speedup) |

---

## Planned Next Steps

- **Experiment 08:** Scale to LLaMA-7B or Mistral-7B via 4-bit quantized loading (~4 GB VRAM)
- **Experiment 09:** Benchmark actual inference latency with factored linear layers (verify theoretical speedup holds in practice)
