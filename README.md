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

- **Experiment 06:** LoRA fine-tuning recovery after ASVD rank=640 on GPT-2 Large — recover from 2.78× to ≤ 1.1× baseline ppl while keeping factored weights
- **Experiment 07:** Scale to LLaMA-7B or Mistral-7B via 4-bit quantized loading (~4 GB VRAM)
- **Experiment 08:** Benchmark actual inference latency with factored linear layers (verify theoretical speedup holds in practice)
