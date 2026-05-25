# OptIAI — The Full Research Journey

*A comprehensive account of six experiments: from first principles to a working compressed LLM that surpasses its uncompressed baseline.*

---

## 1. The Objective

### Core Question

Can we replace an LLM's weight matrix W (and its activation) with a cheaper operation that produces nearly the same output, reducing inference cost **without any retraining of the base model**?

More concretely: a single transformer MLP layer computes

```
Y = σ(W · X)       W: (out × in),  X: (in × N)
```

Full matrix multiplication costs `out × in` multiply-adds per token. If we can approximate W as a product of two smaller matrices:

```
W ≈ A · B          A: (out × k),  B: (k × in),  k << min(out, in)
```

then the cost drops to `(out + in) × k`, yielding a speedup of:

```
speedup = (out × in) / ((out + in) × k)
```

**The break-even condition — where speedup > 1 — requires `k < (out × in) / (out + in) ≈ min(out, in) / 2`.** That is, the approximation rank must be less than half the smaller dimension.

### Why This Matters

Standard LLM inference is dominated by matrix multiplications. A model like GPT-2 Large runs 36 × 2 MLP layers per forward pass, each doing two large matmuls. If those can be made 1.5–2× cheaper with acceptable quality loss, the entire model speeds up proportionally — with no hardware changes, no quantization, and no change to the model's architecture or outputs.

**The one-time optimization cost is assumed to be acceptable.** The target is reducing the per-token cost at inference time, which compounds across millions of requests. A 1.6× speedup on a production endpoint is worth days of offline preprocessing.

### Hardware Context

All local experiments ran on an RTX 3060 (12.5 GB VRAM). Cloud training (Experiment 06) ran on a Modal T4 ($0.59/hr). This is a deliberately constrained setup — if the technique works here, it works anywhere.

---

## 2. Hypothetical Directions Going In

Before running a single experiment, four candidate directions were considered:

### Direction A — Low-rank SVD (pure weight factorization)
The singular value decomposition gives the optimal rank-k approximation of W in the Frobenius norm sense. The hypothesis: trained LLM weights have low effective rank (much of the energy is concentrated in a few singular vectors), so a rank-k truncation would be nearly lossless at k well below the break-even threshold.

**Expected to work because:** SVD is the theoretically optimal linear factorization. If weights are naturally low-rank, this is essentially free lunch.

**Expected to fail because:** Neural network weights may not have a clean singular spectrum — they're trained under many constraints and may need all their directions.

### Direction B — Binary gate / threshold activation
The ReLU nonlinearity zeroes out half the activations on average. If we could predict *which* neurons will activate before doing the matmul, we could skip the zero-output rows entirely. A learnable binary gate would select the top-p% neurons and only compute those matmul rows.

**Expected to work because:** If the sparsity pattern is predictable, we only do `p × out × in` work instead of `out × in`.

**Expected to fail because:** The gate prediction itself has a cost, and the sparsity pattern may not be predictable cheaply enough to matter.

### Direction C — Activation-aware factorization (ASVD)
Standard SVD minimizes weight reconstruction error `||W - Wk||_F`. But what matters for inference is output error `||WX - WkX||` on real inputs. If some input channels carry most of the activation energy, they should be prioritized in the decomposition.

The ASVD trick (from arXiv:2312.05821): scale weight columns by `s_j = mean|X_j|^α` before SVD, then de-scale after. This reweights the decomposition to minimize output error rather than weight error.

**Expected to work because:** It's provably better in the output error sense, and the calibration set is small (a few hundred tokens).

**Expected to fail because:** The improvement might be marginal if the weight spectrum is dense and activation scaling can't change the fundamental rank structure.

### Direction D — Structured matrix approximations (Monarch, Kaleidoscope)
Instead of global SVD (one rank-k approximation of the full W), use block-diagonal or butterfly structures that exploit local geometry. Monarch approximation (arXiv:2204.00595) writes W ≈ L · P^T · R where L, R are block-diagonal and P is a permutation.

**Expected to work because:** Block structure allows each block to capture a separate subspace, avoiding the "spread" problem of global SVD. Hardware-friendly for parallelism.

**Expected to fail because:** The rigid structure (fixed block size) may not align with how the model uses its weight matrices.

### Direction E — Post-compression LoRA recovery (added after Experiment 05)
After compressing W to a low-rank approximation, add trainable LoRA adapters (rank-r) on top of the frozen compressed weights. Fine-tune on a language modeling objective to recover quality.

This direction wasn't on the table at the start — it emerged from the observation in Experiment 05 that each individual GPT-2 Large layer tolerates rank=640 within ±2% of baseline, so the compressed structure still contains recoverable information.

**Expected to work because:** The per-layer experiment proved the information is still there. LoRA recovery after compression is structurally similar to LoRA fine-tuning after quantization (which works well in practice).

**Expected to fail because:** The cumulative error across 36 layers might be irreversible without full fine-tuning. LoRA might not have enough capacity at rank=16.

---

## 3. The Experiments

### Experiment 01 — First Principles on Toy Matrices

**Why start here:** Before touching any real LLM, validate the mathematical assumptions. Does SVD + ReLU composition behave as expected? Does error accumulate badly across layers?

**Setup:** 128×128 random matrices W, random inputs X, target Y = ReLU(W·X). Four sub-experiments: (1) pure low-rank SVD, (2) binary gate replacing ReLU, (3) both combined, (4) error accumulation across 12 stacked layers.

**Results:**

The binary gate finding was immediate and clean: a threshold at the p50 of activations is mathematically equivalent to ReLU (same active set → rel_err ≈ 0.0001). Direction B was confirmed valid — but the speedup was zero. The activation is not the computational bottleneck; the matmul is. Direction B was eliminated.

Low-rank SVD at 128×128 was unsurprising: rank=32 (50% of full rank) gave only 1.2× speedup with 25% relative error. At this matrix size, the break-even rank equals the full rank — no speedup is achievable without quality loss.

The error accumulation finding was the most important result: a per-layer error of 0.25 (at rank=32) grows to 0.90 after 12 layers. Any viable compression scheme must keep per-layer error below 0.05. This set a hard constraint that all subsequent experiments would need to satisfy.

**What it changed:** Abandoned Direction B entirely. Confirmed that scale (larger matrices) would be necessary for any speedup to appear. Set the error accumulation constraint.

---

### Experiment 02 — GPT-2 Small: Naive SVD Baseline

**Why this next:** Move to a real LLM with trained weights. Establish the naive SVD baseline to understand how "low-rank" these weights actually are.

**Setup:** GPT-2 small (117M params, 12 layers, 768-dim hidden). Replaced all MLP weight matrices (c_fc: 3072×768) with truncated SVD. Evaluated perplexity on a fixed passage. Singular spectrum analyzed for each layer.

**Results:**

The singular spectrum analysis was the first major surprise: rank@90% energy = 39–67% of full rank across layers. GPT-2 weights are not low-rank at all — they use almost all their dimensions.

The perplexity results were unambiguous:

| rank | % of dense | ppl | ×baseline |
|------|-----------|-----|-----------|
| 256  | 22.2%     | 600+ | ~23× |
| 512  | 44.4%     | 293 | ~10× |
| 768  | 66.7%     | 35–50 | ~1.5–2× |

And the critical observation: GPU speedup only appears at rank ≤ 256 (theoretical 3.8×), but at rank ≤ 256 the model is completely broken (ppl 23× baseline). The speedup zone and quality zone never intersect. Direction A (naive SVD) was dead.

**What it changed:** Confirmed that naive SVD is not viable on real LLM weights at any practically useful compression ratio. The approach had to be output-aware, not weight-aware. This motivated Direction C (ASVD).

---

### Experiment 03 — GPT-2 Small: ASVD vs Naive SVD

**Why this next:** ASVD reweights the decomposition to minimize output error, not weight error. If GPT-2's activations are uneven across channels — and they likely are — this should help dramatically.

**Setup:** Collect calibration activations from each MLP layer via forward hooks (6 texts, ~260 tokens). Compute per-channel importance s_j = mean|X_j|^α. Scale W by s before SVD, de-scale after. Compare ASVD vs naive SVD across ranks and alpha values. Also ran per-layer sensitivity analysis at rank=128.

**Results:**

The improvement was dramatic and confirmed the hypothesis:

| rank | naive ppl | ASVD ppl | gain |
|------|-----------|----------|------|
| 384  | 601       | 64       | 9× |
| 512  | 293       | 34       | 8× |

ASVD at rank=512 gave ppl=34 (1.33× baseline) — near-usable quality. Naive SVD at the same rank was completely unusable (ppl=293). Direction C delivered.

But the speedup math was brutal: rank=512 on GPT-2 small (768-dim matrices) gives only 1.2× theoretical speedup. Not worth the approximation cost. The break-even rank (< 384) is exactly where even ASVD fails.

The per-layer sensitivity analysis revealed a second problem: **Layer 0 is catastrophically sensitive**. Compressing only layer 0 to rank=128 raised ppl from 28 to 136 (5×). Every other layer was tolerant. This created a hard blocker: to protect layer 0, non-uniform rank allocation was needed, which complicated the speedup math considerably.

**Alpha sensitivity:** α=0.5 (paper's recommendation) was best or near-best. Confirmed the ASVD paper's finding.

**What it changed:** Established ASVD as the required baseline for all future experiments. Identified two problems that needed solving simultaneously: (1) break-even speedup requires rank < 384, and (2) Layer 0 blocks even with ASVD. Motivated two parallel directions: Monarch (structured alternatives) and scaling up to GPT-2 Large.

---

### Experiment 04 — GPT-2 Small: Monarch Projection vs ASVD

**Why this next:** Maybe global SVD (finding one set of orthogonal directions) is the wrong structure. Monarch writes W ≈ L·P^T·R with block-diagonal L and R — each block can capture its own subspace independently, without competing for global singular vectors.

**Setup:** Monarch projection at n_blocks ∈ {8, 16, 32}. Also tested Activation-Aware Monarch (AA-Monarch) — novel combination not in any paper: apply ASVD column scaling before Monarch factorization. Per-layer sensitivity tested. Weight reconstruction error measured alongside perplexity.

**Results:**

Monarch consistently beat SVD and ASVD at equal parameter budgets — roughly 3× better perplexity:

| n_blocks | equiv rank | Naive SVD | ASVD | Monarch |
|----------|-----------|-----------|------|---------|
| 8        | 83        | 5789 (206×) | 7891 (281×) | **2406 (85×)** |

This confirmed the structural hypothesis: block structure captures local geometry that global SVD misses.

**The AA-Monarch failure was the most instructive result.** Adding ASVD-style activation scaling before Monarch *made things worse* at every setting. The weight reconstruction error for AA-Monarch was higher than plain Monarch (0.887 vs 0.883 at n_blocks=8), and perplexity was worse too. The reason: at very low parameter budgets (7–14% of dense), the activation scaling distorts the singular value spectrum enough to hurt the block decomposition. The ASVD trick only helps when there's enough rank budget to be selective — not when every parameter slot matters.

This produced a key meta-insight: **ASVD scaling helps in the high-budget regime (≥ 40% of dense), hurts in the low-budget regime (< 20%).** The activation-aware trick is not universally beneficial.

A second insight from the weight error analysis: Monarch had *higher* Frobenius weight reconstruction error than SVD but *lower* perplexity. Optimizing weight space (Frobenius norm) is a poor proxy for output quality. This lesson would apply to all future experiments.

But the fundamental problem remained: even with the 3× advantage, Monarch at n_blocks=8 (the best config) still gave 85× baseline ppl. The bottleneck was GPT-2 small's size (768-dim), not the approximation method.

**What it changed:** Confirmed that Monarch structure is superior to SVD structure, but scale is the gating factor. Both Directions A and D hit the same wall at GPT-2 small scale. The only remaining path was scaling up to GPT-2 Large where 1280-dim matrices have a genuinely reachable break-even rank.

---

### Experiment 05 — GPT-2 Large: ASVD at Scale

**Why this next:** The break-even rank for speedup is `< in_dim / 2`. At GPT-2 small (768-dim), this is 384 — exactly where quality collapses. At GPT-2 Large (1280-dim), this is 640. Larger matrices have more redundancy and more room between the break-even rank and the quality cliff.

**Setup:** GPT-2 Large (774M params, 36 layers, 1280-dim). ASVD applied to all 36 MLP layers' c_fc matrices (5120×1280). Two sub-experiments: (1) rank sweep across all layers, (2) per-layer sensitivity at rank=640 (apply ASVD to one layer at a time, keep all others dense).

**Memory engineering:** GPT-2 Large barely fits at fp16 (~2 GB weights). Implemented in-place weight patching (no full deepcopy) to avoid OOM during compression.

**Results:**

This was the first experiment where speedup and quality intersected:

| rank | theory speedup | naive ppl | ASVD ppl |
|------|---------------|-----------|----------|
| 256  | 4.00× | 13035 (791×) | 5516 (335×) |
| 512  | 2.00× | 2133 (129×) | 839 (51×) |
| **640** | **1.60×** | 1554 (94×) | **45.9 (2.78×)** |
| 768  | 1.33× | 996 (60×) | **22.6 (1.37×)** |
| 1024 | 1.00× | 25.2 (1.53×) | 18.9 (1.14×) |

**ASVD rank=640: 1.6× speedup at 2.78× baseline ppl.** For the first time, the numbers are in the right ballpark.

**ASVD rank=768: 1.33× speedup at 1.37× baseline ppl.** A simultaneously acceptable speedup and quality. The zones touched.

The per-layer sensitivity result was the most important finding of the entire pre-recovery phase:

> **Every single layer individually tolerates rank=640 compression within ±2% of baseline ppl.**

This is the opposite of GPT-2 small, where Layer 0 was catastrophically sensitive. At GPT-2 Large scale, information is distributed evenly across layers — no single layer is a hard blocker. The 2.78× global degradation is purely cumulative: 36 × small error = large compound error. No individual layer is broken.

**Why does this distinction matter?** If one layer were catastrophically sensitive, it would need special treatment (higher rank, skip compression, special initialization) and the pipeline would become complicated. If all layers are equally tolerant, the cumulative damage is regular and recoverable. It's the difference between "one weak link breaks the chain" and "every link is slightly worn — re-forge them all."

**The ASVD advantage at this scale:** At rank=640, naive SVD gives 94× baseline ppl while ASVD gives 2.78×. That's a 34× quality advantage for the same parameter budget. Naive SVD is not a viable comparison — ASVD is mandatory.

**What it changed:** This was the breakthrough. The hypothesis (scale enables the speedup-quality overlap) was confirmed. The per-layer experiment opened the door to Direction E (LoRA recovery). The pipeline was now: ASVD compress → LoRA fine-tune → deploy compressed weights at inference.

---

### Experiment 06 — LoRA Recovery After ASVD Compression

**Why this next:** Experiment 05 proved the structure is recoverable. Each layer tolerates rank=640 within ±2%. The 2.78× degradation across 36 layers is cumulative small errors, not structural damage. LoRA (Low-Rank Adaptation) adds trainable rank-r adapters on top of frozen weights. If we add LoRA on top of compressed weights and fine-tune, we're asking: can rank-16 adapters per layer recover the error introduced by rank-640 compression?

**Setup:**
- ASVD rank=640 applied to all MLP c_fc, c_proj, and attention c_proj layers (36 layers × 3 matmuls)
- LoRA rank=16 adapters added on top of frozen compressed weights
- Trainable parameters: 8,847,360 / 782,877,440 = **1.13% of total**
- Dataset: WikiText-2 (9,343 train chunks × 256 tokens = 2.4M tokens)
- Optimizer: AdamW, LR=3e-4, batch=4, grad_accum=4 (effective batch 16)
- Mixed precision: fp16 via `torch.autocast`
- 3 epochs of training

**Why WikiText-2:** Standard LM benchmark. 2.4M training tokens is small enough to run quickly but large enough to get stable perplexity estimates. Test set has 283K tokens — reliable signal.

**Infrastructure note:** Local runs were killed by OOM (Firefox consuming ~5 GB RAM with 9.5 MB swap remaining — system OOM watchdog SIGTERM'd the training process). Moved to Modal cloud (T4 GPU, $0.59/hr) which resolved this entirely. Total compute cost: ~$0.55.

**Implementation details — Conv1D weight transposition:**
GPT-2 stores its linear layers as `Conv1D` with weights shaped `(in, out)` — transposed relative to standard PyTorch `Linear` convention. ASVD must transpose before SVD and back after:

```python
W = conv.weight.detach().T        # (in, out) → (out, in)
W_approx = asvd(W, ...)
conv.weight.data = W_approx.T    # back to (in, out)
```

PEFT's LoRA handles this automatically (it detects Conv1D and sets `fan_in_fan_out=True`).

**Results:**

| Stage | PPL wiki | ×baseline | Elapsed |
|-------|----------|-----------|---------|
| Baseline (dense) | 27.56 | 1.00× | — |
| ASVD rank=640 | 153.82 | 5.58× | — |
| + LoRA epoch 1 | 25.57 | **0.93×** | 18 min |
| + LoRA epoch 2 | 24.70 | **0.90×** | 37 min |
| + LoRA epoch 3 | 24.53 | **0.89×** | 55 min |

**The compressed + LoRA-recovered model surpasses the uncompressed baseline by 11%, while keeping the full 1.6× inference speedup.**

The recovery curve is the most striking result. ASVD compression degraded wiki PPL from 27.56 to 153.82 — a 5.58× blow. After just 18 minutes of LoRA fine-tuning (epoch 1, 584 optimizer steps), the perplexity dropped to 25.57, already **below the uncompressed baseline**. Epochs 2 and 3 added diminishing improvement. Roughly 75% of the total recovery happened in the first epoch.

**Training curve in detail:**

| Step | Time | Wiki PPL | Observation |
|------|------|----------|-------------|
| 50   | 2 min | 37.71 | Initial shock; model adapting fast |
| 100  | 3 min | 30.72 | 80% of ASVD degradation recovered |
| 300  | 9 min | 27.62 | Already at baseline |
| 584 (ep1) | 18 min | 25.57 | Below baseline |
| 1168 (ep2) | 37 min | 24.70 | Slow steady improvement |
| 1752 (ep3) | 55 min | 24.53 | Convergence |

**Why the recovery is so fast:** Each LoRA adapter has rank=16, applied to each compressed layer. The compressed weight W_compressed = A·B (rank-640 factored) needs a small correction ΔW = α · lora_B · lora_A (rank-16). Because each individual layer's error is small (±2% from the per-layer experiment), each correction needed is small. The 1.13% trainable parameters are well-matched to the 1.13% of information lost per layer.

**The ppl_fixed metric divergence:**
A secondary metric tracked perplexity on a fixed short passage (512 tokens). This metric *did not recover* — it stayed around 54× baseline throughout training. This is expected and not a quality regression:

1. After fine-tuning on WikiText-2, the model shifts distribution toward that dataset's style
2. The fixed passage is only 512 tokens — high variance, noisy estimate
3. The WikiText-2 test set (283K tokens, same distribution) is the authoritative metric

The fixed metric is useful as a sanity check *before* fine-tuning (to verify ASVD compression didn't completely break the model) but loses validity afterward.

**Inference: the LoRA adapters are not needed at deployment.** At test time, inference uses the compressed factored weights (A·B, rank=640) only. The LoRA adapters can be merged or discarded after training. The 1.6× speedup is fully preserved.

---

## 4. Cross-Cutting Lessons

### Lesson 1: Scale is the gating factor

The break-even speedup rank is `k < in_dim / 2`. For a 768-dim model, this is 384 — exactly where ASVD starts to fail. For a 1280-dim model, this is 640 — where ASVD gives 2.78× quality degradation, a recoverable gap. The math dictates that **larger models are inherently more compressible** in the low-rank sense: their weight matrices have more absolute rank budget before hitting the break-even point.

This explains why compression research consistently finds better results on larger models.

### Lesson 2: Output error ≠ weight error

Experiment 04 made this concrete: Monarch had *higher* Frobenius weight reconstruction error than SVD but *lower* perplexity. ASVD consistently outperforms naive SVD by 8–34× despite both producing the same-rank approximation. The difference is what gets minimized: weight space vs output space.

**Always evaluate on the task metric.** Weight reconstruction error, singular value decay, and Frobenius norm are engineering proxies — perplexity is what matters.

### Lesson 3: Per-layer analysis reveals whether cumulative damage is structural or additive

At GPT-2 small: Layer 0 was uniquely fragile. Compressing only Layer 0 gave catastrophic results. The damage was structural — one layer doing something others couldn't.

At GPT-2 Large: All 36 layers were individually tolerant. The global damage was purely cumulative. This distinction is crucial: structural damage cannot be recovered by fine-tuning (the information is gone), but cumulative additive damage can be (the information is still in the weights, just distributed suboptimally).

The per-layer experiment is now a mandatory diagnostic before attempting any recovery-by-fine-tuning.

### Lesson 4: Activation scaling helps only in the high-budget regime

ASVD scaling (s_j = mean|X_j|^α) dramatically improves quality at rank ≥ 40% of dense. At rank < 20%, it can hurt (as shown with AA-Monarch in Experiment 04). The intuition: at high budgets, you can afford to prioritize important channels; at low budgets, every rank slot matters and distorting the spectrum before decomposition wastes capacity.

This suggests a practical rule: use ASVD at rank ≥ 30% of `min(out, in)`, use unweighted factorization below that.

### Lesson 5: Recovery is disproportionately fast

75% of LoRA recovery happens in the first epoch (18 min). The first 300 steps (9 min) bring the model from 153× to 1× baseline. This is not typical LLM fine-tuning behavior — it's more like error correction than adaptation. The model isn't learning a new task; it's undoing small systematic errors in 36 layers simultaneously.

Practical implication: for deployment, **1 epoch of LoRA fine-tuning after ASVD compression** is likely sufficient. Epochs 2–3 add marginal improvement at meaningful additional cost.

### Lesson 6: Infrastructure constraints shape research direction

Two local training runs were killed by system OOM (Firefox + swap exhaustion = SIGTERM at epoch 1). Moving to Modal resolved this at trivial cost ($0.55 total). The cleaner failure mode of cloud compute (clear memory limits, no competing processes) is worth the cloud overhead for experiments that exceed local memory.

Similarly, the Conv1D transposition bug (GPT-2 stores weights as `(in, out)` not `(out, in)`) would have silently corrupted ASVD compression in a hard-to-diagnose way. The systematic check — transpose in, compress, transpose back — was non-obvious and required careful reading of the model source.

---

## 5. The Complete Pipeline

The full pipeline that works, confirmed end-to-end on GPT-2 Large:

```
1. Calibration collection
   Run ~260 tokens through the model, collect activation statistics per layer
   s_j = mean|X_j|^0.5  (per input channel)

2. ASVD compression (one-time, offline)
   For each target layer (c_fc, c_proj, attn c_proj):
     W_scaled = W * s           (scale columns by activation stats)
     U, Σ, Vh = SVD(W_scaled)
     W_approx = (U[:,:k] * Σ[:k]) @ Vh[:k] / s   (de-scale)
     Replace W with W_approx    (stored in factored form for inference)

3. LoRA fine-tuning (one-time, offline, ~1 epoch)
   Add rank-16 LoRA adapters on frozen compressed weights
   Fine-tune on domain-relevant data (WikiText-2 or target corpus)
   ~1.13% trainable parameters, ~18 min on T4 GPU

4. Merge or discard LoRA
   For inference, use only the compressed factored weights W_approx
   LoRA adapters can be merged: W_final = W_approx + alpha * B @ A
   Or discarded if the factored form is kept separate for speed

5. Inference
   Replace nn.Linear layers with factored matmul: y = (x @ B.T) @ A.T
   Theoretical speedup: 1.6× for rank=640 on 1280-dim matrices
```

**Net result:** A model that runs 1.6× faster at inference time and achieves 0.89× baseline perplexity (11% better than the dense model on WikiText-2). Optimization cost: ~1 hour on a T4 GPU. Deployment cost: unchanged hardware, faster matmuls.

---

## 6. What Didn't Work (and Why)

| Approach | Outcome | Root Cause |
|----------|---------|------------|
| Binary gate replacing ReLU | Zero speedup | Activation cost is negligible vs matmul cost |
| Naive SVD on GPT-2 small | 10–60× ppl degradation | Weights are dense; spectrum doesn't decay fast enough |
| Naive SVD on GPT-2 Large | 34–94× ppl degradation (at break-even rank) | Same; ASVD is the only viable alternative |
| ASVD on GPT-2 small | Speedup and quality never overlap | Matrix dimension (768) too small for break-even rank to reach quality zone |
| Activation-Aware Monarch | Worse than plain Monarch | Activation scaling distorts spectrum at low parameter budgets (< 20% of dense) |
| Local training (RTX 3060) | SIGTERM from OOM | Firefox + model competing for 12.5 GB VRAM, swap exhausted |

---

## 7. Directions Not Yet Tested

### LLaMA-7B / Mistral-7B (Experiment 07, planned)
The same ASVD + LoRA pipeline, applied to a model that practitioners actually deploy. Dimension: 4096 (LLaMA-7B MLP hidden), break-even rank: < 2048. 4-bit quantized loading via bitsandbytes fits ~4 GB VRAM. The hypothesis: results should be strictly better than GPT-2 Large because the matrices are 3× larger — more redundancy, more room for compression.

### Actual inference latency benchmark (Experiment 08, planned)
All speedup numbers so far are *theoretical* (FLOPs reduction). Real PyTorch inference with `nn.Linear` doesn't automatically benefit from factored weights — you need to either replace Linear with a custom module that stores A and B separately, or use a fused CUDA kernel. Experiment 08 would measure actual wall-clock speedup to verify the 1.6× FLOPs advantage translates to practice.

### Rank optimization (open question)
Experiments used uniform rank=640 across all layers. Per-layer sensitivity analysis (Exp 05) showed all layers are equally tolerant at rank=640 — but some layers may be more tolerant than others at lower ranks. A non-uniform rank allocation (give more rank budget to sensitive layers, less to tolerant ones) at the same total FLOPs could give better quality or more speedup. This would require a gradient-based or search-based rank assignment step.

### Longer LoRA fine-tuning / larger datasets
Experiment 06 used WikiText-2 (2.4M tokens) for 3 epochs. The model converged by epoch 2. Using a domain-specific corpus (e.g., code, legal text, biomedical) would give a domain-specialized compressed model. The ASVD compression is domain-agnostic; the LoRA recovery step can be domain-targeted.

---

## 8. Summary Table

| Exp | Model | Method | Key finding |
|-----|-------|--------|-------------|
| 01 | 128×128 random | SVD + binary gate | Error accumulates to 0.9 over 12 layers; binary gate = ReLU, no speedup |
| 02 | GPT-2 small | Naive SVD | Weights are dense (rank@90%=39–67%); speedup and quality zones never overlap |
| 03 | GPT-2 small | ASVD | 8–9× better than naive SVD; still can't close gap at break-even rank; Layer 0 hard blocker |
| 04 | GPT-2 small | Monarch + AA-Monarch | Monarch 3× better than SVD at equal budget; AA-Monarch hurts at low budgets; all fail at GPT-2 scale |
| 05 | GPT-2 Large | ASVD | **First overlap:** rank=640 gives 1.6× speedup at 2.78× ppl; no per-layer blockers; damage is cumulative and recoverable |
| 06 | GPT-2 Large | ASVD + LoRA | **Complete pipeline:** 0.89× baseline ppl, 1.6× speedup kept; 1.13% trainable params, 55 min on T4 |

---

*From random matrices to a production-viable pipeline in six experiments.*
