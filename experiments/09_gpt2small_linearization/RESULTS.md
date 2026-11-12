# Exp 09 — Linearize GPT-2 small: fold blocks into one matrix

**Question.** Can we replace parts of GPT-2 with calibrated linear maps, multiply the
consecutive matrices together, and get a smaller end-to-end computation?

**Answer.** Only in part. A fully linear GPT-2 collapses to one 768×768 matrix but has PPL 892
(baseline 47, probe subset). Folding 3 consecutive blocks into one matrix costs +26% PPL
after recovery, for a measured 1.22× speedup. Folding 6 blocks costs 2.1× PPL for a 1.60× speedup.

## Why a full collapse fails

```
 embed ─▶ [ LN1 ─▶ Wqkv ─▶ softmax(QKᵀ)V ─▶ Wo ─▶ LN2 ─▶ Wfc ─▶ GELU ─▶ Wproj ] ×12 ─▶ ln_f ─▶ unembed
           ✗                ✗ (mixes tokens)          ✗              ✗
```

Matrices between two barriers (✗) multiply into one matrix. If all 49 barriers go, the model
becomes one per-token linear map. It cannot read context, so it behaves about like a bigram model.

## Part A — probe: which barriers are nearly linear? (`probe.py`, `probe2.py`)

Calibration: 32 train chunks. Eval: first 80 test chunks (baseline **47.02**). No training.

| Replaced with a calibrated linear map | 1 layer alone | Cumulative |
|---|---|---|
| Whole block → 768×768 (ridge fit on block input/output) | 48.0–55.2 | 2: 52.9, 3: 72.7, 5: 122.8, 12: 741.8 |
| MLP → 768×768 (ridge fit on MLP input/output) | 48.1–54.9 | 2: 71.6, 5: 98.7, 12: 721.4 |
| GELU → per-channel line (then MLP = 768×768) | 50.9–243.1 | no layer passes a 1.10× budget |
| LN2 → fixed scale | 46.9–50.3 | 6 layers kept, PPL 50.0 (no compute saved) |
| LN1 → fixed scale | 71.8–215 811 | unusable |
| Everything linear (one matrix) | — | **891.7** |

- Each barrier alone looks almost linear. The errors compound when you remove several.
- All MLPs linear with attention intact gives 721: the GELU carries most of the capability.
- LN1 probably fails because of a few tokens with very large norms (likely position 0).
  A single scale cannot fit them. A per-position scale was not tested.
- A local GELU fit is much worse than a fit on the MLP output. Fit to the output you care about.

## Part B — fold k consecutive blocks + distillation recovery (`experiment.py`)

| Setting | Value |
|---|---|
| Model | GPT-2 small (124M), RTX 3060 |
| Fold | blocks i..i+k−1 → one affine map h·M + c (ridge, 64 train chunks) |
| Span choice | lowest PPL on 40 *validation* chunks (test never used for choices) |
| Recovery | 2 epochs WikiText-2 train, loss 0.5·CE + 0.5·KL(teacher‖student) |
| Trainable | LoRA r=16 on all remaining dense matrices + folded M, c |
| Deploy | LoRA merged into dense weights (free: they are full rank) |
| Eval | full WikiText-2 test, 1106 × 256-token chunks (Exp 06–08 protocol) |
| Control | k = 0: same training, no fold |

| k | Span | PPL after fold | PPL after recovery | vs control | Block MACs | Total MACs (+LM head) | Measured fwd |
|---|---|---|---|---|---|---|---|
| — | original GPT-2 | — | 43.53 | — | 1.00× | 1.00× | 91.9 ms |
| 0 | control | — | **28.70** | 1.00× | 1.00× | 1.00× | 0.99× |
| 3 | L4–L6 | 63.94 | **36.21** | +26% | 1.32× less | 1.20× less | **1.22× faster** |
| 6 | L3–L8 | 129.14 | **60.69** | +111% | 1.97× less | 1.52× less | **1.60× faster** |

Forward time: batch 8×256, fp32, after the LoRA merge.

### Findings

1. **Compare against the control, not the original model.** Training on WikiText-2 alone
   takes GPT-2 small from 43.53 to 28.70. The k=3 model (36.21) beats the original only
   because of this domain adaptation. Against the control it loses 26%.
   This confirms the open question about the "surpasses baseline" claims in Exp 06–08.
2. **Contiguous folds beat scattered replacements.** 6 scattered blocks: 265.5 (probe).
   6 contiguous blocks in one matrix: 117.2 (val) before recovery.
3. **The middle layers are the most linear.** The best spans are L4–L6 and L3–L8.
   The first and last layers cost the most (L0–L5: 211, L6–L11: 220).
4. **Recovery closes 68–79% of the gap to the control.** k=3: 63.9 → 36.2 (79%; control 28.7).
   k=6: 129.1 → 60.7 (68%). Val PPL was still falling a little in epoch 2.
5. **Measured speedup matches the MAC count.** The folded blocks are real removed work:
   1.22× and 1.60× wall-clock. The LM head (38.6M MACs/token) caps the total gain.

## Next steps

- More recovery epochs and full fine-tuning instead of LoRA (the k=6 gap is still large).
- Make the folded map slightly nonlinear: h + B·GELU(A·h) with a small rank. This is cheap and may
  recover the context-free loss.
- Per-position LN1 scale, to remove LN barriers inside the remaining blocks.
- Repeat on GPT-2 Large (36 layers: more redundant middle layers).
