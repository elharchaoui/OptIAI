---
base_model: gpt2-large
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - gpt2
  - asvd
  - compression
  - peft
  - transformers
license: mit
---

# GPT-2 Large — ASVD-compressed + LoRA recovery adapter

LoRA adapter (rank=16) trained to recover quality after ASVD (activation-aware SVD) rank=640 compression of GPT-2 Large. The compressed model achieves **1.6× inference speedup** with perplexity **11% below the uncompressed baseline** on WikiText-2.

## Results

| Stage | WikiText-2 PPL | vs baseline |
|-------|---------------|-------------|
| GPT-2 Large (baseline) | 27.56 | 1.00× |
| + ASVD rank=640 | 153.82 | 5.58× worse |
| + this LoRA adapter | **24.53** | **0.89× (11% better)** |

**Inference speedup retained: 1.6×** (ASVD factored weights, not LoRA, do inference).

## How it works

1. **ASVD compression**: each MLP/attention projection weight matrix `W` (5120×1280) is replaced by a rank-640 factorisation `U·Σ·Vᵀ`, weighted by input activation statistics before SVD. This reduces matmul cost by 1.6×.
2. **LoRA recovery**: rank-16 adapters `A·B` are added on top of frozen compressed weights, trained for 3 epochs on WikiText-2. Only 8.85M params (1.13% of model) are trainable.
3. At inference, the LoRA weights can be merged into the factored form — the speedup is preserved.

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# Load compressed base (you must apply ASVD first — see repo for script)
# OR load the full model and merge adapter weights normally
base = GPT2LMHeadModel.from_pretrained("gpt2-large")
model = PeftModel.from_pretrained(base, "elharchaoui/gpt2-large-asvd640-lora")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

inputs = tokenizer("The history of machine learning", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0]))
```

> **Note:** This adapter was trained on top of ASVD-compressed weights. Loading it on an uncompressed GPT-2 Large will work (the adapter corrects for the compression error) but the inference speedup requires the factored weights. See the [OptIAI repository](https://github.com/elharchaoui/OptIAI) for the full compression + inference pipeline.

## Training details

- **Base model:** `gpt2-large` (774M params)
- **Compression:** ASVD rank=640 on all `c_fc`, `c_proj` layers (MLP + attention projection), 36 layers
- **LoRA config:** rank=16, alpha=32, dropout=0.05, targets: `c_fc`, `c_proj`
- **Dataset:** WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`), 9,343 train chunks × 256 tokens
- **Training:** 3 epochs, batch=4, grad_accum=4 (effective=16), LR=3e-4, mixed precision
- **Hardware:** Tesla T4 (Modal), 55 min

## Perplexity vs training time

| Epoch | Time | PPL wiki |
|-------|------|----------|
| (post-ASVD, pre-LoRA) | 0 min | 153.82 |
| 1 | 18 min | 25.57 |
| 2 | 37 min | 24.70 |
| **3** | **55 min** | **24.53** |

Recovery is fast: epoch 1 already crosses below the uncompressed baseline (27.56).

## Citation / context

Part of the **OptIAI** research project investigating cheap inference-time approximations of LLM weight matrices via structured decomposition.
- ASVD paper: [arXiv:2312.05821](https://arxiv.org/abs/2312.05821)
