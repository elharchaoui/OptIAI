"""
Experiment 06 — LoRA Recovery After ASVD Compression (GPT-2 Large)
===================================================================
Pipeline:
  1. Load GPT-2 Large (float16, ~1.5 GB VRAM)
  2. Apply ASVD rank=640 to all 36 MLP layers  → ppl: 16.5 → 45.9
  3. Add LoRA adapters (rank=16, ~7M trainable params = 0.95% of model)
  4. Fine-tune on WikiText-2 train
  5. Evaluate on WikiText-2 test after each epoch

Key question: does LoRA recover the 2.78× ppl degradation from ASVD compression?
Expected peak VRAM: ~5 GB.  Expected runtime: ~25 min (3 epochs).
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ── Config ───────────────────────────────────────────────────────────────────
ASVD_RANK    = 640
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["c_fc", "c_proj"]  # hits mlp.c_fc, mlp.c_proj, and attn.c_proj

SEQ_LEN      = 256
BATCH_SIZE   = 4
GRAD_ACCUM   = 4        # effective batch = 16
LR           = 3e-4
WARMUP_STEPS = 100
N_EPOCHS     = 3
EVAL_STEPS   = 50       # optimizer steps between mid-epoch evals

CALIB_SAMPLES = 64      # ASVD calibration chunks from WikiText-2 train

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading GPT-2 Large...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-large", torch_dtype=torch.float16)
model.eval().to(DEVICE)
N_LAYERS = len(model.transformer.h)
print(f"  {N_LAYERS} layers | {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
if DEVICE.type == "cuda":
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Load WikiText-2 (tokenize per article to avoid truncation issues) ─────────
print("\nLoading WikiText-2...")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

def tokenize_and_chunk(split):
    all_ids = []
    for text in raw[split]["text"]:
        if text.strip():
            all_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    ids = torch.tensor(all_ids, dtype=torch.long)
    n = (len(ids) // SEQ_LEN) * SEQ_LEN
    return ids[:n].reshape(-1, SEQ_LEN)

train_chunks = tokenize_and_chunk("train")
test_chunks  = tokenize_and_chunk("test")
calib_chunks = train_chunks[:CALIB_SAMPLES].to(DEVICE)
print(f"  Train: {len(train_chunks):,} chunks  ({len(train_chunks)*SEQ_LEN/1e6:.2f}M tokens)")
print(f"  Test:  {len(test_chunks):,} chunks   ({len(test_chunks)*SEQ_LEN/1e3:.0f}K tokens)")
print(f"  Calib: {CALIB_SAMPLES} chunks from train")

# ── Perplexity helpers ────────────────────────────────────────────────────────
# Fixed text for direct comparison with Experiment 05 numbers
FIXED_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by master "
    "craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe "
    "the process of human thinking as the mechanical manipulation of symbols. This work "
    "culminated in the invention of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning. This device and the ideas behind "
    "it inspired a handful of scientists to begin seriously discussing the possibility of "
    "building an electronic brain. The field of AI research was founded at a workshop held "
    "on the campus of Dartmouth College during the summer of 1956."
)
fixed_ids = tokenizer(FIXED_TEXT, return_tensors="pt",
                      truncation=True, max_length=512)["input_ids"].to(DEVICE)


def ppl_fixed(mdl):
    mdl.eval()
    with torch.no_grad():
        loss = mdl(fixed_ids, labels=fixed_ids).loss
    return torch.exp(loss).item()


def ppl_wikitext(mdl, chunks, max_batches=None):
    mdl.eval()
    total_loss, n = 0.0, 0
    limit = len(chunks) if max_batches is None else min(len(chunks), max_batches * BATCH_SIZE)
    with torch.no_grad():
        for i in range(0, limit, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE].to(DEVICE)
            total_loss += mdl(batch, labels=batch).loss.item()
            n += 1
    return torch.exp(torch.tensor(total_loss / n)).item()


# ── Baseline ─────────────────────────────────────────────────────────────────
print("\nBaseline evaluation...")
base_fixed = ppl_fixed(model)
base_wiki  = ppl_wikitext(model, test_chunks)
print(f"  PPL (fixed text):  {base_fixed:.2f}")
print(f"  PPL (WikiText-2):  {base_wiki:.2f}")

# ── ASVD compression ─────────────────────────────────────────────────────────
print(f"\nApplying ASVD rank={ASVD_RANK} to all {N_LAYERS} MLP layers...")
t0 = time.time()

calib_acts = {i: {"c_fc": [], "c_proj": []} for i in range(N_LAYERS)}
hooks = []
for i, block in enumerate(model.transformer.h):
    def _fc(idx):
        def h(m, inp, out): calib_acts[idx]["c_fc"].append(inp[0].detach().cpu().float())
        return h
    def _cp(idx):
        def h(m, inp, out): calib_acts[idx]["c_proj"].append(inp[0].detach().cpu().float())
        return h
    hooks.append(block.mlp.c_fc.register_forward_hook(_fc(i)))
    hooks.append(block.mlp.c_proj.register_forward_hook(_cp(i)))

with torch.no_grad():
    for ids in calib_chunks:
        model(ids.unsqueeze(0))
for h in hooks:
    h.remove()

for i in range(N_LAYERS):
    for k in ("c_fc", "c_proj"):
        calib_acts[i][k] = torch.cat([a.flatten(0, 1) for a in calib_acts[i][k]], dim=0)


def asvd(W_std, acts, rank, alpha=0.5):
    s = acts.float().abs().mean(0).pow(alpha).clamp(min=1e-8)
    U, Sv, Vh = torch.linalg.svd(W_std.float() * s.unsqueeze(0), full_matrices=False)
    k = min(rank, len(Sv))
    return ((U[:, :k] * Sv[:k]) @ Vh[:k] / s.unsqueeze(0)).to(W_std.dtype)


for i, block in enumerate(model.transformer.h):
    for key, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
        W = conv.weight.detach().T.to(DEVICE)   # Conv1D stores (in, out); transpose → (out, in)
        W_approx = asvd(W, calib_acts[i][key].to(DEVICE), ASVD_RANK)
        conv.weight.data = W_approx.T.to(conv.weight.dtype).cpu().contiguous()

print(f"  Done in {time.time()-t0:.0f}s")
model.to(DEVICE)

comp_fixed = ppl_fixed(model)
comp_wiki  = ppl_wikitext(model, test_chunks)
print(f"  PPL (fixed text):  {comp_fixed:.2f}  ({comp_fixed/base_fixed:.2f}× baseline)")
print(f"  PPL (WikiText-2):  {comp_wiki:.2f}  ({comp_wiki/base_wiki:.2f}× baseline)")
if DEVICE.type == "cuda":
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Add LoRA ──────────────────────────────────────────────────────────────────
print(f"\nAdding LoRA adapters (rank={LORA_RANK}, targets={LORA_TARGETS})...")
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Cast LoRA params to float32 for training stability (base stays float16)
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()

if DEVICE.type == "cuda":
    print(f"  VRAM after LoRA: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Training setup ────────────────────────────────────────────────────────────
loader = DataLoader(TensorDataset(train_chunks), batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01,
)
total_opt_steps = N_EPOCHS * len(loader) // GRAD_ACCUM
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_opt_steps)

steps_per_epoch = len(loader) // GRAD_ACCUM
print(f"\n{'='*60}")
print(f"Training: {N_EPOCHS} epochs | {steps_per_epoch} opt-steps/epoch | {total_opt_steps} total")
print(f"batch={BATCH_SIZE} | grad_accum={GRAD_ACCUM} | effective_batch={BATCH_SIZE*GRAD_ACCUM} | lr={LR}")
print(f"{'='*60}")

history = {
    "base_fixed": base_fixed,
    "base_wiki":  base_wiki,
    "comp_fixed": comp_fixed,
    "comp_wiki":  comp_wiki,
    "epoch_ppl_fixed": [comp_fixed],
    "epoch_ppl_wiki":  [comp_wiki],
    "step_log": [],
}

global_step = 0
t_train = time.time()

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss, n_batches = 0.0, 0
    optimizer.zero_grad()

    for batch_idx, (batch,) in enumerate(loader):
        batch = batch.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch, labels=batch)
        (out.loss / GRAD_ACCUM).backward()
        epoch_loss += out.loss.item()
        n_batches += 1

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % EVAL_STEPS == 0:
                pf = ppl_fixed(model)
                pw = ppl_wikitext(model, test_chunks, max_batches=50)
                elapsed = (time.time() - t_train) / 60
                log = {
                    "step": global_step, "epoch": epoch + 1,
                    "loss": round(epoch_loss / n_batches, 4),
                    "ppl_fixed": round(pf, 2), "ppl_wiki": round(pw, 2),
                    "elapsed_min": round(elapsed, 1),
                }
                history["step_log"].append(log)
                print(f"  [ep{epoch+1} step{global_step:3d}]  "
                      f"loss={log['loss']:.3f}  "
                      f"ppl_fixed={pf:.1f} ({pf/base_fixed:.2f}×)  "
                      f"ppl_wiki={pw:.1f} ({pw/base_wiki:.2f}×)  "
                      f"{elapsed:.0f}min", flush=True)
                model.train()

    # End-of-epoch full eval
    pf = ppl_fixed(model)
    pw = ppl_wikitext(model, test_chunks)
    history["epoch_ppl_fixed"].append(pf)
    history["epoch_ppl_wiki"].append(pw)
    elapsed = (time.time() - t_train) / 60

    print(f"\n  === Epoch {epoch+1}/{N_EPOCHS} complete ({elapsed:.0f} min elapsed) ===")
    print(f"    Train loss:      {epoch_loss/n_batches:.4f}")
    print(f"    PPL fixed:       {pf:.2f}  ({pf/base_fixed:.2f}× baseline)")
    print(f"    PPL WikiText-2:  {pw:.2f}  ({pw/base_wiki:.2f}× baseline)\n", flush=True)

    # Save per-epoch checkpoint
    ckpt_dir = os.path.join(SCRIPT_DIR, f"lora_epoch{epoch+1}")
    model.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump({"epoch": epoch+1, "ppl_fixed": pf, "ppl_wiki": pw,
                   "ratio_fixed": pf/base_fixed, "ratio_wiki": pw/base_wiki}, f, indent=2)
    print(f"  Saved checkpoint → {ckpt_dir}/", flush=True)

# ── Save final history ────────────────────────────────────────────────────────
with open(os.path.join(SCRIPT_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exp 06 — LoRA Recovery After ASVD rank=640 (GPT-2 Large)",
             fontsize=13, fontweight="bold")

epoch_x      = list(range(N_EPOCHS + 1))
epoch_labels = ["Compressed"] + [f"Epoch {i+1}" for i in range(N_EPOCHS)]

for ax, key, base_val, title in [
    (axes[0], "epoch_ppl_fixed", base_fixed, "Fixed eval text (vs Exp 05)"),
    (axes[1], "epoch_ppl_wiki",  base_wiki,  "WikiText-2 test split"),
]:
    vals = history[key]
    ax.plot(epoch_x, vals, marker="o", color="steelblue", linewidth=2, markersize=8)
    ax.axhline(base_val,         color="green",  linestyle="--", linewidth=1.5,
               label=f"Baseline ({base_val:.1f})")
    ax.axhline(base_val * 1.1,   color="orange", linestyle=":",  linewidth=1.2,
               label="1.1× baseline target")
    ax.set_xticks(epoch_x)
    ax.set_xticklabels(epoch_labels, fontsize=9)
    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for x, y in zip(epoch_x, vals):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), fontsize=9, ha="center")

plt.tight_layout()
out_png = os.path.join(SCRIPT_DIR, "results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved plot → {out_png}")

# ── Final summary ─────────────────────────────────────────────────────────────
total_min = (time.time() - t_train) / 60
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
fmt = f"  {{:<25}}  {{:>10}}  {{:>8}}  {{:>10}}  {{:>8}}"
print(fmt.format("Stage", "PPL fixed", "×base", "PPL wiki", "×base"))
print(fmt.format("Baseline",       f"{base_fixed:.2f}", "1.00×", f"{base_wiki:.2f}", "1.00×"))
print(fmt.format("ASVD rank=640",  f"{comp_fixed:.2f}",
                 f"{comp_fixed/base_fixed:.2f}×", f"{comp_wiki:.2f}", f"{comp_wiki/base_wiki:.2f}×"))
for i, (pf, pw) in enumerate(zip(history["epoch_ppl_fixed"][1:],
                                  history["epoch_ppl_wiki"][1:])):
    print(fmt.format(f"  LoRA epoch {i+1}", f"{pf:.2f}",
                     f"{pf/base_fixed:.2f}×", f"{pw:.2f}", f"{pw/base_wiki:.2f}×"))

best_f = min(history["epoch_ppl_fixed"][1:])
best_w = min(history["epoch_ppl_wiki"][1:])
print(f"\nBest recovery:  PPL fixed={best_f:.2f} ({best_f/base_fixed:.2f}×)  "
      f"| PPL wiki={best_w:.2f} ({best_w/base_wiki:.2f}×)")
print(f"Training time:  {total_min:.0f} min")
print(f"Inference speedup maintained: ASVD rank=640 → 1.6× theoretical")
