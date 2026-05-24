"""
Experiment 06 — LoRA Recovery After ASVD Compression (GPT-2 Large)
===================================================================
Pipeline:
  1. Load GPT-2 Large (float16, ~1.5 GB VRAM)
  2. Apply ASVD rank=640 to all 36 MLP layers  → ppl: 16.5 → 45.9
  3. Add LoRA adapters (rank=16, ~7M trainable params = 0.95% of model)
  4. Fine-tune on WikiText-2 train (~21 min/epoch on RTX 3060)
  5. Evaluate on WikiText-2 test after each epoch

Key question: does LoRA recover the 2.78× ppl degradation from ASVD compression?
Expected peak VRAM: ~5 GB.  Expected runtime: ~90 min (3 epochs).
"""

import os
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
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ── Config ───────────────────────────────────────────────────────────────────
ASVD_RANK     = 640
LORA_RANK     = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["c_fc", "c_proj"]   # MLP layers (also hits attn.c_proj)

SEQ_LEN       = 256
BATCH_SIZE    = 4
GRAD_ACCUM    = 4       # effective batch = 16
LR            = 3e-4
WARMUP_STEPS  = 100
N_EPOCHS      = 3
EVAL_INTERVAL = 200     # optimizer steps

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

# ── Load WikiText-2 ───────────────────────────────────────────────────────────
print("\nLoading WikiText-2...")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

def tokenize_and_chunk(split, max_chunks=None):
    text = "\n".join(raw[split]["text"])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = (len(ids) // SEQ_LEN) * SEQ_LEN
    chunks = ids[:n].reshape(-1, SEQ_LEN)
    return chunks if max_chunks is None else chunks[:max_chunks]

train_chunks = tokenize_and_chunk("train")
test_chunks  = tokenize_and_chunk("test")
calib_chunks = train_chunks[:CALIB_SAMPLES].to(DEVICE)
print(f"  Train: {len(train_chunks):,} chunks  ({len(train_chunks)*SEQ_LEN/1e6:.1f}M tokens)")
print(f"  Test:  {len(test_chunks):,} chunks  ({len(test_chunks)*SEQ_LEN/1e3:.0f}K tokens)")
print(f"  Calib: {CALIB_SAMPLES} chunks from train")

# ── Perplexity helpers ────────────────────────────────────────────────────────
EVAL_TEXT = (
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
fixed_ids = tokenizer(EVAL_TEXT, return_tensors="pt",
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
            batch = chunks[i:i+BATCH_SIZE].to(DEVICE)
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

# Collect calibration activations
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
        W = conv.weight.detach().T.to(DEVICE)           # Conv1D: weight is (in, out) → transpose to (out, in)
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

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training: {N_EPOCHS} epochs | batch={BATCH_SIZE} | grad_accum={GRAD_ACCUM} | lr={LR}")
print(f"Optimizer steps: {total_opt_steps} | ~{len(loader)//GRAD_ACCUM} steps/epoch")
print(f"{'='*60}")

history = {
    "epoch_ppl_fixed": [comp_fixed],
    "epoch_ppl_wiki":  [comp_wiki],
    "step_ppl_fixed":  [],
    "step_ppl_wiki":   [],
    "step":            [],
    "train_loss":      [],
}

global_step = 0
t_train = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss, n_batches = 0.0, 0
    optimizer.zero_grad()

    for batch_idx, (batch,) in enumerate(loader):
        batch = batch.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch, labels=batch)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
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

            if global_step % EVAL_INTERVAL == 0:
                pf = ppl_fixed(model)
                pw = ppl_wikitext(model, test_chunks, max_batches=50)
                elapsed = (time.time() - t_train) / 60
                print(f"  [ep{epoch+1} step{global_step}]  "
                      f"loss={epoch_loss/n_batches:.3f}  "
                      f"ppl_fixed={pf:.1f} ({pf/base_fixed:.2f}×)  "
                      f"ppl_wiki={pw:.1f} ({pw/base_wiki:.2f}×)  "
                      f"{elapsed:.0f}min")
                history["step"].append(global_step)
                history["train_loss"].append(epoch_loss / n_batches)
                history["step_ppl_fixed"].append(pf)
                history["step_ppl_wiki"].append(pw)
                model.train()

    # End-of-epoch full eval
    pf = ppl_fixed(model)
    pw = ppl_wikitext(model, test_chunks)
    history["epoch_ppl_fixed"].append(pf)
    history["epoch_ppl_wiki"].append(pw)
    elapsed = (time.time() - t_train) / 60
    print(f"\n  Epoch {epoch+1}/{N_EPOCHS} complete  ({elapsed:.0f} min elapsed)")
    print(f"    Train loss:      {epoch_loss/n_batches:.4f}")
    print(f"    PPL fixed:       {pf:.2f}  ({pf/base_fixed:.2f}× baseline)")
    print(f"    PPL WikiText-2:  {pw:.2f}  ({pw/base_wiki:.2f}× baseline)\n")

# ── Save LoRA weights ─────────────────────────────────────────────────────────
save_dir = os.path.join(os.path.dirname(__file__), "lora_weights")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
print(f"Saved LoRA adapter weights → {save_dir}/")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exp 06 — LoRA Recovery After ASVD rank=640 (GPT-2 Large)",
             fontsize=13, fontweight="bold")

epoch_labels = ["Compressed"] + [f"Epoch {i+1}" for i in range(N_EPOCHS)]
epoch_x = list(range(N_EPOCHS + 1))

for ax, key, base_val, title in [
    (axes[0], "epoch_ppl_fixed", base_fixed, "Fixed eval text (comparability with Exp 05)"),
    (axes[1], "epoch_ppl_wiki",  base_wiki,  "WikiText-2 test split"),
]:
    vals = history[key]
    ax.plot(epoch_x, vals, marker="o", color="steelblue", linewidth=2, markersize=8, label="LoRA recovery")
    ax.axhline(base_val,        color="green",  linestyle="--", linewidth=1.5, label=f"Baseline ({base_val:.1f})")
    ax.axhline(base_val * 1.1,  color="orange", linestyle=":",  linewidth=1.2, label="1.1× baseline target")
    ax.axhline(base_val * 2.78, color="tomato", linestyle=":",  linewidth=1.0, label="ASVD start (2.78×)")
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
out_png = os.path.join(os.path.dirname(__file__), "results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved plot → {out_png}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_min = (time.time() - t_train) / 60
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Stage':<28}  {'PPL (fixed)':>11}  {'×base':>7}  {'PPL (wiki)':>11}  {'×base':>7}")
print(f"{'Baseline':<28}  {base_fixed:>11.2f}  {'1.00×':>7}  {base_wiki:>11.2f}  {'1.00×':>7}")
print(f"{'ASVD rank=640':<28}  {comp_fixed:>11.2f}  {comp_fixed/base_fixed:>6.2f}×  {comp_wiki:>11.2f}  {comp_wiki/base_wiki:>6.2f}×")
for i, (pf, pw) in enumerate(zip(history["epoch_ppl_fixed"][1:], history["epoch_ppl_wiki"][1:])):
    tag = f"LoRA epoch {i+1}"
    print(f"  {tag:<26}  {pf:>11.2f}  {pf/base_fixed:>6.2f}×  {pw:>11.2f}  {pw/base_wiki:>6.2f}×")

best_fixed = min(history["epoch_ppl_fixed"][1:])
best_wiki  = min(history["epoch_ppl_wiki"][1:])
print(f"\nBest recovery:  PPL fixed={best_fixed:.2f} ({best_fixed/base_fixed:.2f}× baseline)"
      f"  |  PPL wiki={best_wiki:.2f} ({best_wiki/base_wiki:.2f}× baseline)")
print(f"Training time: {total_min:.0f} min")
print(f"Speedup maintained: ASVD rank=640 → 1.6× theoretical speedup at inference")
