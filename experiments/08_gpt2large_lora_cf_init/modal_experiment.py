"""
Experiment 08 — LoRA Recovery with Closed-Form Initialization (LoftQ-style)
============================================================================
Identical to Exp 06 (ASVD rank=640 + LoRA rank=16, 3 epochs, WikiText-2)
with one change: LoRA weights are initialized from the closed-form correction
instead of the standard zero/random init.

Initialization:
  SVD(W_orig * diag(s)) = U Σ Vᵀ
  ASVD  → top-640 singular vectors (as in Exp 06)
  Init  → next-16 singular vectors:
    lora_A.weight = B · √(r/α)   shape (r, in)
    lora_B.weight = A · √(r/α)   shape (out, r)
  At t=0 the model is at: W_approx + (α/r)·lora_B·lora_A = W_approx + A·B
  i.e. the best rank-16 activation-aware correction of the ASVD residual.

Hypothesis: starting from the closed-form correction rather than zero should
compress the convergence curve — epoch 1 might land where Exp 06 epoch 3 lands.

Compare to Exp 06:
  Epoch 1: 25.57 ppl  (0.93× baseline)
  Epoch 3: 24.53 ppl  (0.89× baseline)  ← target to beat in fewer epochs

Usage:
  uv run modal run experiments/08_gpt2large_lora_cf_init/modal_experiment.py
"""

import math
import os

import modal

app = modal.App("optiai-exp08-lora-cf-init")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "transformers>=4.40.0",
        "peft>=0.9.0",
        "datasets>=2.18.0",
        "matplotlib",
        "accelerate",
        "numpy",
    ])
)

hf_vol     = modal.Volume.from_name("hf-cache",       create_if_missing=True)
result_vol = modal.Volume.from_name("optiai-results",  create_if_missing=True)

RESULTS_DIR = "/results/exp08"
HF_CACHE    = "/cache/hf"


@app.function(
    gpu="t4",
    image=image,
    timeout=7200,
    volumes={HF_CACHE: hf_vol, "/results": result_vol},
)
def train():
    import json, time, math, torch, matplotlib.pyplot as plt
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import DataLoader, TensorDataset

    os.environ["HF_HOME"] = HF_CACHE
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(42)
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n", flush=True)

    # ── Config (identical to Exp 06) ──────────────────────────────────────────
    ASVD_RANK    = 640
    LORA_RANK    = 16
    LORA_ALPHA   = 32
    LORA_DROPOUT = 0.05
    LORA_TARGETS = ["c_fc", "c_proj"]
    SEQ_LEN      = 256
    BATCH_SIZE   = 4
    GRAD_ACCUM   = 4
    LR           = 3e-4
    WARMUP_STEPS = 100
    N_EPOCHS     = 3
    EVAL_STEPS   = 50
    CALIB_SAMPLES = 64
    ALPHA        = 0.5   # ASVD activation scaling exponent

    # Scale factor so that (α/r)·lora_B·lora_A = A·B at init
    # (α/r)·(A·√(r/α))·(B·√(r/α)) = A·B  ✓
    LORA_INIT_SCALE = math.sqrt(LORA_RANK / LORA_ALPHA)   # √(16/32) ≈ 0.707

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading GPT-2 Large...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-large", torch_dtype=torch.float16)
    model.eval().to(DEVICE)
    N_LAYERS = len(model.transformer.h)
    print(f"  {N_LAYERS} layers | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params | "
          f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # ── Load WikiText-2 (identical to Exp 06) ─────────────────────────────────
    print("\nLoading WikiText-2...", flush=True)
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
    print(f"  Train: {len(train_chunks):,} chunks | Test: {len(test_chunks):,} chunks", flush=True)

    # ── Perplexity helpers (identical to Exp 06) ──────────────────────────────
    FIXED_TEXT = (
        "The history of artificial intelligence began in antiquity, with myths, stories "
        "and rumors of artificial beings endowed with intelligence or consciousness by master "
        "craftsmen. The seeds of modern AI were planted by philosophers who attempted to "
        "describe the process of human thinking as the mechanical manipulation of symbols. "
        "This work culminated in the invention of the programmable digital computer in the "
        "1940s, a machine based on the abstract essence of mathematical reasoning."
    )
    fixed_ids = tokenizer(FIXED_TEXT, return_tensors="pt",
                          truncation=True, max_length=512)["input_ids"].to(DEVICE)

    def ppl_fixed(mdl):
        mdl.eval()
        with torch.no_grad():
            return torch.exp(mdl(fixed_ids, labels=fixed_ids).loss).item()

    def ppl_wiki(mdl, chunks, max_batches=None):
        mdl.eval()
        total_loss, n = 0.0, 0
        limit = len(chunks) if max_batches is None else min(len(chunks), max_batches * BATCH_SIZE)
        with torch.no_grad():
            for i in range(0, limit, BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE].to(DEVICE)
                total_loss += mdl(batch, labels=batch).loss.item()
                n += 1
        return torch.exp(torch.tensor(total_loss / n)).item()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\nBaseline evaluation...", flush=True)
    base_fixed = ppl_fixed(model)
    base_wiki  = ppl_wiki(model, test_chunks)
    print(f"  PPL fixed: {base_fixed:.2f}  |  PPL wiki: {base_wiki:.2f}", flush=True)

    # ── Collect calibration activations ───────────────────────────────────────
    print(f"\nApplying ASVD rank={ASVD_RANK} + computing closed-form init...", flush=True)
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

    # ── ASVD + closed-form correction in one SVD per layer ────────────────────
    # Single SVD of W*diag(s) gives both ASVD (top-640) and correction (next-16).
    # No second SVD needed — the correction is literally the residual singular vectors.
    cf_init = {}   # (layer_idx, key) → (A, B) on CPU float32

    def asvd_and_cf(W, acts, asvd_rank, corr_rank, alpha=0.5):
        """
        Returns W_approx (ASVD) and (A, B) for the closed-form correction.
        A: (out, r) left singular vectors of scaled residual
        B: (r, in)  right singular vectors de-scaled
        Correction:  A @ B  (matches W_orig - W_approx in activation-aware sense)
        """
        s    = acts.float().abs().mean(0).pow(alpha).clamp(min=1e-8)
        W_f  = W.float()
        U, Sv, Vh = torch.linalg.svd(W_f * s.unsqueeze(0), full_matrices=False)

        k        = min(asvd_rank, len(Sv))
        W_approx = ((U[:, :k] * Sv[:k]) @ Vh[:k] / s.unsqueeze(0)).to(W.dtype)

        # Next r singular vectors: the activation-aware residual decomposition
        start = k
        end   = min(k + corr_rank, U.shape[1])
        A = U[:, start:end].float()                               # (out, r)
        B = (Sv[start:end].unsqueeze(1) * Vh[start:end]) / s.unsqueeze(0)  # (r, in)
        B = B.float()

        return W_approx, A.cpu(), B.cpu()

    for i, block in enumerate(model.transformer.h):
        for key, conv in [("c_fc", block.mlp.c_fc), ("c_proj", block.mlp.c_proj)]:
            W = conv.weight.detach().T.to(DEVICE)   # Conv1D: (in,out) → transpose → (out,in)
            W_approx, A, B = asvd_and_cf(
                W, calib_acts[i][key].to(DEVICE), ASVD_RANK, LORA_RANK, ALPHA
            )
            conv.weight.data = W_approx.T.to(conv.weight.dtype).cpu().contiguous()
            cf_init[(i, key)] = (A, B)
        if (i + 1) % 12 == 0:
            print(f"  {i+1}/{N_LAYERS} layers processed", flush=True)

    model.to(DEVICE)
    comp_fixed = ppl_fixed(model)
    comp_wiki  = ppl_wiki(model, test_chunks)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    print(f"  PPL fixed: {comp_fixed:.2f} ({comp_fixed/base_fixed:.2f}×)  |  "
          f"PPL wiki: {comp_wiki:.2f} ({comp_wiki/base_wiki:.2f}×)", flush=True)

    # ── Add LoRA ──────────────────────────────────────────────────────────────
    print(f"\nAdding LoRA (rank={LORA_RANK})...", flush=True)
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, task_type=TaskType.CAUSAL_LM, bias="none",
    ))
    model.print_trainable_parameters()

    # Cast LoRA params to float32 (base stays float16)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # ── Closed-form initialization (the only change vs Exp 06) ────────────────
    # At t=0: (α/r)·lora_B·lora_A = A·B = best rank-16 correction of ASVD residual
    print(f"Initializing LoRA weights from closed-form correction "
          f"(scale={LORA_INIT_SCALE:.4f})...", flush=True)
    n_init = 0
    for i, block in enumerate(model.base_model.model.transformer.h):
        for key in ("c_fc", "c_proj"):
            A, B = cf_init[(i, key)]     # (out, r), (r, in) float32 CPU
            lora_layer = getattr(block.mlp, key)
            # lora_A["default"].weight: (r, in_features)
            lora_layer.lora_A["default"].weight.data = (B * LORA_INIT_SCALE).to(DEVICE)
            # lora_B["default"].weight: (out_features, r)
            lora_layer.lora_B["default"].weight.data = (A * LORA_INIT_SCALE).to(DEVICE)
            n_init += 1
    print(f"  Initialized {n_init} LoRA pairs", flush=True)

    # ── Training (identical to Exp 06) ────────────────────────────────────────
    loader = DataLoader(TensorDataset(train_chunks), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01,
    )
    total_steps = N_EPOCHS * len(loader) // GRAD_ACCUM
    scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    print(f"\n{'='*60}", flush=True)
    print(f"Training: {N_EPOCHS} epochs | {len(loader)//GRAD_ACCUM} opt-steps/epoch", flush=True)
    print(f"Init: closed-form (LoftQ-style) vs Exp 06 zero init", flush=True)
    print(f"{'='*60}", flush=True)

    history = {
        "base_fixed": base_fixed, "base_wiki": base_wiki,
        "comp_fixed": comp_fixed, "comp_wiki": comp_wiki,
        # Exp 06 reference numbers for comparison
        "exp06_epoch_ppl_wiki": [153.82, 25.57, 24.70, 24.53],
        "epoch_ppl_fixed": [comp_fixed], "epoch_ppl_wiki": [comp_wiki],
        "step_log": [],
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
                    pw = ppl_wiki(model, test_chunks, max_batches=50)
                    elapsed = (time.time() - t_train) / 60
                    entry = {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": round(epoch_loss / n_batches, 4),
                        "ppl_fixed": round(pf, 2), "ppl_wiki": round(pw, 2),
                        "elapsed_min": round(elapsed, 1),
                    }
                    history["step_log"].append(entry)
                    print(f"  [ep{epoch+1} step{global_step:3d}]  "
                          f"loss={entry['loss']:.3f}  "
                          f"ppl_fixed={pf:.1f} ({pf/base_fixed:.2f}×)  "
                          f"ppl_wiki={pw:.1f} ({pw/base_wiki:.2f}×)  "
                          f"{elapsed:.0f}min", flush=True)
                    model.train()

        pf = ppl_fixed(model)
        pw = ppl_wiki(model, test_chunks)
        history["epoch_ppl_fixed"].append(pf)
        history["epoch_ppl_wiki"].append(pw)
        elapsed = (time.time() - t_train) / 60

        exp06_ref = history["exp06_epoch_ppl_wiki"][epoch + 1]
        print(f"\n  === Epoch {epoch+1}/{N_EPOCHS} ({elapsed:.0f} min) ===")
        print(f"    loss={epoch_loss/n_batches:.4f}  "
              f"ppl_fixed={pf:.2f} ({pf/base_fixed:.2f}×)  "
              f"ppl_wiki={pw:.2f} ({pw/base_wiki:.2f}×)  "
              f"[Exp06 ep{epoch+1}: {exp06_ref:.2f}]\n", flush=True)

        ckpt_dir = os.path.join(RESULTS_DIR, f"lora_epoch{epoch+1}")
        model.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump({"epoch": epoch+1, "ppl_fixed": pf, "ppl_wiki": pw,
                       "ratio_fixed": pf/base_fixed, "ratio_wiki": pw/base_wiki,
                       "exp06_ppl_wiki": exp06_ref}, f, indent=2)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot: Exp 08 vs Exp 06 comparison
    epoch_x = list(range(N_EPOCHS + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Exp 08 vs Exp 06 — LoRA with Closed-Form Init vs Zero Init\n"
                 "ASVD rank=640 + LoRA rank=16 | GPT-2 Large | WikiText-2", fontsize=11)

    ax.plot(epoch_x, history["epoch_ppl_wiki"],     marker="o", color="steelblue",
            linewidth=2, markersize=8, label="Exp 08: CF-init LoRA")
    ax.plot(epoch_x, history["exp06_epoch_ppl_wiki"], marker="s", color="coral",
            linewidth=2, markersize=8, linestyle="--", label="Exp 06: zero-init LoRA")
    ax.axhline(base_wiki, color="green", linestyle=":", linewidth=1.5,
               label=f"Baseline ({base_wiki:.2f})")
    ax.set_xticks(epoch_x)
    ax.set_xticklabels(["Compressed"] + [f"Epoch {i+1}" for i in range(N_EPOCHS)])
    ax.set_ylabel("WikiText-2 Perplexity")
    ax.legend()
    ax.grid(alpha=0.3)
    for x, y in zip(epoch_x, history["epoch_ppl_wiki"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), fontsize=9, ha="center", color="steelblue")
    for x, y in zip(epoch_x, history["exp06_epoch_ppl_wiki"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -16), fontsize=9, ha="center", color="coral")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "results.png"), dpi=150, bbox_inches="tight")

    result_vol.commit()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_min = (time.time() - t_train) / 60
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Stage':<30} {'PPL wiki':>10}  {'×base':>8}  {'Exp06 wiki':>12}")
    print(f"  {'Baseline':<30} {base_wiki:>10.2f}  {'1.00×':>8}  {'27.56':>12}")
    print(f"  {'ASVD rank=640':<30} {comp_wiki:>10.2f}  {comp_wiki/base_wiki:>7.2f}×  {'153.82':>12}")
    for i, (pw, ref) in enumerate(zip(history["epoch_ppl_wiki"][1:],
                                       history["exp06_epoch_ppl_wiki"][1:])):
        delta = ref - pw
        print(f"  {'CF-init LoRA epoch '+str(i+1):<30} {pw:>10.2f}  {pw/base_wiki:>7.2f}×  "
              f"{ref:>12.2f}  (Δ={delta:+.2f})")
    print(f"\nTraining time: {total_min:.0f} min  |  Speedup retained: 1.6× (ASVD rank=640)")
    print(f"Results saved to: {RESULTS_DIR}/", flush=True)


# ── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("Submitting Exp 08 to Modal (T4 GPU, ~55 min, ~$0.55)...")
    train.remote()

    local_dir = os.path.dirname(__file__)
    print(f"\nDownloading results to {local_dir}/...")

    import modal
    for entry in result_vol.listdir("/exp08", recursive=True):
        remote_path = entry.path
        local_path  = os.path.join(local_dir, os.path.relpath(remote_path, "exp08"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            data = b"".join(result_vol.read_file(remote_path))
        except modal.exception.ConflictError:
            continue
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"  ✓ {os.path.relpath(local_path, local_dir)}")

    print("\nDone. Results in experiments/08_gpt2large_lora_cf_init/")
