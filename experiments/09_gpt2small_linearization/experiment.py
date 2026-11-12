"""
Exp 09 — Fold consecutive GPT-2 small blocks into one matrix, then recover.

Pipeline (per config k = number of consecutive blocks to fold)
--------
1. Span search  : for every start i, fit one affine map  h_out(i+k-1) ~ h_in(i) M + c
                  (ridge least squares on calibration data). Pick the span with the
                  lowest PPL on the WikiText-2 *validation* split (test is never used for choices).
2. Fold         : block i -> Affine(M, c), blocks i+1..i+k-1 -> Skip. The k blocks
                  are now one 768x768 matrix.
3. Recover      : distill from the original model. Loss = 0.5*CE + 0.5*KL(teacher||student).
                  Trainable: LoRA r=16 on all remaining dense matrices + the folded M, c.
4. Deploy       : merge LoRA into the dense weights (free: they are full-rank), measure
                  test PPL, MACs/token and real forward time.

Control: k = 0 (no fold), same training. It shows how much of the gain is the training itself.

Eval protocol = Exp 06-08: WikiText-2 test, 256-token chunks, mean chunk loss.
"""

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

p = argparse.ArgumentParser()
p.add_argument("--ks", type=int, nargs="+", default=[0, 3, 6])
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--out", default="results.json")
p.add_argument("--max-steps", type=int, default=0, help="smoke test: steps per epoch")
args = p.parse_args()

torch.manual_seed(0)
DEV = "cuda"
SEQ, BS = 256, 8
CALIB, VAL = 64, 40                 # chunks
RIDGE = 1e-3
LORA_R, LR_LORA, LR_AFFINE = 16, 2e-4, 5e-5
D, FF, V = 768, 3072, 50257

tok = GPT2TokenizerFast.from_pretrained("gpt2")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


def chunks(split):
    ids = []
    for t in raw[split]["text"]:
        if t.strip():
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
    ids = torch.tensor(ids)
    return ids[: len(ids) // SEQ * SEQ].view(-1, SEQ)


train_ch = chunks("train")
val_ch = chunks("validation")[:VAL].to(DEV)
test_ch = chunks("test").to(DEV)
calib_ch = train_ch[:CALIB].to(DEV)
print(f"train {len(train_ch)} | val {len(val_ch)} | test {len(test_ch)} chunks", flush=True)


def load():
    m = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV)
    m.config.use_cache = False
    return m.eval()


@torch.no_grad()
def ppl(m, data):
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), BS):
        b = data[i:i + BS]
        tot += m(b, labels=b).loss.item(); n += 1
    return math.exp(tot / n)


class Affine(nn.Module):
    """k folded blocks: residual stream h -> h M + c."""
    def __init__(self, M, c):
        super().__init__()
        self.M, self.c = nn.Parameter(M), nn.Parameter(c)

    def forward(self, x, *a, **k):
        return x @ self.M + self.c


class Skip(nn.Module):
    def forward(self, x, *a, **k):
        return x


@torch.no_grad()
def fit_span(m, i, k):
    blocks = m.transformer.h
    xs, ys = [], []

    def pre(mod, a, kw):
        x = a[0] if a else kw["hidden_states"]
        xs.append(x.reshape(-1, D))

    def post(mod, a, o):
        ys.append(o.reshape(-1, D))

    h1 = blocks[i].register_forward_pre_hook(pre, with_kwargs=True)
    h2 = blocks[i + k - 1].register_forward_hook(post)
    for j in range(0, len(calib_ch), BS):
        m(calib_ch[j:j + BS])
    h1.remove(); h2.remove()
    X, Y = torch.cat(xs).double(), torch.cat(ys).double()
    Xa = torch.cat([X, torch.ones(len(X), 1, device=DEV, dtype=X.dtype)], 1)
    A = Xa.T @ Xa
    A += RIDGE * A.diagonal().mean() * torch.eye(len(A), device=DEV, dtype=A.dtype)
    W = torch.linalg.solve(A, Xa.T @ Y).float()
    return W[:-1].contiguous(), W[-1].contiguous()


def fold(m, i, k):
    M, c = fit_span(m, i, k)
    m.transformer.h[i] = Affine(M, c)
    for j in range(i + 1, i + k):
        m.transformer.h[j] = Skip()
    return m


def macs_per_token(k):
    """Matmul MACs per token. Attention QK^T and AV use the mean context (SEQ+1)/2."""
    dense_block = 4 * D * D + 2 * D * FF + 2 * D * (SEQ + 1) / 2
    blocks = (12 - k) * dense_block + (D * D if k else 0)
    return blocks, blocks + D * V          # without / with LM head


@torch.no_grad()
def fwd_time(m, n=30):
    x = test_ch[:BS]
    for _ in range(5):
        m(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        m(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3          # ms per batch of 8x256


def recover(student, teacher, log):
    cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                     target_modules=["c_attn", "c_proj", "c_fc"], fan_in_fan_out=True)
    student = get_peft_model(student, cfg)
    aff = []
    for mod in student.modules():
        if isinstance(mod, Affine):
            mod.M.requires_grad_(True); mod.c.requires_grad_(True); aff += [mod.M, mod.c]
    lora = [q for n, q in student.named_parameters() if q.requires_grad and "lora_" in n]
    groups = [{"params": lora, "lr": LR_LORA}]
    if aff:
        groups.append({"params": aff, "lr": LR_AFFINE})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    n_lora = sum(q.numel() for q in lora); n_aff = sum(q.numel() for q in aff)
    print(f"  trainable: LoRA {n_lora/1e6:.2f}M + folded {n_aff/1e6:.2f}M", flush=True)

    steps_ep = args.max_steps or len(train_ch) // BS
    total, warm = steps_ep * args.epochs, 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step = 0
    for ep in range(args.epochs):
        student.train()
        perm = torch.randperm(len(train_ch))
        t0, run, nrun = time.time(), 0.0, 0
        for b in range(steps_ep):
            x = train_ch[perm[b * BS:(b + 1) * BS]].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    tl = teacher(x).logits
                out = student(x, labels=x)
            s = out.logits.float().view(-1, V)
            kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(tl.float().view(-1, V), -1),
                          log_target=True, reduction="batchmean")
            loss = 0.5 * out.loss.float() + 0.5 * kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(lora + aff, 1.0)
            opt.step(); sched.step(); step += 1
            run += loss.item(); nrun += 1
            if step % 200 == 0:
                print(f"    ep{ep} step {step}/{total} loss {run/nrun:.4f} "
                      f"({(time.time()-t0)/(b+1)*1e3:.0f} ms/step)", flush=True)
                run, nrun = 0.0, 0
        v = ppl(student, val_ch)
        log.append({"epoch": ep, "val_ppl": v})
        print(f"  epoch {ep}: val PPL {v:.2f}", flush=True)
    return student.merge_and_unload()


def main():
    teacher = load().to(torch.bfloat16)
    for q in teacher.parameters():
        q.requires_grad_(False)
    base = load()
    res = {"baseline": {"test_ppl": ppl(base, test_ch), "val_ppl": ppl(base, val_ch),
                        "ms_per_batch": fwd_time(base),
                        "macs_blocks": macs_per_token(0)[0], "macs_total": macs_per_token(0)[1]},
           "configs": {}}
    print(f"baseline: test {res['baseline']['test_ppl']:.2f} | val {res['baseline']['val_ppl']:.2f} "
          f"| {res['baseline']['ms_per_batch']:.1f} ms/batch", flush=True)
    del base

    for k in args.ks:
        print(f"\n=== k = {k} {'(control: no fold)' if k == 0 else 'blocks folded'} ===", flush=True)
        r = {"k": k}
        if k:
            spans = {}
            for i in range(12 - k + 1):
                spans[i] = ppl(fold(load(), i, k), val_ch)
                print(f"  span L{i}-L{i+k-1}: val PPL {spans[i]:.2f}", flush=True)
            best = min(spans, key=spans.get)
            r["span_val_ppl"], r["start"] = spans, best
            student = fold(load(), best, k)
            r["folded_test_ppl"] = ppl(student, test_ch)
            print(f"  chosen L{best}-L{best+k-1} | test PPL after fold, before recovery: "
                  f"{r['folded_test_ppl']:.2f}", flush=True)
        else:
            student = load()
        log = []
        t0 = time.time()
        student = recover(student, teacher, log)
        r["train_min"] = (time.time() - t0) / 60
        r["log"] = log
        r["recovered_test_ppl"] = ppl(student, test_ch)
        r["ms_per_batch"] = fwd_time(student)
        r["macs_blocks"], r["macs_total"] = macs_per_token(k)
        b = res["baseline"]
        print(f"  RESULT k={k}: test PPL {r['recovered_test_ppl']:.2f} (baseline {b['test_ppl']:.2f}) | "
              f"block MACs {b['macs_blocks']/r['macs_blocks']:.2f}x less | "
              f"total MACs {b['macs_total']/r['macs_total']:.2f}x less | "
              f"measured {b['ms_per_batch']/r['ms_per_batch']:.2f}x faster", flush=True)
        res["configs"][k] = r
        json.dump(res, open(args.out, "w"), indent=1)
        del student; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
