"""
Exp 09 probe — How linear is GPT-2 small?

Question
--------
If we replace the nonlinear parts of GPT-2 with calibrated linear maps,
consecutive matrices can be multiplied together into one matrix.
Which nonlinear parts can we remove, and what does each removal cost in PPL?

Operations (each one turns a nonlinear part into an affine map)
----------
  gelu  : GELU -> per-channel affine a*z+b (least squares on calibration data).
          Then the MLP collapses: W_fc diag(a) W_proj = one 768x768 matrix (8x fewer MACs).
  ln1/ln2: LayerNorm -> fixed scale (per-token std replaced by one calibrated constant).
          Then LN is affine and folds into the next matrix.
  block : the whole transformer block -> one 768x768 affine map on the residual
          stream (ridge least squares). This also removes attention (token mixing).

Protocol
--------
- Calibration: WikiText-2 train, first CALIB chunks of 256 tokens.
- Eval: WikiText-2 test, first EVAL chunks of 256 tokens (subset of the Exp 06-08 protocol).
- Part 1: one operation at a time, one layer at a time -> delta PPL.
- Part 2: cumulative, cheapest first, refit on the current (already modified) model.
- Part 3: fully linear model (all blocks + ln_f linear) = one matrix.
"""

import json
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ, CALIB, EVAL, BS = 256, 32, 80, 8
RIDGE = 1e-3
PPL_BUDGET = 1.10     # Part 2 stops adding operations above baseline * budget

tok = GPT2TokenizerFast.from_pretrained("gpt2")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


def chunks(split):
    ids = []
    for t in raw[split]["text"]:
        if t.strip():
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
    ids = torch.tensor(ids)
    return ids[: len(ids) // SEQ * SEQ].view(-1, SEQ)


calib = chunks("train")[:CALIB].to(DEV)
test = chunks("test")[:EVAL].to(DEV)


def load():
    return GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()


@torch.no_grad()
def ppl(m):
    tot = 0.0
    for i in range(0, len(test), BS):
        b = test[i:i + BS]
        tot += m(b, labels=b).loss.item()
    return float(torch.exp(torch.tensor(tot / (len(test) / BS))))


@torch.no_grad()
def capture(m, module, pre=True):
    """Return (inputs, outputs) of `module` on the calibration set, flattened to rows."""
    xs, ys = [], []
    def hook(mod, i, o):                                     # must return None
        xs.append(i[0].reshape(-1, i[0].shape[-1]))
        ys.append(o.reshape(-1, o.shape[-1]))
    h = module.register_forward_hook(hook)
    for i in range(0, len(calib), BS):
        m(calib[i:i + BS])
    h.remove()
    return torch.cat(xs), torch.cat(ys)


def ridge(X, Y):
    """Affine least squares Y ~ X M + c."""
    Xa = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1).double()
    A = Xa.T @ Xa
    A += RIDGE * A.diagonal().mean() * torch.eye(len(A), device=X.device, dtype=A.dtype)
    W = torch.linalg.solve(A, Xa.T @ Y.double()).float()
    return W[:-1], W[-1]


class Affine(nn.Module):
    def __init__(self, M, c):
        super().__init__()
        self.M, self.c = nn.Parameter(M, requires_grad=False), nn.Parameter(c, requires_grad=False)

    def forward(self, x, *a, **k):
        return x @ self.M + self.c


class FixedLN(nn.Module):
    """LayerNorm with the per-token std replaced by one constant -> affine map."""
    def __init__(self, ln, s):
        super().__init__()
        self.w, self.b, self.s = ln.weight, ln.bias, s

    def forward(self, x):
        return self.w * (x - x.mean(-1, keepdim=True)) / self.s + self.b


# ── the three operations ─────────────────────────────────────────────────────
@torch.no_grad()
def lin_gelu(m, l):
    mlp = m.transformer.h[l].mlp
    Z, A = capture(m, mlp.act)                               # pre/post GELU, (N, 3072)
    zc = Z - Z.mean(0)
    a = (zc * (A - A.mean(0))).sum(0) / (zc * zc).sum(0)     # per-channel slope
    b = A.mean(0) - a * Z.mean(0)
    Wfc, bfc = mlp.c_fc.weight, mlp.c_fc.bias                # Conv1D: y = x W + b
    Wp, bp = mlp.c_proj.weight, mlp.c_proj.bias
    M = (Wfc * a) @ Wp                                       # 768x768
    c = ((bfc * a) + b) @ Wp + bp
    m.transformer.h[l].mlp = Affine(M, c)


@torch.no_grad()
def lin_ln(m, l, which):
    ln = getattr(m.transformer.h[l], which) if l >= 0 else m.transformer.ln_f
    X, _ = capture(m, ln)
    std = (X.var(-1, unbiased=False) + ln.eps).sqrt()
    s = 1.0 / (1.0 / std).mean()                             # best single scale for 1/std
    fixed = FixedLN(ln, s)
    if l >= 0:
        setattr(m.transformer.h[l], which, fixed)
    else:
        m.transformer.ln_f = fixed


@torch.no_grad()
def lin_block(m, l):
    X, Y = capture(m, m.transformer.h[l])
    M, c = ridge(X, Y)
    m.transformer.h[l] = Affine(M, c)


OPS = {"gelu": lin_gelu,
       "ln1": lambda m, l: lin_ln(m, l, "ln_1"),
       "ln2": lambda m, l: lin_ln(m, l, "ln_2"),
       "block": lin_block}

# MACs per token (GPT-2 small, attention score/value ops excluded: they depend on T)
MACS = {"block_dense": 4 * 768 * 768 + 2 * 768 * 3072,      # 7.08M
        "mlp_dense": 2 * 768 * 3072, "mlp_lin": 768 * 768}


def main():
    t0 = time.time()
    base = ppl(load())
    print(f"baseline PPL (first {EVAL} test chunks): {base:.2f}")
    out = {"baseline": base, "single": {}, "cumulative": [], "fully_linear": None}

    # Part 1: one at a time
    print("\nPart 1 — one operation, one layer at a time (PPL)")
    print("layer " + "".join(f"{o:>9s}" for o in OPS))
    for l in range(12):
        row = {}
        for o, f in OPS.items():
            m = load(); f(m, l); row[o] = ppl(m)
        out["single"][l] = row
        print(f"{l:5d} " + "".join(f"{row[o]:9.2f}" for o in OPS))
    m = load(); lin_ln(m, -1, None); out["single"]["ln_f"] = ppl(m)
    print(f"ln_f fixed: {out['single']['ln_f']:.2f}")

    # Part 2: cumulative greedy (cheapest first), refit after each step
    print(f"\nPart 2 — cumulative, cheapest first, stop at {PPL_BUDGET:.2f}x baseline")
    cands = sorted(((out["single"][l][o], o, l) for l in range(12)
                    for o in ("gelu", "ln1", "ln2")), key=lambda t: t[0])
    m = load(); applied = []
    for _, o, l in cands:
        trial = load()
        for o2, l2 in applied + [(o, l)]:
            OPS[o2](trial, l2)                                # sequential refit
        p = ppl(trial)
        keep = p <= base * PPL_BUDGET
        print(f"  +{o:5s} L{l:<2d} -> {p:8.2f}  {'keep' if keep else 'skip'}")
        if keep:
            applied.append((o, l)); m = trial
        out["cumulative"].append({"op": o, "layer": l, "ppl": p, "kept": keep})
    n_mlp = sum(o == "gelu" for o, _ in applied)
    dense = 12 * MACS["block_dense"]
    saved = n_mlp * (MACS["mlp_dense"] - MACS["mlp_lin"])
    print(f"  kept {len(applied)} ops ({n_mlp} MLPs collapsed) | final PPL {ppl(m):.2f} | "
          f"matmul MACs/token {dense/1e6:.1f}M -> {(dense-saved)/1e6:.1f}M "
          f"({dense/(dense-saved):.2f}x)")
    out["kept"] = applied

    # Part 3: fully linear model -> one 768x768 matrix between embedding and unembedding
    m = load()
    for l in range(12):
        lin_block(m, l)
    lin_ln(m, -1, None)
    out["fully_linear"] = ppl(m)
    print(f"\nPart 3 — all 12 blocks + ln_f linear (= one matrix): PPL {out['fully_linear']:.2f}")

    json.dump(out, open("probe_results.json", "w"), indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s, saved probe_results.json")


if __name__ == "__main__":
    main()
