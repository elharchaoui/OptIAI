"""Exp 09 probe 2 — cumulative block linearization and output-aware MLP fit.
Reuses probe.py helpers. Order = Part 1 single-block PPL (cheapest first).
Each step fits on the current, already-modified model (sequential calibration)."""
import json
from probe import load, ppl, capture, ridge, Affine, lin_block

single = json.load(open("probe_results.json"))
base = single["baseline"]

def lin_mlp_ridge(m, l):
    blk = m.transformer.h[l]
    X, Y = capture(m, blk.mlp)          # ln_2 output -> MLP output
    M, c = ridge(X, Y)
    blk.mlp = Affine(M, c)

out = {"baseline": base}
for name, op, key in [("block", lin_block, "block"), ("mlp_ridge", lin_mlp_ridge, None)]:
    if key:
        order = sorted(range(12), key=lambda l: single["single"][str(l)][key])
    else:
        sing = {}
        for l in range(12):
            m = load(); op(m, l); sing[l] = ppl(m)
        print(f"{name} single:", {l: round(p, 2) for l, p in sing.items()})
        order = sorted(range(12), key=lambda l: sing[l])
    m, curve = load(), []
    for l in order:
        op(m, l); p = ppl(m); curve.append((l, p))
        print(f"  {name} cumulative +L{l:<2d} ({len(curve):2d} replaced) -> PPL {p:9.2f}")
    out[name] = curve
json.dump(out, open("probe2_results.json", "w"), indent=1)
