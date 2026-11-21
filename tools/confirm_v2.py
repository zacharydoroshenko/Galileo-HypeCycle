"""Paired confirmation of rv_v2 configs vs plain on the larger disjoint train sets.

    .venv/bin/python tools/confirm_v2.py [--workers 8]
For each config: success per set, and paired gained/lost stars vs plain + sign-test p-value.
Appends to build/confirm.jsonl.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools")]

CONFIGS = {
    "plain": {},
    "occam0.5": {"occam_weight": 0.5},                      # = upload #1
    "occpk0.5": {"occam_weight": 0.5, "occam_mode": "pk"},
    "occpk1": {"occam_weight": 1.0, "occam_mode": "pk"},
    "ext": {"e_grid_ext": True},
    "starts5": {"n_starts": 5},
    "ext+starts5": {"e_grid_ext": True, "n_starts": 5},
    "win2.5": {"freq_window": 2.5},
    "nfev600": {"max_nfev": 600},
    "jitdof": {"jitter_fit": True, "jitter_dof": True},
    "occ0.5+sp1": {"occam_weight": 0.5, "occam_weight_sparse": 1.0},
    "pk0.5+ext+s5": {"occam_weight": 0.5, "occam_mode": "pk", "e_grid_ext": True, "n_starts": 5},
    "pk0.5+ext+s5+jd": {"occam_weight": 0.5, "occam_mode": "pk", "e_grid_ext": True, "n_starts": 5,
                         "jitter_fit": True, "jitter_dof": True},
}
SETS = {"std1000": "confirm_std1000.npz", "sparse600": "confirm_sparse600.npz", "hiecc300": "train_hiecc300.npz"}
REFS = ("plain", "occam0.5")  # paired comparisons against the baseline rule and against upload #1


def sign_p(gain, loss):
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    import rv_v2
    from evaluate import success
    base = dict(rv_v2.CONFIG)
    data = {k: np.load(ROOT / "data/exo" / v) for k, v in SETS.items()}
    ref = {r: {} for r in REFS}
    order = [c for c in REFS if c in a.configs] + [c for c in a.configs if c not in REFS]
    for name in order:
        rv_v2.CONFIG.clear()
        rv_v2.CONFIG.update(base, **CONFIGS[name])
        t0 = time.time()
        rec = {"config": name, "overrides": CONFIGS[name]}
        parts = []
        for k, d in data.items():
            ok = success(rv_v2.predict(d["X"], n_workers=a.workers), d["Y"])[0]
            if name in REFS:
                ref[name][k] = ok
            rec[k] = float(ok.mean())
            cmp = []
            for r in REFS:
                if k in ref[r] and r != name:
                    g, l = int((ok & ~ref[r][k]).sum()), int((~ok & ref[r][k]).sum())
                    rec[f"{k}_vs_{r}"] = [g, l, sign_p(g, l)]
                    cmp.append(f"{'vB' if r == 'plain' else 'v1'} +{g}/-{l} p={sign_p(g, l):.2f}")
            parts.append(f"{k} {ok.mean():.4f} [" + "; ".join(cmp) + "]")
        rec["sec"] = time.time() - t0
        print(f"{name:17s} " + "  ".join(parts) + f"  ({rec['sec']:.0f}s)", flush=True)
        with open(ROOT / "build/confirm.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
