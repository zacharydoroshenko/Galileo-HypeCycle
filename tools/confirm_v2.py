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
    "occam0.5": {"occam_weight": 0.5},
    "occam1": {"occam_weight": 1.0},
    "occam1.5": {"occam_weight": 1.5},
    "occam2": {"occam_weight": 2.0},
    "occam1+jit": {"occam_weight": 1.0, "jitter_fit": True},
}
SETS = {"std1000": "confirm_std1000.npz", "sparse600": "confirm_sparse600.npz"}


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
    ref = {}
    for name in a.configs:
        rv_v2.CONFIG.clear()
        rv_v2.CONFIG.update(base, **CONFIGS[name])
        t0 = time.time()
        rec = {"config": name}
        parts = []
        for k, d in data.items():
            ok = success(rv_v2.predict(d["X"], n_workers=a.workers), d["Y"])[0]
            if name == "plain":
                ref[k] = ok
            g = int((ok & ~ref[k]).sum()) if k in ref else 0
            l = int((~ok & ref[k]).sum()) if k in ref else 0
            rec[k] = float(ok.mean())
            rec[k + "_gain_loss"] = [g, l]
            parts.append(f"{k} {ok.mean():.4f} (+{g}/-{l}, p={sign_p(g, l):.3f})")
        rec["sec"] = time.time() - t0
        print(f"{name:11s} " + "   ".join(parts) + f"   [{rec['sec']:.0f}s]", flush=True)
        with open(ROOT / "build/confirm.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
