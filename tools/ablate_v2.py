"""Ablation of rv_v2 switches on train-only eval sets (standard / sparse proxy / high-e proxy).

    .venv/bin/python tools/ablate_v2.py [--n 300] [--workers 8] [--configs plain jitter ...]
Prints success per set + mean, and CPU time. Results -> build/ablation.jsonl (for the report).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools")]

CONFIGS = {
    "plain": {},
    "jitter": {"jitter_fit": True},
    "occam": {"occam_weight": 1.0},
    "yearly": {"yearly_aliases": True},
    "bounds": {"bound_penalty": 5.0},
    "peaks5": {"n_peaks": 5},
    "all": {"jitter_fit": True, "occam_weight": 1.0, "yearly_aliases": True, "bound_penalty": 5.0},
}
SETS = {"standard": "train_sub300.npz", "sparse15": "train_sparse300.npz", "hi_e": "train_hiecc300.npz"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    a = ap.parse_args()
    import rv_v2
    from evaluate import success
    base = dict(rv_v2.CONFIG)
    data = {k: np.load(ROOT / "data/exo" / v) for k, v in SETS.items()}
    (ROOT / "build").mkdir(exist_ok=True)
    print(f"{'config':10s} " + " ".join(f"{k:>9s}" for k in SETS) + "      mean   s/300")
    for name in a.configs:
        rv_v2.CONFIG.clear()
        rv_v2.CONFIG.update(base, **CONFIGS[name])
        row, t0 = {}, time.time()
        for k, d in data.items():
            p = rv_v2.predict(d["X"][: a.n], n_workers=a.workers)
            row[k] = float(success(p, d["Y"][: a.n])[0].mean())
        dt = (time.time() - t0) / len(data)
        mean = float(np.mean(list(row.values())))
        print(f"{name:10s} " + " ".join(f"{row[k]:9.3f}" for k in SETS) + f"   {mean:7.3f}  {dt:6.0f}", flush=True)
        with open(ROOT / "build/ablation.jsonl", "a") as f:
            f.write(json.dumps({"config": name, "overrides": CONFIGS[name], **row, "mean": mean, "sec": dt}) + "\n")
