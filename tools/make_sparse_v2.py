"""Simulator-faithful sparse proxy from TRAIN stars (fixes the span bias of train_sparse300).

The generator (package simulator/generate_rv_data.py, draw_times) picks ONE RANDOM night in the first
season, one in the last, and N_obs-2 uniformly from the rest. Keeping a 30-60-point star's earliest
and latest observations (old proxy) over-stretches the span. Here: seasons = clusters split by gaps
> 60 d; pick one random obs in the first and last cluster, 13 uniformly from the remaining obs.
Disjoint from every other eval set (uses stars not in train_sub300 / sparse300 / hiecc300 / confirm_*).

    python tools/make_sparse_v2.py      -> data/exo/confirm_sparse_v2_1000.npz
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    X = np.load(ROOT / "data/exo/train/train_inputs.npy").astype(float)
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = set(np.random.default_rng(0).choice(len(X), 300, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(1).choice(pool, 300, replace=False).tolist())
    used |= set(pool[np.argsort(Y[pool, 2])[::-1][:300]].tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(2).choice(pool, 1600, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    rng = np.random.default_rng(3)
    idx = rng.choice(pool, 1000, replace=False)
    Xs = np.zeros_like(X[idx])
    spans_new, spans_old = [], []
    for k, i in enumerate(idx):
        v = np.where(X[i, 3] > 0.5)[0]
        t = X[i, 0, v]
        season = np.concatenate([[0], np.cumsum(np.diff(t) > 60.0)])
        first = rng.choice(v[season == 0])
        last = rng.choice(v[season == season.max()])
        rest = np.setdiff1d(v, [first, last])
        keep = np.sort(np.concatenate([[first, last], rng.choice(rest, 13, replace=False)]))
        Xs[k, :, :15] = X[i][:, keep]
        Xs[k, 0, :15] -= Xs[k, 0, 0]
        spans_new.append(Xs[k, 0, 14])
        spans_old.append(t[-1] - t[0])
    np.savez(ROOT / "data/exo/confirm_sparse_v2_1000.npz", X=Xs, Y=Y[idx])
    print(f"median span: faithful proxy {np.median(spans_new):.0f} d vs old first/last-kept {np.median(spans_old):.0f} d")
