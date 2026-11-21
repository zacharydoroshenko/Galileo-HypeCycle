"""Larger confirmation sets from TRAIN stars disjoint from all *_300 sets (seed 2).
  data/exo/confirm_std1000.npz, data/exo/confirm_sparse600.npz (15 obs, first+last kept)
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    X = np.load(ROOT / "data/exo/train/train_inputs.npy").astype(float)
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = set(np.random.default_rng(0).choice(len(X), 300, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(1).choice(pool, 300, replace=False).tolist())   # sparse300
    used |= set(pool[np.argsort(Y[pool, 2])[::-1][:300]].tolist())                     # hiecc300
    pool = np.array([i for i in range(len(X)) if i not in used])
    rng = np.random.default_rng(2)
    pick = rng.choice(pool, 1600, replace=False)
    a, b = pick[:1000], pick[1000:]
    np.savez(ROOT / "data/exo/confirm_std1000.npz", X=X[a], Y=Y[a])
    Xs = np.zeros_like(X[b])
    for k, i in enumerate(b):
        v = np.where(X[i, 3] > 0.5)[0]
        keep = np.sort(np.concatenate([[v[0], v[-1]], rng.choice(v[1:-1], 13, replace=False)]))
        Xs[k, :, :15] = X[i][:, keep]
        Xs[k, 0, :15] -= Xs[k, 0, 0]
    np.savez(ROOT / "data/exo/confirm_sparse600.npz", X=Xs, Y=Y[b])
    print("wrote confirm_std1000.npz, confirm_sparse600.npz")
