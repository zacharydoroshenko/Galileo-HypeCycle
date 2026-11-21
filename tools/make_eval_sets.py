"""Train-only proxies for the validation groups (no new stars, no change to the orbit distribution).

  data/exo/train_sparse300.npz : 300 train stars subsampled to 15 observations (first + last kept,
                                 mirroring the simulator's "one obs in first and last season" rule)
  data/exo/train_hiecc300.npz  : the 300 most eccentric train stars (train has no e >= 0.6; proxy only)
Disjoint from train_sub300 (seed 0). Disclose in the report.

    .venv/bin/python tools/make_eval_sets.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    X = np.load(ROOT / "data/exo/train/train_inputs.npy").astype(float)
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = set(np.random.default_rng(0).choice(len(X), 300, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    rng = np.random.default_rng(1)

    idx = rng.choice(pool, 300, replace=False)
    Xs = np.zeros_like(X[idx])
    for k, i in enumerate(idx):
        valid = np.where(X[i, 3] > 0.5)[0]
        keep = np.concatenate([[valid[0], valid[-1]], rng.choice(valid[1:-1], 13, replace=False)])
        keep = np.sort(keep)
        Xs[k, :, :15] = X[i][:, keep]
        Xs[k, 0, :15] -= Xs[k, 0, 0]  # t = days since first observation
    np.savez(ROOT / "data/exo/train_sparse300.npz", X=Xs, Y=Y[idx])

    hi = pool[np.argsort(Y[pool, 2])[::-1][:300]]
    np.savez(ROOT / "data/exo/train_hiecc300.npz", X=X[hi], Y=Y[hi])
    print(f"sparse: n_obs={int(Xs[0, 3].sum())}  | hiecc: e range {Y[hi, 2].min():.3f}-{Y[hi, 2].max():.3f}")
