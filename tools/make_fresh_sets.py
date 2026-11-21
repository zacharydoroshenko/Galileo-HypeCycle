"""FRESH held-out TRAIN sets for late experiments (never used for any earlier selection or calibration).

Excludes every star used so far (train_sub300, sparse300, hiecc300, confirm_*, confirm_sparse_v2_1000, calib 6,000).
  data/exo/fresh_std2000.npz     2,000 standard stars
  data/exo/fresh_sparse2000.npz  2,000 stars subsampled to 15 obs (simulator rule)
  data/exo/fresh_hie600.npz      the 600 most eccentric remaining stars (e up to 0.6; proxy only)
    python tools/make_fresh_sets.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_calib_aug import sparse15  # noqa: E402
from make_calib_preds import used_indices  # noqa: E402

if __name__ == "__main__":
    X = np.load(ROOT / "data/exo/train/train_inputs.npy").astype(float)
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = used_indices(X, Y)
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(4).choice(pool, 6000, replace=False).tolist())  # calib stars
    pool = np.array([i for i in range(len(X)) if i not in used])
    hie = pool[np.argsort(Y[pool, 2])[::-1][:600]]
    pool = np.setdiff1d(pool, hie)
    rng = np.random.default_rng(7)
    pick = rng.choice(pool, 4000, replace=False)
    a, b = pick[:2000], pick[2000:]
    np.savez(ROOT / "data/exo/fresh_std2000.npz", X=X[a], Y=Y[a])
    np.savez(ROOT / "data/exo/fresh_sparse2000.npz", X=np.stack([sparse15(X[i], rng) for i in b]), Y=Y[b])
    np.savez(ROOT / "data/exo/fresh_hie600.npz", X=X[hie], Y=Y[hie])
    print(f"unused pool {len(pool) + 600}; wrote fresh_std2000, fresh_sparse2000, fresh_hie600 (e {Y[hie, 2].min():.3f}-{Y[hie, 2].max():.3f})")
