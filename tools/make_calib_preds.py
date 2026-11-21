"""Predictions of a config on a fresh, disjoint TRAIN calibration set (for K-bias analysis/calibration).

    .venv/bin/python tools/make_calib_preds.py --n 6000 --workers 8 [--occam 0.5]
Writes data/exo/calib_preds.npz with X, Y, idx, pred and per-star features.
Disjoint from train_sub300, train_sparse300, train_hiecc300, confirm_std1000/sparse600, confirm_sparse_v2_1000.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))


def used_indices(X, Y):
    used = set(np.random.default_rng(0).choice(len(X), 300, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(1).choice(pool, 300, replace=False).tolist())
    used |= set(pool[np.argsort(Y[pool, 2])[::-1][:300]].tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(2).choice(pool, 1600, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(3).choice(pool, 1000, replace=False).tolist())
    return used


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--occam", type=float, default=0.5)
    a = ap.parse_args()
    import rv_v2
    X = np.load(ROOT / "data/exo/train/train_inputs.npy")
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = used_indices(X, Y)
    pool = np.array([i for i in range(len(X)) if i not in used])
    idx = np.sort(np.random.default_rng(4).choice(pool, a.n, replace=False))
    rv_v2.CONFIG["occam_weight"] = a.occam
    t0 = time.time()
    pred = rv_v2.predict(X[idx], n_workers=a.workers)
    m = X[idx, 3] > 0.5
    nobs = m.sum(1)
    sig = np.array([np.sqrt(np.mean(X[i, 2, mm] ** 2) + 4.0) for i, mm in zip(idx, m)])
    np.savez(ROOT / "data/exo/calib_preds.npz", X=X[idx], Y=Y[idx], idx=idx, pred=pred, nobs=nobs, sig_eff=sig)
    print(f"wrote calib_preds.npz: {len(idx)} stars in {time.time() - t0:.0f}s")
