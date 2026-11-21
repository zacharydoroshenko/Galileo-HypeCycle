"""Augmented TRAIN calibration set for hyperparameter fitting (organizer hint: augment train data).

From the 6,000 calibration stars of make_calib_preds.py (disjoint from all eval sets), builds 3 x 2,000:
  std     : unchanged
  sparse15: 15 obs, simulator rule (1 random obs in first season, 1 in last, 13 uniform)
  noisy   : + N(0, 2 m/s) extra velocity noise, reported sigma unchanged (acts as extra unreported jitter)
No new stars are simulated. Predictions with occam_weight=0.5 and NO K-debias (c is what we fit).

    .venv/bin/python tools/make_calib_aug.py --workers 8   -> data/exo/calib_aug.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools")]
from make_calib_preds import used_indices  # noqa: E402


def sparse15(x, rng):
    v = np.where(x[3] > 0.5)[0]
    t = x[0, v]
    season = np.concatenate([[0], np.cumsum(np.diff(t) > 60.0)])
    first, last = rng.choice(v[season == 0]), rng.choice(v[season == season.max()])
    keep = np.sort(np.concatenate([[first, last], rng.choice(np.setdiff1d(v, [first, last]), 13, replace=False)]))
    out = np.zeros_like(x)
    out[:, :15] = x[:, keep]
    out[0, :15] -= out[0, 0]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import rv_v2
    X = np.load(ROOT / "data/exo/train/train_inputs.npy").astype(float)
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = used_indices(X, Y)
    pool = np.array([i for i in range(len(X)) if i not in used])
    idx = np.sort(np.random.default_rng(4).choice(pool, 6000, replace=False))  # same stars as calib_preds
    rng = np.random.default_rng(5)
    parts, kinds = [], []
    for j, i in enumerate(idx):
        x = X[i].copy()
        kind = j % 3
        if kind == 1:
            x = sparse15(x, rng)
        elif kind == 2:
            m = x[3] > 0.5
            x[1, m] += rng.normal(0.0, 2.0, m.sum())
        parts.append(x)
        kinds.append(kind)
    XA, kinds = np.stack(parts), np.array(kinds)
    rv_v2.CONFIG["occam_weight"] = 0.5
    rv_v2.CONFIG["return_diag"] = True
    t0 = time.time()
    full = rv_v2.predict(XA, n_workers=a.workers)
    pred, resid_noise = full[:, :3], full[:, 3]
    m = XA[:, 3] > 0.5
    nobs = m.sum(1)
    sig = np.array([np.sqrt(np.mean(XA[k, 2, mm] ** 2) + 4.0) for k, mm in enumerate(m)])
    np.savez(ROOT / "data/exo/calib_aug.npz", Y=Y[idx], pred=pred, nobs=nobs, sig_eff=sig, kind=kinds, idx=idx, resid_noise=resid_noise)
    print(f"wrote calib_aug.npz ({len(idx)} stars, kinds std/sparse15/noisy = {np.bincount(kinds)}) in {time.time() - t0:.0f}s")
