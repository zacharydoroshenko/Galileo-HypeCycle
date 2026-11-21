"""Second independent fresh std set (replication), disjoint from everything incl. fresh_* (seed 8).
    python tools/make_fresh_b.py  -> data/exo/freshB_std2500.npz
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_calib_preds import used_indices  # noqa: E402

if __name__ == "__main__":
    X = np.load(ROOT / "data/exo/train/train_inputs.npy").astype(float)
    Y = np.load(ROOT / "data/exo/train/train_targets.npy").astype(float)
    used = used_indices(X, Y)
    pool = np.array([i for i in range(len(X)) if i not in used])
    used |= set(np.random.default_rng(4).choice(pool, 6000, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    hie = pool[np.argsort(Y[pool, 2])[::-1][:600]]
    pool = np.setdiff1d(pool, hie)
    used |= set(hie.tolist()) | set(np.random.default_rng(7).choice(pool, 4000, replace=False).tolist())
    pool = np.array([i for i in range(len(X)) if i not in used])
    b = np.random.default_rng(8).choice(pool, 2500, replace=False)
    np.savez(ROOT / "data/exo/freshB_std2500.npz", X=X[b], Y=Y[b])
    print(f"remaining pool {len(pool)}; wrote freshB_std2500")
