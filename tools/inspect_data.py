"""First thing to run on the real data: figure out the channel layout and the label distributions,
so CONFIG (channels, time units, period range) and tools/synth.py can be matched to reality.

    python tools/inspect_data.py ~/work/exoplanet-orbits/data/train
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train import load_train  # noqa: E402


def q(a):
    a = a[np.isfinite(a)]
    return np.round(np.percentile(a, [0, 5, 25, 50, 75, 95, 100]), 4).tolist() if a.size else "empty"


def main(path):
    p = Path(path)
    print("files:", sorted(x.name for x in p.iterdir()))
    X, Y = load_train(p)
    print("X", X.shape, X.dtype, " Y", Y.shape, Y.dtype)
    for c in range(X.shape[1]):
        v = X[:, c, :]
        uniq = np.unique(v[np.isfinite(v)])
        print(f"channel {c}: pct {q(v.ravel())}  n_unique={uniq.size}  nan={np.isnan(v).mean():.3f}"
              f"  zeros={(v == 0).mean():.3f}  monotone_rows={np.mean(np.all(np.diff(v, axis=1) >= 0, 1)):.2f}")
    for j, name in enumerate(["log10 P", "log10 K", "e"][: Y.shape[1]]):
        print(f"{name:8s}: {q(Y[:, j])}")
    print("first star:\n", np.round(X[0][:, :12], 3))


if __name__ == "__main__":
    main(sys.argv[1])
