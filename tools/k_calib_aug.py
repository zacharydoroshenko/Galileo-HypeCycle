"""Fit the K-debias constant c on the augmented TRAIN calibration set; held-out half B per variant.

    .venv/bin/python tools/k_calib_aug.py
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from k_calib import corrected, ok  # noqa: E402

NAMES = ["std", "sparse15", "noisy"]


def sign_p(g, l):
    n = g + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(g, l) + 1)) / 2**n)


if __name__ == "__main__":
    d = np.load(ROOT / "data/exo/calib_aug.npz")
    Y, pred, nobs, sig, kind = d["Y"], d["pred"], d["nobs"].astype(float), d["sig_eff"], d["kind"]
    A = np.random.default_rng(0).random(len(Y)) < 0.5
    B = ~A
    cs = np.concatenate([[0.0], np.geomspace(0.25, 30, 50)])

    def n_ok(mask, c):
        return int(ok(pred[mask, 0], corrected(pred[mask], nobs[mask], sig[mask], c), Y[mask]).sum())

    c_all = cs[int(np.argmax([n_ok(A, c) for c in cs]))]
    print(f"c fitted on half A of the full augmented mix: {c_all:.2f}   (standard-only fit earlier: 6.95)")
    for k, name in enumerate(NAMES):
        mk = kind == k
        c_k = cs[int(np.argmax([n_ok(A & mk, c) for c in cs]))]
        for label, c in [("c_mix", c_all), ("c=6.95", 6.95)]:
            mb = B & mk
            base = ok(pred[mb, 0], pred[mb, 1], Y[mb])
            new = ok(pred[mb, 0], corrected(pred[mb], nobs[mb], sig[mb], c), Y[mb])
            g, l = int((new & ~base).sum()), int((~new & base).sum())
            print(f"  {name:8s} {label:7s} half B: {base.mean():.4f} -> {new.mean():.4f}  +{g}/-{l}  p={sign_p(g, l):.3f}")
        print(f"  {name:8s} c fitted on this variant alone (half A): {c_k:.2f}")
