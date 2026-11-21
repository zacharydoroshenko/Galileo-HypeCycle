"""Residual-noise SNR vs assumed-noise SNR for the K-debias (augmented TRAIN calibration set).

If the residual estimate captures each star's real noise, the fitted c should agree across std / sparse15 / noisy.
    .venv/bin/python tools/k_calib_resid.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from k_calib import ok  # noqa: E402
from k_calib_aug import NAMES, sign_p  # noqa: E402


def corr(pred, nobs, noise, c, floor=0.5):
    k = 10 ** pred[:, 1]
    snr2 = k**2 * nobs / np.maximum(noise**2, 1e-12)
    return np.log10(k * np.maximum(1.0 / np.sqrt(1.0 + c / np.maximum(snr2, 1e-9)), floor))


if __name__ == "__main__":
    d = np.load(ROOT / "data/exo/calib_aug.npz")
    Y, pred, nobs, sig, kind, rn = d["Y"], d["pred"], d["nobs"].astype(float), d["sig_eff"], d["kind"], d["resid_noise"]
    A = np.random.default_rng(0).random(len(Y)) < 0.5
    B = ~A
    cs = np.concatenate([[0.0], np.geomspace(0.25, 30, 60)])
    fit = lambda mask, noise: cs[int(np.argmax([ok(pred[mask, 0], corr(pred[mask], nobs[mask], noise[mask], c), Y[mask]).sum() for c in cs]))]
    c_res_mix = fit(A, rn)
    print(f"residual-noise SNR: c fitted on half A of the mix = {c_res_mix:.2f}")
    for k, name in enumerate(NAMES):
        mk = kind == k
        c_res_k = fit(A & mk, rn)
        mb = B & mk
        base = ok(pred[mb, 0], pred[mb, 1], Y[mb])
        cur = ok(pred[mb, 0], corr(pred[mb], nobs[mb], sig[mb], 6.95), Y[mb])        # current: assumed noise, c=6.95
        new = ok(pred[mb, 0], corr(pred[mb], nobs[mb], rn[mb], c_res_mix), Y[mb])     # residual noise, c_mix
        g, l = int((new & ~cur).sum()), int((~new & cur).sum())
        print(f"  {name:8s} c_res(alone)={c_res_k:5.2f} | half B: none {base.mean():.4f}  current {cur.mean():.4f}"
              f"  residual {new.mean():.4f}  (residual vs current +{g}/-{l}, p={sign_p(g, l):.3f})")
