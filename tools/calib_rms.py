"""Fit c for K-debias with REALISED SNR (k_debias_snr="rms") on the 6,000 TRAIN calibration stars; half A fit, half B test.
    python tools/calib_rms.py --workers 15
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools")]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=15); a = ap.parse_args()
    import rv_v2
    from k_calib import ok
    from k_calib_aug import sign_p
    d = np.load(ROOT / "data/exo/calib_preds.npz")
    X, Y = d["X"], d["Y"]
    rv_v2.CONFIG.update(occam_weight=0.5, return_diag=True)
    full = rv_v2.predict(X, n_workers=a.workers)
    pred, snr2_rms = full[:, :3], full[:, 4]
    snr2_k = 10 ** (2 * pred[:, 1]) * d["nobs"] / d["sig_eff"] ** 2
    def corr(snr2, c):
        return pred[:, 1] + np.log10(np.maximum(1 / np.sqrt(1 + c / np.maximum(snr2, 1e-9)), 0.5))
    A = np.random.default_rng(0).random(len(Y)) < 0.5; B = ~A
    cs = np.concatenate([[0.0], np.geomspace(0.25, 30, 60)])
    res = {}
    for name, snr2 in [("K", snr2_k), ("rms", snr2_rms)]:
        c = cs[int(np.argmax([ok(pred[A, 0], corr(snr2, c)[A], Y[A]).sum() for c in cs]))]
        res[name] = (c, ok(pred[B, 0], corr(snr2, c)[B], Y[B]))
        print(f"{name:4s} SNR: c fitted on half A = {c:.2f}; half B success {res[name][1].mean():.4f}")
    g, l = int((res["rms"][1] & ~res["K"][1]).sum()), int((~res["rms"][1] & res["K"][1]).sum())
    print(f"half B rms vs K: +{g}/-{l}  p={sign_p(g, l):.3f}   (eccentric stars e>0.3: n={int((Y[B,2]>0.3).sum())})")
    np.save(ROOT / "build/c_rms.npy", np.array([res["rms"][0]]))
