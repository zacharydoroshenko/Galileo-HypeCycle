"""Is ML K biased at low SNR, and does a train-learned correction help? (TRAIN calibration set only)

Model: K_hat^2 ~= K^2 + c * sig_eff^2 / N  =>  K_corr = K_hat / sqrt(1 + c / snr^2), snr = K_hat sqrt(N) / sig_eff
(clipped so K_corr >= K_hat * floor). c fitted on half A by maximising successes, scored on half B (paired).

    .venv/bin/python tools/k_calib.py
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ok(lp, lk, Y):
    return (np.abs(10 ** lp - 10 ** Y[:, 0]) / 10 ** Y[:, 0] < 0.01) & (np.abs(10 ** lk - 10 ** Y[:, 1]) / 10 ** Y[:, 1] < 0.10)


def corrected(pred, nobs, sig, c, floor=0.5):
    k = 10 ** pred[:, 1]
    snr2 = k**2 * nobs / sig**2
    fac = np.maximum(1.0 / np.sqrt(1.0 + c / np.maximum(snr2, 1e-9)), floor)
    return np.log10(k * fac)


if __name__ == "__main__":
    d = np.load(ROOT / "data/exo/calib_preds.npz")
    Y, pred, nobs, sig = d["Y"], d["pred"], d["nobs"].astype(float), d["sig_eff"]
    pok = np.abs(10 ** pred[:, 0] - 10 ** Y[:, 0]) / 10 ** Y[:, 0] < 0.01
    r = pred[:, 1] - Y[:, 1]
    snr = 10 ** pred[:, 1] * np.sqrt(nobs) / sig
    print(f"N={len(Y)}  success {ok(pred[:, 0], pred[:, 1], Y).mean():.4f}  P-correct {pok.mean():.4f}")
    print("log10 K residual (pred - true) on P-correct stars, by SNR bin:")
    bins = [0, 5, 8, 12, 20, 40, 1e9]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = pok & (snr >= lo) & (snr < hi)
        if m.sum() > 20:
            print(f"  snr {lo:>4}-{hi:<5} n={m.sum():5d}  median {np.median(r[m]):+.4f}  mean {np.mean(r[m]):+.4f}"
                  f"  K-miss rate {np.mean(np.abs(10 ** r[m] - 1) >= 0.1):.3f}")
    rng = np.random.default_rng(0)
    A = rng.random(len(Y)) < 0.5
    B = ~A
    cs = np.concatenate([[0.0], np.geomspace(0.1, 30, 40)])
    sA = [ok(pred[A, 0], corrected(pred[A], nobs[A], sig[A], c), Y[A]).sum() for c in cs]
    c_best = cs[int(np.argmax(sA))]
    base_B = ok(pred[B, 0], pred[B, 1], Y[B])
    new_B = ok(pred[B, 0], corrected(pred[B], nobs[B], sig[B], c_best), Y[B])
    g, l = int((new_B & ~base_B).sum()), int((~new_B & base_B).sum())
    print(f"best c on half A = {c_best:.3f} (A: {sA[0]} -> {max(sA)} successes)")
    print(f"half B (held out): {base_B.mean():.4f} -> {new_B.mean():.4f}   +{g}/-{l}")
