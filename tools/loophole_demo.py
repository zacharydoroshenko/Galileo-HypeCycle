"""Reward-hacking audit (LOCAL ONLY, never uploaded): metric-only transformations of submission 16's predictions.

  e_const : every e -> 0.1                      (e is not in the success criterion)
  k_recent: K -> 0.995 K                        (recentre the log-asymmetric box [0.9K, 1.1K] = [-0.0458, +0.0414] dex)
  both    : e_const + k_recent  (the "loophole model")
Scored on (a) 6,000 TRAIN calibration stars and (b) validation (audit of the metric, not used to tune anything).
    .venv/bin/python tools/loophole_demo.py   -> results/loophole_demo.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from k_calib import corrected  # noqa: E402


def metrics(pred, Y, groups=None):
    P, Pt, K, Kt = 10 ** pred[:, 0], 10 ** Y[:, 0], 10 ** pred[:, 1], 10 ** Y[:, 1]
    ok = (np.abs(P - Pt) / Pt < 0.01) & (np.abs(K - Kt) / Kt < 0.10)
    out = {"success": float(ok.mean()), "median_abs_e_error": float(np.median(np.abs(pred[:, 2] - Y[:, 2]))),
           "median_rel_k_error": float(np.median(np.abs(K - Kt) / Kt))}
    if groups is not None:
        for g, name in [(0, "in_distribution"), (1, "high_ecc"), (2, "sparse")]:
            out[f"success_{name}"] = float(ok[groups == g].mean())
            out[f"median_abs_e_error_{name}"] = float(np.median(np.abs(pred[groups == g, 2] - Y[groups == g, 2])))
    return out, ok


def variants(pred):
    e = pred.copy(); e[:, 2] = 0.1
    k = pred.copy(); k[:, 1] += math.log10(0.995)
    b = e.copy(); b[:, 1] += math.log10(0.995)
    return {"submission16": pred, "e_const": e, "k_recent": k, "loophole_both": b}


def sign_p(g, l):
    n = g + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(g, l) + 1)) / 2**n)


if __name__ == "__main__":
    res = {}
    d = np.load(ROOT / "data/exo/calib_preds.npz")
    p_tr = d["pred"].copy(); p_tr[:, 1] = corrected(d["pred"], d["nobs"].astype(float), d["sig_eff"], 6.95)
    Vy = np.load(ROOT / "data/exo/val/validation_targets.npy").astype(float)
    Vg = np.load(ROOT / "data/exo/val/validation_groups.npy")
    p_va = np.load(ROOT / "results/val_preds_final.npy")
    for split, pred, Y, G in [("train_calib6000", p_tr, d["Y"], None), ("validation", p_va, Vy, Vg)]:
        res[split] = {}
        base_ok = None
        for name, pv in variants(pred).items():
            m, ok = metrics(pv, Y, G)
            if base_ok is None:
                base_ok = ok
            g, l = int((ok & ~base_ok).sum()), int((~ok & base_ok).sum())
            m["vs_submission16"] = [g, l, sign_p(g, l)]
            res[split][name] = m
            print(f"{split:16s} {name:14s} success {m['success']:.5f} (+{g}/-{l})  median|de| {m['median_abs_e_error']:.4f}"
                  f"  median rel dK {m['median_rel_k_error']:.4f}")
    (ROOT / "results/loophole_demo.json").write_text(json.dumps(res, indent=1))
    print("wrote results/loophole_demo.json")
