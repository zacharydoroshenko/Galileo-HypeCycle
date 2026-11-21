"""Search-failure vs identifiability diagnostic (uses only train labels P, K, e).

For every star we compare the penalised neg-log-likelihood of
  (a) our chosen solution, vs
  (b) an "oracle" fit started at the TRUE P and e (grid over phase, then polish).
If (b) is clearly better than (a) and (a) is wrong -> SEARCH failure (fixable: better candidates/optimiser).
If (a) is as good or better yet wrong -> data do not identify the answer (fix needs priors, not search).

    python tools/diagnose.py --synth standard --n 80
    python tools/diagnose.py --data_dir ~/work/exoplanet-orbits/data/train --n 200
Writes build/diagnose.csv for plots / the report.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "train"))
import rv_plugin as rp  # noqa: E402

DNLL = 2.0  # significance threshold in neg-log-likelihood units


def one(args):
    x, ytrue = args
    cfg = rp.CONFIG
    pred, best = rp.fit_star(x, cfg)
    t, y, s = rp.parse_star(x, cfg)
    t = t - t.min()
    T = max(t.max() - t.min(), 1.0)
    Pt, Kt, et = 10 ** ytrue[0], 10 ** ytrue[1], float(ytrue[2])
    # oracle: phase grid at true P/e, then polish
    M0s = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    vals = [rp.nll(t, y, s, Pt, et, m, cfg["jitter_init"])[0] for m in M0s]
    orc = rp.polish(t, y, s, Pt, et, M0s[int(np.argmin(vals))], cfg["jitter_init"], cfg)
    okP = abs(10 ** pred[0] / Pt - 1) <= 0.01
    okK = abs(10 ** pred[1] / Kt - 1) <= 0.10
    oP = abs(orc["P"] / Pt - 1) <= 0.01
    oK = abs(orc["K"] / Kt - 1) <= 0.10
    if okP and okK:
        cat = "found"
    elif orc["score"] < best["score"] - DNLL:
        cat = "SEARCH failure"
    elif not (oP and oK):
        cat = "unidentifiable (even oracle fit misses)"
    else:
        cat = "ambiguous (wrong solution fits as well)"
    snr = Kt / np.sqrt(np.mean(s**2) + best.get("jitter", 0) ** 2)
    return dict(cat=cat, n_obs=len(t), P_true=Pt, K_true=Kt, e_true=et, P_over_T=Pt / T, snr=snr,
                P_pred=10 ** pred[0], K_pred=10 ** pred[1], e_pred=pred[2],
                nll_pred=best["score"], nll_oracle=orc["score"], okP=okP, okK=okK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", choices=["standard", "eccentric", "sparse"])
    ap.add_argument("--data_dir", type=Path)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.synth:
        from synth import make_dataset
        X, Y = make_dataset(a.n, a.synth, a.seed)
    else:
        from train import load_train
        X, Y = load_train(a.data_dir)
        idx = np.random.default_rng(a.seed).choice(len(X), min(a.n, len(X)), replace=False)
        X, Y = X[idx], Y[idx]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, zip(X, Y)))
    cats = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r)
    print(f"N={len(rows)}")
    for c, rs in sorted(cats.items(), key=lambda z: -len(z[1])):
        print(f"  {c:42s} {len(rs) / len(rows):.3f}   median SNR {np.median([r['snr'] for r in rs]):6.1f}"
              f"  median P/T {np.median([r['P_over_T'] for r in rs]):5.2f}"
              f"  median n_obs {np.median([r['n_obs'] for r in rs]):4.0f}"
              f"  median e {np.median([r['e_true'] for r in rs]):.2f}")
    (ROOT / "build").mkdir(exist_ok=True)
    with open(ROOT / "build" / "diagnose.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote build/diagnose.csv")


if __name__ == "__main__":
    main()
