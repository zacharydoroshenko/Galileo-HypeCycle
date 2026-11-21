"""Local scorer: success_rate = share of stars with |P/P_true - 1| <= 1% and |K/K_true - 1| <= 10%.

    python tools/evaluate.py --data data/synth_standard.npz [--n 200] [--workers 8]
    python tools/evaluate.py --synth standard --n 200        # generate on the fly

Also prints a failure breakdown (period alias / harmonic / K-only miss) - this is what
the report's error analysis should be built from. Keep the official evaluator as ground
truth once it ships with the package; this is only a mirror of the stated metric.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_plugin(path):
    spec = importlib.util.spec_from_file_location("rv_plugin_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def success(pred, true, p_tol=0.01, k_tol=0.10):
    P, Pt = 10 ** pred[:, 0], 10 ** true[:, 0]
    K, Kt = 10 ** pred[:, 1], 10 ** true[:, 1]
    okP = np.abs(P / Pt - 1) <= p_tol
    okK = np.abs(K / Kt - 1) <= k_tol
    return okP & okK, okP, okK


def breakdown(pred, true):
    ratio = 10 ** (pred[:, 0] - true[:, 0])
    _, okP, okK = success(pred, true)
    cats = {"found": okP & okK, "P ok, K miss": okP & ~okK}
    harm = np.zeros(len(ratio), bool)
    for r in (0.5, 2.0, 1 / 3, 3.0):
        harm |= np.abs(ratio / r - 1) < 0.02
    cats["P harmonic (x2,x1/2,x3,x1/3)"] = ~okP & harm
    cats["P other miss"] = ~okP & ~harm
    return {k: float(v.mean()) for k, v in cats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--synth", choices=["standard", "eccentric", "sparse"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plugin", type=Path, default=ROOT / "solution" / "rv_plugin.py")
    a = ap.parse_args()

    if a.synth:
        sys.path.insert(0, str(ROOT / "tools"))
        from synth import make_dataset
        X, Y = make_dataset(a.n, a.synth, a.seed)
    else:
        d = np.load(a.data)
        X, Y = d["X"][: a.n], d["Y"][: a.n]

    plug = load_plugin(a.plugin)
    t0 = time.time()
    pred = plug.predict(X, n_workers=a.workers)
    dt = time.time() - t0
    assert pred.shape == (len(X), 3) and np.all(np.isfinite(pred)), "contract violated"
    ok, okP, okK = success(pred, Y)
    print(f"N={len(X)}  success_rate={ok.mean():.4f}  (P ok {okP.mean():.3f}, K ok {okK.mean():.3f})")
    print(f"time {dt:.1f}s  ({dt / len(X) * 1000:.0f} ms/star with {a.workers} workers)")
    for k, v in breakdown(pred, Y).items():
        print(f"  {k:32s} {v:.3f}")
    e_err = np.abs(pred[:, 2] - Y[:, 2])
    print(f"  median |e err|                   {np.median(e_err):.3f}")


if __name__ == "__main__":
    main()
