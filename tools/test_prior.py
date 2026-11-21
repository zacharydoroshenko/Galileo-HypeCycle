"""A/B the learned (log P, log K) prior: build it from a TRAIN npz, score on synthetic test regimes.

    python tools/test_prior.py --train data/synth_train --n 60 --weights 0 1 2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools"), str(ROOT / "train")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--regimes", nargs="+", default=["standard", "sparse"])
    ap.add_argument("--kep", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import rv_plugin as rp
    from evaluate import success
    from synth import make_dataset
    from train import build_prior, load_train
    _, Ytr = load_train(a.train)
    rp.CONFIG.update(build_prior(Ytr), kep_on=a.kep)
    for reg in a.regimes:
        X, Y = make_dataset(a.n, reg, 123)
        for w in a.weights:
            rp.CONFIG["prior_weight"] = w
            t0 = time.time()
            p = rp.predict(X, n_workers=a.workers)
            ok, okP, okK = success(p, Y)
            print(f"{reg:9s} prior_weight={w:<4} success {ok.mean():.3f}  P ok {okP.mean():.3f}  "
                  f"K ok {okK.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
