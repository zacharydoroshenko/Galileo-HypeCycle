"""Share of stars whose TRUE period is among the candidate periods (search recall, before fitting).

    python tools/candidate_recall.py --synth eccentric --n 80 [--kep_off]
    python tools/candidate_recall.py --data_dir <train dir> --n 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools"), str(ROOT / "train")]
import rv_plugin as rp  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", choices=["standard", "eccentric", "sparse"])
    ap.add_argument("--data_dir", type=Path)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--kep_off", action="store_true")
    a = ap.parse_args()
    if a.synth:
        from synth import make_dataset
        X, Y = make_dataset(a.n, a.synth, a.seed)
    else:
        from train import load_train
        X, Y = load_train(a.data_dir)
        X, Y = X[: a.n], Y[: a.n]
    rp.CONFIG["kep_on"] = not a.kep_off
    hit, ncand = [], []
    for x, y in zip(X, Y):
        t, v, s = rp.parse_star(x)
        t = t - t.min()
        c, T = rp.candidate_periods(t, v, s, rp.CONFIG)
        hit.append(np.min(np.abs(1 / c - 1 / 10 ** y[0])) * T < 0.5)
        ncand.append(len(c))
    print(f"recall {np.mean(hit):.3f}   mean candidates {np.mean(ncand):.1f}   kep_on={rp.CONFIG['kep_on']}")
