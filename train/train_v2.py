"""Reproduce the final rv_v2 algorithm artifact (kind=algorithm, no learned weights).

    python train/train_v2.py --data_dir "$DATA_DIR" --seed "$SEED" --out final_model.py

The algorithm source is train/rv_v2_source.py (= official baseline + switchable improvements, all OFF
by default, i.e. the baseline decision rule; identical to initial_algorithm.py). This script writes the
final plugin with the SELECTED settings baked into CONFIG:

    occam_weight = 0.5   Laplace evidence term 0.5*logdet(J^T J) in the candidate score
    k_debias_c   = 6.95  low-SNR amplitude-bias correction K/sqrt(1 + c/SNR^2); c fitted on 6,000 TRAIN
                         stars disjoint from every evaluation set (train/selection/make_calib_preds.py +
                         k_calib.py: maximise successes on half A; held-out half B +54/-28)

Selection was done on the TRAIN split only (never validation): paired comparisons vs the baseline rule
on held-out train stars (1,000 standard + 600 subsampled to 15 observations), see
train/selection/confirm_v2.py and train/selection/ablate_v2.py. The P/K bounds are read from the
train labels here (they only matter if bound_penalty > 0, which is off). The algorithm draws no
random numbers; --seed is recorded for the interface.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SELECTED = {"occam_weight": 0.5, "k_debias_c": 6.95}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    np.random.seed(a.seed)
    src = (Path(__file__).resolve().parent / "rv_v2_source.py").read_text()
    cfg = dict(SELECTED)
    tgt = a.data_dir / "train_targets.npy"
    if tgt.exists():
        y = np.load(tgt)
        cfg["P_bounds"] = [float(10 ** y[:, 0].min()), float(10 ** y[:, 0].max())]
        cfg["K_bounds"] = [float(10 ** y[:, 1].min()), float(10 ** y[:, 1].max())]
    for k, v in cfg.items():
        src, n = re.subn(rf'^(    "{k}": ).*?,(\s*#.*)?$', lambda m: f"{m.group(1)}{v!r},{m.group(2) or ''}", src,
                         count=1, flags=re.M)
        assert n == 1, f"CONFIG key {k} not found"
    a.out.write_text(src)
    print(f"wrote {a.out} with {cfg}")


if __name__ == "__main__":
    main()
