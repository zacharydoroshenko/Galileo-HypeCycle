"""'Training' for the algorithm submission: tune CONFIG hyperparameters on the TRAIN split only,
then write the final self-contained plugin with the tuned CONFIG baked in.

Reproduction command (submission.json train_command):
    python train/train.py --data_dir "$DATA_DIR" --seed "$SEED" --out "$OUTPUT_DIR/rv_plugin.py"

Only reads $DATA_DIR (train files). Never point it at validation/test labels.
The template is train/rv_template.py (identical copy of initial_model.py at packaging time).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(path, name="rv_template_mod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_train(data_dir: Path):
    """Return X [N,4,60], Y [N,3]. EDIT ME once the real file layout is known (see TASK_CARD.md).
    Tries common layouts: *.npz with X/Y, *.npy pairs, *.h5."""
    data_dir = Path(data_dir)
    if (data_dir / "train_inputs.npy").exists():  # official exoplanet-orbits layout
        return np.load(data_dir / "train_inputs.npy"), np.load(data_dir / "train_targets.npy")
    npz = sorted(data_dir.glob("*train*.npz")) or sorted(data_dir.glob("*.npz"))
    if npz:
        d = np.load(npz[0])
        kx = next(k for k in d.files if k.lower() in ("x", "inputs", "features", "data"))
        ky = next(k for k in d.files if k.lower() in ("y", "targets", "labels", "params"))
        return d[kx], d[ky]
    xs = sorted(data_dir.glob("*X*train*.npy")) or sorted(data_dir.glob("X*.npy"))
    ys = sorted(data_dir.glob("*y*train*.npy")) or sorted(data_dir.glob("[yY]*.npy"))
    if xs and ys:
        return np.load(xs[0]), np.load(ys[0])
    h5 = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("*.hdf5"))
    if h5:
        import h5py
        with h5py.File(h5[0], "r") as f:
            keys = list(f.keys())
            kx = next(k for k in keys if k.lower() in ("x", "inputs", "features"))
            ky = next(k for k in keys if k.lower() in ("y", "targets", "labels"))
            return f[kx][()], f[ky][()]
    raise FileNotFoundError(f"no recognised train files in {data_dir}: {os.listdir(data_dir)}")


def build_prior(Y, n_bins=24, alpha=0.5):
    """Smoothed 2-D histogram of TRAIN (log10 P, log10 K) -> log density table + edges."""
    lp, lk = Y[:, 0], Y[:, 1]
    pad_p, pad_k = 0.1 * np.ptp(lp) + 0.05, 0.1 * np.ptp(lk) + 0.05
    ep = np.linspace(lp.min() - pad_p, lp.max() + pad_p, n_bins + 1)
    ek = np.linspace(lk.min() - pad_k, lk.max() + pad_k, n_bins + 1)
    H, _, _ = np.histogram2d(lp, lk, bins=[ep, ek])
    k = np.array([0.25, 0.5, 0.25])  # light separable blur
    for ax in (0, 1):
        H = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 1, mode="edge"), k, "valid"), ax, H)
    H = H + alpha
    dens = H / (H.sum() * np.diff(ep)[0] * np.diff(ek)[0])
    return {"prior_lp_edges": np.round(ep, 4).tolist(), "prior_lk_edges": np.round(ek, 4).tolist(),
            "prior_logpdf": np.round(np.log(dens), 3).tolist()}


def score(mod, X, Y, cfg, workers):
    mod.CONFIG.update(cfg)
    pred = mod.predict(X, n_workers=workers)
    okP = np.abs(10 ** (pred[:, 0] - Y[:, 0]) - 1) <= 0.01
    okK = np.abs(10 ** (pred[:, 1] - Y[:, 1]) - 1) <= 0.10
    return float((okP & okK).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_tune", type=int, default=400, help="train stars used for tuning")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("N_WORKERS", os.cpu_count() or 1)))
    ap.add_argument("--budget_s", type=float, default=2400, help="stop tuning after this many seconds")
    ap.add_argument("--no_tune", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    template = HERE / "rv_template.py"
    mod = load_module(template)
    base = dict(mod.CONFIG)
    best_cfg, best = dict(base), None
    log = []

    X, Y = load_train(a.data_dir)
    base.update(build_prior(Y))            # prior from ALL train labels (cheap, legitimate)
    best_cfg = dict(base)
    if not a.no_tune:
        idx = rng.choice(len(X), size=min(a.n_tune, len(X)), replace=False)
        Xt, Yt = X[idx], Y[idx]
        # Coordinate-wise search (keeps the retrain inside the 3600 s limit). First value = default.
        space = {
            "prior_weight": [0.0, 0.5, 1.0, 2.0],
            "kep_on": [True, False],
            "e_penalty": [0.0, 1.0, 3.0],
            "n_peaks": [6, 10],
        }
        t0 = time.time()
        best = score(mod, Xt, Yt, best_cfg, a.workers)
        log.append({"cfg": "base", "train_success": best})
        print(json.dumps(log[-1]), flush=True)
        for key, values in space.items():
            for v in values:
                if v == best_cfg.get(key) or time.time() - t0 > a.budget_s:
                    continue
                cfg = dict(best_cfg, **{key: v})
                sc = score(mod, Xt, Yt, cfg, a.workers)
                log.append({"cfg": {key: v}, "train_success": sc})
                print(json.dumps(log[-1]), flush=True)
                if sc > best:
                    best, best_cfg = sc, cfg

    src = template.read_text()
    new_cfg = "CONFIG = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in best_cfg.items()) + "}\n"
    src, n = re.subn(r"^CONFIG = \{.*?^\}\n", new_cfg, src, count=1, flags=re.S | re.M)
    assert n == 1, "could not locate CONFIG block in template"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(src)
    (a.out.parent / "tuning_log.json").write_text(json.dumps({"best": best, "cfg": best_cfg, "log": log}, indent=2))
    print(f"wrote {a.out} (train success {best})")


if __name__ == "__main__":
    main()
