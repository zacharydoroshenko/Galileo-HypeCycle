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

    if not a.no_tune:
        X, Y = load_train(a.data_dir)
        idx = rng.choice(len(X), size=min(a.n_tune, len(X)), replace=False)
        X, Y = X[idx], Y[idx]
        # Small search space: extend once you know what helps (keep it inside the 3600 s retrain limit).
        space = {
            "e_penalty": [0.0, 1.0, 3.0],
            "n_peaks": [6, 10],
        }
        t0 = time.time()
        for vals in itertools.product(*space.values()):
            if time.time() - t0 > a.budget_s:
                break
            cfg = dict(base, **dict(zip(space, vals)))
            s = score(mod, X, Y, cfg, a.workers)
            log.append({"cfg": dict(zip(space, vals)), "train_success": s})
            print(json.dumps(log[-1]), flush=True)
            if best is None or s > best:
                best, best_cfg = s, cfg

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
