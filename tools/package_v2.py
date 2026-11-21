"""Stage + reproduce-check + zip the rv_v2 algorithm submission (mirrors the official example layout).

    .venv/bin/python tools/package_v2.py --data_dir data/exo/train
Zip lands in build/v2-submission-<HHMMSS>.zip with submission.json at the root.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "sub_v2"
SEED = 0
MANIFEST = {
    "kind": "algorithm",
    "seed": SEED,
    "initial_artifact": "initial_algorithm.py",
    "plugin": "final_model.py",
    "train_command": 'python train/train_v2.py --data_dir "$DATA_DIR" --seed "$SEED" --out final_model.py',
    "train_dir": "train/",
}


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "train" / "selection").mkdir(parents=True)
    src = ROOT / "solution" / "rv_v2.py"
    shutil.copy(src, STAGE / "initial_algorithm.py")
    shutil.copy(src, STAGE / "train" / "rv_v2_source.py")
    shutil.copy(ROOT / "train" / "train_v2.py", STAGE / "train" / "train_v2.py")
    for f in ["confirm_v2.py", "ablate_v2.py", "make_eval_sets.py", "make_confirm_sets.py", "evaluate.py",
              "make_sparse_v2.py", "make_calib_preds.py", "k_calib.py"]:
        shutil.copy(ROOT / "tools" / f, STAGE / "train" / "selection" / f)
    (STAGE / "submission.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")


def reproduce(data_dir):
    work = Path(tempfile.mkdtemp(prefix="repro_v2_"))
    shutil.copytree(STAGE, work, dirs_exist_ok=True)
    (work / "final_model.py").unlink(missing_ok=True)
    env = dict(os.environ, DATA_DIR=str(data_dir), OUTPUT_DIR=str(work), SEED=str(SEED),
               PACKAGE_DIR=str(ROOT / "package_exo"))
    subprocess.run(["bash", "-c", MANIFEST["train_command"]], cwd=work, env=env, check=True)
    out = work / "final_model.py"
    assert out.exists(), "train_command did not write final_model.py"
    return out


def smoke(plugin, data_dir):
    spec = importlib.util.spec_from_file_location("v2_smoke", plugin)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.path.insert(0, str(ROOT / "train"))
    from train_v2 import SELECTED
    for k, v in SELECTED.items():
        assert m.CONFIG[k] == v, (k, m.CONFIG[k], v)
    X = np.load(data_dir / "train_inputs.npy")[:32]
    p = m.predict(X, n_workers=4)
    assert p.shape == (32, 3) and np.all(np.isfinite(p)), "bad predictions"
    assert np.all((p[:, 2] >= 0) & (p[:, 2] < 1)), "e out of [0,1)"
    assert np.std(p[:, 0]) > 0.05, "period dispersion below health floor"
    print("[smoke] OK, CONFIG:", {k: m.CONFIG[k] for k in (*SELECTED, "P_bounds", "K_bounds")})


def make_zip():
    z = ROOT / "build" / f"v2-submission-{time.strftime('%H%M%S')}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store":
                zf.write(f, f.relative_to(STAGE).as_posix())
    names = zipfile.ZipFile(z).namelist()
    assert "submission.json" in names
    for n in names:
        assert not any(s in n.lower() for s in ("key", ".env", "secret")), n
    print("[zip]", names)
    print(f"[zip] {z}\n      {z.stat().st_size} bytes  sha256 {hashlib.sha256(z.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True, help="TRAIN-only directory")
    a = ap.parse_args()
    d = a.data_dir.resolve()
    stage()
    final = reproduce(d)
    shutil.copy(final, STAGE / "final_model.py")
    smoke(STAGE / "final_model.py", d)
    make_zip()
