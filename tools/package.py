"""Stage, reproduce-check, and zip an algorithm submission for exoplanet-orbits.

    python tools/package.py --data_dir ~/work/exoplanet-orbits/data/train [--no_tune] [--quick]

Steps (mirrors the track2-guide checklist):
  1. stage build/submission/ with submission.json at the root, initial_model.py,
     train/train.py + train/rv_template.py
  2. run the exact train_command in a FRESH temp dir with DATA_DIR=train only -> rv_plugin.py
  3. smoke-test predict() on a few train stars (shape, finite, not all identical)
  4. zip the *contents* (not the folder) to build/submission-<timestamp>.zip and print sha256
Read the manifest against TASK_CARD.md before uploading - field names come from the guide.
"""
from __future__ import annotations

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
BUILD = ROOT / "build"
SEED = 0
TRAIN_CMD = 'python train/train.py --data_dir "$DATA_DIR" --seed "$SEED" --out "$OUTPUT_DIR/rv_plugin.py"'


def stage(extra_train_args: str) -> Path:
    d = BUILD / "submission"
    if d.exists():
        shutil.rmtree(d)
    (d / "train").mkdir(parents=True)
    src = ROOT / "solution" / "rv_plugin.py"
    shutil.copy(src, d / "initial_model.py")
    shutil.copy(src, d / "train" / "rv_template.py")
    shutil.copy(ROOT / "train" / "train.py", d / "train" / "train.py")
    manifest = {
        "kind": "algorithm",
        "seed": SEED,
        "initial_artifact": "initial_model.py",
        "plugin": "rv_plugin.py",
        "train_command": TRAIN_CMD + (" " + extra_train_args if extra_train_args else ""),
        "train_dir": "train/",
    }
    (d / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return d


def reproduce(stage_dir: Path, data_dir: Path, manifest: dict) -> Path:
    work = Path(tempfile.mkdtemp(prefix="repro_"))
    shutil.copytree(stage_dir, work, dirs_exist_ok=True)
    (work / "rv_plugin.py").unlink(missing_ok=True)
    env = dict(os.environ, DATA_DIR=str(data_dir), PACKAGE_DIR=str(data_dir.parent),
               OUTPUT_DIR=str(work), SEED=str(manifest["seed"]))
    t0 = time.time()
    print(f"[repro] running in {work}: {manifest['train_command']}")
    subprocess.run(["bash", "-c", manifest["train_command"]], cwd=work, env=env, check=True)
    print(f"[repro] train_command took {time.time() - t0:.0f}s (limit 3600s)")
    out = work / manifest["plugin"]
    assert out.exists(), "train_command did not write the plugin"
    return out


def smoke(plugin_path: Path, data_dir: Path):
    sys.path.insert(0, str(ROOT / "train"))
    from train import load_train
    X, _ = load_train(data_dir)
    spec = importlib.util.spec_from_file_location("plug_smoke", plugin_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    p = m.predict(X[:16], n_workers=4)
    assert p.shape == (16, 3), p.shape
    assert np.all(np.isfinite(p)), "non-finite predictions"
    assert not np.all(p == p[0]), "all predictions identical"
    print("[smoke] predict OK:", p[:3].round(3).tolist())


def make_zip(stage_dir: Path) -> Path:
    z = BUILD / f"submission-{time.strftime('%H%M%S')}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(stage_dir.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.name != ".DS_Store":
                zf.write(f, f.relative_to(stage_dir).as_posix())
    names = zipfile.ZipFile(z).namelist()
    assert "submission.json" in names, names
    print("[zip] contents:", names)
    print(f"[zip] {z}  {z.stat().st_size} bytes  sha256 {hashlib.sha256(z.read_bytes()).hexdigest()}")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True, help="TRAIN split directory only")
    ap.add_argument("--no_tune", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tune on 60 stars (pipeline check only)")
    a = ap.parse_args()
    extra = " ".join(x for x in ["--no_tune" if a.no_tune else "", "--n_tune 60" if a.quick else ""] if x)
    d = stage(extra)
    manifest = json.loads((d / "submission.json").read_text())
    plugin = reproduce(d, a.data_dir.resolve(), manifest)
    shutil.copy(plugin, d / manifest["plugin"])
    log = plugin.parent / "tuning_log.json"
    if log.exists():
        shutil.copy(log, BUILD / "tuning_log.json")  # keep for the report, not in the zip
    smoke(d / manifest["plugin"], a.data_dir.resolve())
    make_zip(d)


if __name__ == "__main__":
    main()
