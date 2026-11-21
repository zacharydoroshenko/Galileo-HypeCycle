"""One-command setup of the exoplanet-orbits task package + data (nothing large is committed to git).

    python3 tools/get_data.py            # stdlib only, works before the venv exists

Downloads from the official sources and verifies checksums:
  - GitHub release track2-2026-11-21: exoplanet-orbits.zip and the official example submission
    (sha256 checked against the release's SHA256SUMS.txt)
  - Zenodo record 24361910: the data files (md5 checked against the package's data_source.json)
Creates:
  package_exo/                  task card, baseline, simulator, metric plugins (organizer code)
  package_exo/official_submission/   official example submission (manifest format reference)
  data/raw/                     the Zenodo files under their original names
  data/exo/train/               train_inputs.npy, train_targets.npy        <- the ONLY dir tuning may read
  data/exo/val/                 validation_{inputs,targets,groups}.npy     <- checking only, never tune on it
  data/exo/train_sub300.npz     fixed 300-star train subset (seed 0) for quick A/B runs (needs numpy)
Re-running is safe: verified files are not downloaded again.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/yuema137/siderius-hackathon/releases/download/track2-2026-11-21"
ZENODO = "https://zenodo.org/records/24361910/files"
DATA_FILES = ["rv_train_inputs.npy", "rv_train_targets.npy", "rv_train_params.npy",
              "rv_validation_inputs.npy", "rv_validation_targets.npy", "rv_validation_groups.npy"]


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, check=None):
    if dest.exists() and (check is None or check(dest)):
        print(f"  ok (cached) {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "galileo-team-setup"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    if check is not None and not check(dest):
        dest.unlink()
        sys.exit(f"checksum mismatch for {dest.name}; deleted it, re-run to retry")
    print(f"  ok {dest.relative_to(ROOT)}")


def main():
    dl = ROOT / "data" / "downloads"
    print("[1/3] task package (GitHub release)")
    download(f"{RELEASE}/SHA256SUMS.txt", dl / "SHA256SUMS.txt")
    sums = {}
    for line in (dl / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    for name in ["exoplanet-orbits.zip", "exoplanet-orbits-official-submission.zip"]:
        download(f"{RELEASE}/{name}", dl / name, lambda p, n=name: digest(p, "sha256") == sums[n])
    pkg = ROOT / "package_exo"
    if pkg.exists():
        shutil.rmtree(pkg)
    tmp = dl / "unzipped"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(dl / "exoplanet-orbits.zip") as z:
        z.extractall(tmp)
    shutil.move(str(tmp / "rv-exoplanet"), str(pkg))
    with zipfile.ZipFile(dl / "exoplanet-orbits-official-submission.zip") as z:
        z.extractall(pkg / "official_submission")
    shutil.rmtree(tmp)
    print(f"  unpacked -> {pkg.relative_to(ROOT)}/ (read TASK_CARD.md and RELEASE_CONTRACT.md)")

    print("[2/3] data (Zenodo)")
    md5 = {f["name"]: f["md5"] for f in json.loads((pkg / "data_source.json").read_text())["files"]}
    raw = ROOT / "data" / "raw"
    for name in DATA_FILES:
        download(f"{ZENODO}/{name}?download=1", raw / name, lambda p, n=name: digest(p, "md5") == md5[n])

    print("[3/3] layout")
    layout = {
        "train/train_inputs.npy": "rv_train_inputs.npy",
        "train/train_targets.npy": "rv_train_targets.npy",
        "val/validation_inputs.npy": "rv_validation_inputs.npy",
        "val/validation_targets.npy": "rv_validation_targets.npy",
        "val/validation_groups.npy": "rv_validation_groups.npy",
    }
    exo = ROOT / "data" / "exo"
    for dst, src in layout.items():
        (exo / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw / src, exo / dst)
    try:
        import numpy as np
        X = np.load(exo / "train" / "train_inputs.npy")
        Y = np.load(exo / "train" / "train_targets.npy")
        i = np.random.default_rng(0).choice(len(X), 300, replace=False)
        np.savez(exo / "train_sub300.npz", X=X[i].astype(float), Y=Y[i].astype(float))
        print(f"  train {X.shape}, wrote data/exo/train_sub300.npz")
    except ImportError:
        print("  numpy not found: re-run with .venv/bin/python to also build train_sub300.npz")
    print("done.")


if __name__ == "__main__":
    main()
