"""ONE-TIME CHECK on validation (never tune on it): score a plugin with the task-card metric, per group.

    .venv/bin/python tools/score_val.py --plugin build/sub_v2/final_model.py [--plain] [--workers 8]
--plain forces every rv_v2 switch off (= official baseline decision rule).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import success  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", type=Path, required=True)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", nargs="*", default=[], help="CONFIG overrides key=value (python literals)")
    ap.add_argument("--save", type=Path, help="save predictions [N,3] .npy (for analysis/headroom.py)")
    a = ap.parse_args()
    v = ROOT / "data/exo/val"
    X, Y, G = (np.load(v / f"validation_{k}.npy") for k in ("inputs", "targets", "groups"))
    nobs = X[:, 3].sum(1)
    names = {}
    for g in np.unique(G):
        m = G == g
        names[g] = "in_distribution" if m.sum() == 2000 else ("sparse" if np.all(nobs[m] == 15) else "high_ecc")
    spec = importlib.util.spec_from_file_location(a.plugin.stem, a.plugin)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if a.plain:
        mod.CONFIG.update(jitter_fit=False, occam_weight=0.0, bound_penalty=0.0, yearly_aliases=False, n_peaks=3)
    import ast
    for kv in a.set:
        k, v = kv.split("=", 1)
        mod.CONFIG[k] = ast.literal_eval(v)
    t0 = time.time()
    p = mod.predict(X, n_workers=a.workers)
    if a.save:
        np.save(a.save, p)
    dt = time.time() - t0
    ok = success(p, Y.astype(float))[0]
    res = {"plugin": str(a.plugin.name), "plain": a.plain, "overrides": a.set, "success_rate": float(ok.mean()), "sec": dt}
    for g, n in names.items():
        res[f"success_rate_{n}"] = float(ok[G == g].mean())
    res["median_abs_e_error"] = float(np.median(np.abs(p[:, 2] - Y[:, 2])))
    res["median_rel_k_error"] = float(np.median(np.abs(10 ** p[:, 1] - 10 ** Y[:, 1]) / 10 ** Y[:, 1]))
    print(json.dumps(res, indent=1))
    with open(ROOT / "build/val_checks.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
