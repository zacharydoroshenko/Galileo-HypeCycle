"""Sanity check: pool == serial, CONFIG reaches workers, no crashes.  python tools/check_parallel.py"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools")]

if __name__ == "__main__":
    import rv_plugin as rp
    from synth import make_dataset
    X, _ = make_dataset(8, "standard", 1)
    t = time.time(); a = rp.predict(X, n_workers=1); t1 = time.time() - t
    t = time.time(); b = rp.predict(X, n_workers=4); t2 = time.time() - t
    print(f"serial {t1:.1f}s  pool {t2:.1f}s  identical={np.allclose(a, b)}")
    rp.CONFIG.update(n_peaks=1, add_aliases=False, add_harmonics=False, n_phase=4, n_e=2, n_polish=1)
    c_pool = rp.predict(X, n_workers=4)
    c_ser = rp.predict(X, n_workers=1)
    print("config reaches workers (pool == serial under modified CONFIG, and differs from default):",
          np.allclose(c_pool, c_ser) and not np.allclose(b, c_pool))
