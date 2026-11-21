"""Synthetic RV dataset generator mimicking the exoplanet-orbits task description.

Produces X [N, 4, 60] (t, rv, err, mask; zero padded) and Y [N, 3] = [log10 P, log10 K, e].
Regimes: "standard", "eccentric" (e beyond the training range), "sparse" (few obs).
Tune the distributions once the real TRAIN data is inspected (tools/inspect_data.py).

    python tools/synth.py --n 500 --regime standard --out data/synth_standard.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution"))
from rv_plugin import rv_model  # noqa: E402

L = 60


def sample_times(rng, n_obs, n_seasons):
    """Night-time observations clustered in observing seasons (~yearly), integer-ish days."""
    season_len = rng.uniform(90, 240)
    starts = np.arange(n_seasons) * 365.25 + rng.uniform(0, 30, n_seasons)
    per = rng.multinomial(n_obs, np.ones(n_seasons) / n_seasons)
    ts = []
    for st, k in zip(starts, per):
        nights = np.sort(rng.choice(int(season_len), size=min(k, int(season_len)), replace=False))
        ts.append(st + nights + rng.normal(0.0, 0.08, len(nights)))  # night-time scatter
    return np.sort(np.concatenate(ts))


def make_star(rng, regime):
    if regime == "sparse":
        n_obs = int(rng.integers(8, 16))
    else:
        n_obs = int(rng.integers(20, L + 1))
    n_seasons = int(rng.integers(2, 6))
    t = sample_times(rng, n_obs, n_seasons)
    T = t.max() - t.min()
    P = 10 ** rng.uniform(0.0, np.log10(max(2.0, 1.5 * T)))
    K = 10 ** rng.uniform(0.3, 2.3)
    if regime == "eccentric":
        e = rng.uniform(0.6, 0.92)
    else:
        e = min(rng.beta(0.867, 3.03), 0.6)  # Kipping 2013 prior, truncated
    w = rng.uniform(0, 2 * np.pi)
    M0 = rng.uniform(0, 2 * np.pi)
    gamma = rng.normal(0, 20)
    err = rng.uniform(0.8, 4.0, len(t))
    jitter = rng.uniform(0.0, 4.0)
    y = rv_model(t, P, K, e, w, M0, gamma) + rng.normal(0, 1, len(t)) * np.sqrt(err**2 + jitter**2)
    x = np.zeros((4, L))
    n = len(t)
    t0 = rng.uniform(0, 3000) + 2450000 * 0  # set to real offset style if data uses BJD
    x[0, :n], x[1, :n], x[2, :n], x[3, :n] = t + t0, y, err, 1.0
    return x, np.array([np.log10(P), np.log10(K), e])


def make_dataset(n, regime="standard", seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = zip(*(make_star(rng, regime) for _ in range(n)))
    return np.stack(xs), np.stack(ys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--regime", default="standard", choices=["standard", "eccentric", "sparse"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    X, Y = make_dataset(a.n, a.regime, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, X=X, Y=Y)
    print(f"wrote {a.out}: X{X.shape} Y{Y.shape}")
