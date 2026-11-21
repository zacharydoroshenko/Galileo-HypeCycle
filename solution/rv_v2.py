"""rv_v2 = official baseline (below, unchanged where not noted) + team improvements, all switchable
via CONFIG (tuned on the TRAIN split only by train/train.py):
  - jitter_fit: per-star ML jitter, refit, candidates scored by full Gaussian NLL (baseline: fixed 2 m/s, chi2)
  - occam_weight: Laplace evidence term 0.5*logdet(J^T J) in the score ("posterior-based choice among peaks")
  - bound_penalty: penalise fits outside the P/K range seen in the train labels (P_bounds, K_bounds)
  - yearly_aliases: f +- 1/365.25 as extra candidates; n_peaks configurable
  - macOS-safe parallel predict (fork on Linux, spawn elsewhere), CONFIG shipped to workers
Original baseline docstring follows.

Classical baseline algorithm for rv_exoplanet: GLS periodogram + alias check + Keplerian least squares.

``framework: classic`` / ``model_format: py``: this file IS the model. It has no
weights and no randomness; ``run_baseline.py`` imports it and calls
:func:`predict`. Only numpy and scipy (both in the SIDERIUS environment) are used.

Per star (only slots with mask == 1 are used):

1. Weights w_i = 1 / (sigma_i^2 + JITTER_GUESS^2), a fixed guess of the
   unreported stellar jitter.
2. Generalised Lomb-Scargle periodogram (Zechmeister & Kuerster 2009, floating
   mean, weighted) on a grid uniform in frequency, FREQ_MIN to FREQ_MAX cycles
   per day in steps of FREQ_STEP (vectorised in numpy).
3. The N_PEAKS highest local maxima at least one peak width (1 / time span)
   apart in frequency.
4. Alias check (standard practice for night-time data, e.g. Dawson &
   Fabrycky 2010): every peak frequency f also contributes its 1-day alias
   |1 - f| as a candidate (when inside the grid and not already a candidate).
5. For each candidate, a coarse grid over (e, omega, M0) at the candidate
   period (12 circular + 4 x 8 x 12 eccentric = 396 points); for every grid
   point K and gamma are solved exactly by weighted linear least squares (the
   model is linear in K and gamma). The N_STARTS best grid points seed step 6.
6. scipy.optimize.least_squares (trust-region reflective) on all six
   Keplerian parameters (ln P, ln K, sqrt(e) cos omega, sqrt(e) sin omega,
   M0, gamma), the frequency bounded to +-FREQ_WINDOW peak widths around the
   candidate (a local refinement that cannot jump to another alias).
7. The fit with the smallest weighted chi-square over all candidates wins.
   Output [log10 P, log10 K, e] with e clipped to [0, E_MAX].

Determinism: fixed grids, no random numbers; every star is fitted
independently, so the result does not depend on N_WORKERS or chunking.
"""

from __future__ import annotations

import multiprocessing
import os

import numpy as np
from scipy.optimize import least_squares

FREQ_MIN = 1.0 / 1500.0
FREQ_MAX = 1.0 / 0.9
FREQ_STEP = 1.0e-4
N_PEAKS = 3
ONE_DAY_ALIASES = True
E_GRID = (0.0, 0.25, 0.5, 0.7, 0.85)
N_OMEGA = 8
N_PHASE = 12
N_STARTS = 3
FREQ_WINDOW = 1.5
JITTER_GUESS = 2.0
E_MAX = 0.95
KEPLER_ITERATIONS = 20
MAX_NFEV = 300
N_WORKERS = 2
CHUNK = 25

HYPERPARAMETERS = {
    "period_grid": "uniform in frequency", "freq_min_per_day": FREQ_MIN, "freq_max_per_day": FREQ_MAX,
    "freq_step_per_day": FREQ_STEP, "n_peaks": N_PEAKS, "peak_separation": "1 / time span (one peak width)",
    "one_day_alias_candidates": ONE_DAY_ALIASES, "e_grid": list(E_GRID), "n_omega": N_OMEGA,
    "n_phase": N_PHASE, "n_starts_per_candidate": N_STARTS, "freq_window_peak_widths": FREQ_WINDOW,
    "jitter_guess_mps": JITTER_GUESS, "e_max": E_MAX, "kepler_newton_iterations": KEPLER_ITERATIONS,
    "least_squares": "scipy trf, x_scale=jac", "max_nfev": MAX_NFEV, "n_workers": N_WORKERS, "chunk": CHUNK,
}

CONFIG = {
    "n_peaks": 3,
    "yearly_aliases": False,
    "jitter_fit": False,
    "n_rescore": 4,          # best fits (by chi2) that get jitter refit + rescoring
    "occam_weight": 0.0,
    "bound_penalty": 0.0,
    "P_bounds": [1.0, 1000.0],   # overwritten from train labels by train/train.py
    "K_bounds": [2.0, 100.0],
}

_FREQS = np.arange(FREQ_MIN, FREQ_MAX, FREQ_STEP)
_INPUTS: np.ndarray | None = None


def kepler_e(mean_anomaly: np.ndarray, ecc) -> np.ndarray:
    m = np.mod(mean_anomaly, 2.0 * np.pi)
    e = np.broadcast_to(ecc, m.shape)
    big_e = m + 0.85 * e * np.sign(np.sin(m))
    for _ in range(KEPLER_ITERATIONS):
        big_e = big_e - (big_e - e * np.sin(big_e) - m) / (1.0 - e * np.cos(big_e))
    return big_e


def shape(t, period, ecc, omega, m0):
    """cos(nu + omega) + e cos(omega): the Keplerian curve with K = 1, gamma = 0."""
    big_e = kepler_e(2.0 * np.pi * t / period + m0, ecc)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + ecc) * np.sin(big_e / 2.0), np.sqrt(1.0 - ecc) * np.cos(big_e / 2.0))
    return np.cos(nu + omega) + ecc * np.cos(omega)


def gls_power(t, y, w, freqs=_FREQS):
    """Generalised Lomb-Scargle power (ZK09 eq. 4-20) at frequencies in cycles/day; w sums to 1."""
    phase = (2.0 * np.pi * freqs)[:, None] * t[None, :]
    c = np.cos(phase)
    s = np.sin(phase)
    wy = w * y
    y_mean = wy.sum()
    yy = (wy * y).sum() - y_mean**2
    cm = c @ w
    sm = s @ w
    yc = c @ wy - y_mean * cm
    ys = s @ wy - y_mean * sm
    cc = (c * c) @ w - cm * cm
    ss = 1.0 - (c * c) @ w - sm * sm
    cs = (c * s) @ w - cm * sm
    d = cc * ss - cs * cs
    with np.errstate(divide="ignore", invalid="ignore"):
        power = (ss * yc * yc + cc * ys * ys - 2.0 * cs * yc * ys) / (yy * d)
    return np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)


def top_peaks(power, separation, freqs=_FREQS, n=N_PEAKS):
    """Frequencies of the n highest local maxima, pairwise more than `separation` apart."""
    interior = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    idx = np.nonzero(interior)[0] + 1
    idx = idx[np.argsort(-power[idx], kind="stable")]
    chosen: list[float] = []
    for i in idx:
        if all(abs(freqs[i] - f) > separation for f in chosen):
            chosen.append(float(freqs[i]))
        if len(chosen) == n:
            break
    if not chosen:
        chosen = [float(freqs[int(np.argmax(power))])]
    return chosen


def candidates(peaks, separation, yearly=False):
    """Peak frequencies plus their 1-day aliases |1 - f| (and optionally yearly f +- 1/365.25)."""
    out = list(peaks)
    alias_list = []
    for f in peaks:
        if ONE_DAY_ALIASES:
            alias_list.append(abs(1.0 - f))
        if yearly:
            alias_list += [f + 1.0 / 365.25, abs(f - 1.0 / 365.25)]
    for alias in alias_list:
        if FREQ_MIN <= alias <= FREQ_MAX and all(abs(alias - g) > separation for g in out):
            out.append(alias)
    return out


def _linear_k_gamma(f, y, w):
    """Weighted LS of y ~ K f + gamma for each row of f; returns K, gamma, chi2."""
    sw = w.sum()
    f_mean = (f * w).sum(-1) / sw
    y_mean = (y * w).sum() / sw
    df = f - f_mean[..., None]
    dy = y - y_mean
    var = (w * df * df).sum(-1)
    k = np.where(var > 0, (w * df * dy).sum(-1) / np.where(var > 0, var, 1.0), 0.0)
    gamma = y_mean - k * f_mean
    resid = y - (k[..., None] * f + gamma[..., None])
    return k, gamma, (w * resid * resid).sum(-1)


def _grid():
    es, oms, ms = [], [], []
    for e in E_GRID:
        omegas = [0.0] if e == 0.0 else list(np.arange(N_OMEGA) * 2.0 * np.pi / N_OMEGA)
        for om in omegas:
            for m0 in np.arange(N_PHASE) * 2.0 * np.pi / N_PHASE:
                es.append(e)
                oms.append(om)
                ms.append(m0)
    return np.array(es), np.array(oms), np.array(ms)


_GRID_E, _GRID_OMEGA, _GRID_M0 = _grid()  # 396 points


def _starts_for_period(t, y, w, period):
    es, oms, ms = _GRID_E, _GRID_OMEGA, _GRID_M0
    f = shape(t[None, :], period, es[:, None], oms[:, None], ms[:, None])
    k, gamma, chi2 = _linear_k_gamma(f, y, w)
    best = np.argsort(chi2, kind="stable")[:N_STARTS]
    starts = []
    for i in best:
        kk, om = k[i], oms[i]
        if kk < 0:
            kk, om = -kk, om + np.pi
        kk = max(kk, 1e-2)
        sq = np.sqrt(es[i])
        starts.append(np.array([np.log(period), np.log(kk), sq * np.cos(om), sq * np.sin(om), ms[i], gamma[i]]))
    return starts


def _unpack(x):
    lnp, lnk, h, k, m0, gamma = x
    e = min(h * h + k * k, E_MAX)
    omega = np.arctan2(k, h)
    return np.exp(lnp), np.exp(lnk), e, omega, m0, gamma


def _model(x, t):
    period, k_amp, e, omega, m0, gamma = _unpack(x)
    return k_amp * shape(t, period, e, omega, m0) + gamma


def refine(t, y, sig_eff, x0, f_lo, f_hi):
    """Local Keplerian least squares from x0 with the frequency kept in [f_lo, f_hi]; (cost, x) or None."""
    lower = np.array([-np.log(f_hi), np.log(1e-3), -0.98, -0.98, -np.inf, -np.inf])
    upper = np.array([-np.log(f_lo), np.log(1e4), 0.98, 0.98, np.inf, np.inf])
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)
    try:
        res = least_squares(
            lambda x: (y - _model(x, t)) / sig_eff, x0, bounds=(lower, upper),
            method="trf", x_scale="jac", max_nfev=MAX_NFEV,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    return (float(res.cost), res.x, res.jac) if np.isfinite(res.cost) else None


def _ml_jitter(r, sig):
    """Maximum-likelihood jitter s >= 0 for residuals r with reported errors sig (1-D, bounded)."""
    from scipy.optimize import minimize_scalar

    def nll(s_):
        v = sig**2 + s_**2
        return 0.5 * np.sum(r * r / v + np.log(v))

    res = minimize_scalar(nll, bounds=(0.0, 20.0), method="bounded", options={"xatol": 1e-3})
    return float(res.x), float(res.fun)


def prepare(x_star: np.ndarray):
    """(t centred on the median epoch, v, effective sigma, normalised weights, time span) for one star."""
    mask = x_star[3] > 0.5
    t = x_star[0, mask].astype(np.float64)
    y = x_star[1, mask].astype(np.float64)
    sig = x_star[2, mask].astype(np.float64)
    span = max(float(t.max() - t.min()), 1.0)
    t = t - np.median(t)  # reference time; M0 refers to the median epoch
    sig_eff = np.sqrt(sig**2 + JITTER_GUESS**2)
    w = 1.0 / sig_eff**2
    return t, y, sig_eff, w, span


def fit_star(x_star: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """[log10 P, log10 K, e] for one star given its [4, 60] input."""
    cfg = CONFIG if cfg is None else cfg
    t, y, sig_eff, w, span = prepare(x_star)
    sig = np.sqrt(np.maximum(sig_eff**2 - JITTER_GUESS**2, 1e-12))
    width = 1.0 / span
    peaks = top_peaks(gls_power(t, y, w / w.sum()), separation=width, n=int(cfg["n_peaks"]))
    fits = []  # (chi2 cost, x, jac, f_lo, f_hi)
    for freq in candidates(peaks, separation=width, yearly=bool(cfg["yearly_aliases"])):
        f_lo = max(freq - FREQ_WINDOW * width, 0.5 * FREQ_MIN)
        f_hi = freq + FREQ_WINDOW * width
        for x0 in _starts_for_period(t, y, w, 1.0 / freq):
            fit = refine(t, y, sig_eff, x0, f_lo, f_hi)
            if fit is not None:
                fits.append((fit[0], fit[1], fit[2], f_lo, f_hi))
    if not fits:
        return np.array([-np.log10(peaks[0]), 0.0, 0.0])

    plain = not (cfg["jitter_fit"] or cfg["occam_weight"] or cfg["bound_penalty"])
    if plain:  # exactly the baseline decision rule
        best_x = min(fits, key=lambda z: z[0])[1]
    else:
        fits.sort(key=lambda z: z[0])
        lp_lo, lp_hi = np.log(cfg["P_bounds"][0]), np.log(cfg["P_bounds"][1])
        lk_lo, lk_hi = np.log(cfg["K_bounds"][0]), np.log(cfg["K_bounds"][1])
        best_score, best_x = np.inf, None
        for cost, x, jac, f_lo, f_hi in fits[: int(cfg["n_rescore"])]:
            if cfg["jitter_fit"]:
                for _ in range(2):  # alternate: ML jitter given orbit, orbit given jitter
                    s_j, _ = _ml_jitter(y - _model(x, t), sig)
                    se = np.sqrt(sig**2 + s_j**2)
                    fit = refine(t, y, se, x, f_lo, f_hi)
                    if fit is None:
                        break
                    cost, x, jac = fit
                s_j, nll = _ml_jitter(y - _model(x, t), sig)
            else:
                nll = cost  # chi2/2 at the fixed jitter guess
            score = nll
            if cfg["occam_weight"]:
                sign, logdet = np.linalg.slogdet(jac.T @ jac + 1e-9 * np.eye(jac.shape[1]))
                if sign > 0:
                    score += cfg["occam_weight"] * 0.5 * logdet
            if cfg["bound_penalty"]:
                lnp, lnk = x[0], x[1]
                out = max(0.0, lp_lo - lnp, lnp - lp_hi) + max(0.0, lk_lo - lnk, lnk - lk_hi)
                score += cfg["bound_penalty"] * out
            if score < best_score:
                best_score, best_x = score, x
    period, k_amp, e, _, _, _ = _unpack(best_x)
    return np.array([np.log10(period), np.log10(k_amp), float(np.clip(e, 0.0, E_MAX))])


def _fit_one(arg):
    x, cfg = arg
    try:
        out = fit_star(x, cfg)
        if np.all(np.isfinite(out)):
            return out
    except Exception:
        pass
    return np.array([1.0, 1.0, 0.1])


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def predict(inputs: np.ndarray, n_workers: int = N_WORKERS) -> np.ndarray:
    """Predictions float64 [N, 3] for inputs float32 [N, 4, 60]."""
    import sys
    X = np.asarray(inputs)
    args = [(x, dict(CONFIG)) for x in X]
    out = None
    if n_workers and n_workers > 1 and len(X) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            if sys.platform.startswith("linux"):
                mod = sys.modules.get(__name__)
                if mod is None or getattr(mod, "_fit_one", None) is not _fit_one:
                    import types
                    mod = types.ModuleType(__name__)
                    mod.__dict__.update(globals())
                    sys.modules[__name__] = mod
                ctx, fn, init = multiprocessing.get_context("fork"), _fit_one, _worker_init
            else:
                import importlib
                here = os.path.dirname(os.path.abspath(__file__))
                if here not in sys.path:
                    sys.path.insert(0, here)
                _m = importlib.import_module(os.path.splitext(os.path.basename(__file__))[0])
                fn, init = _m._fit_one, _m._worker_init
                ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=ctx, initializer=init) as ex:
                out = list(ex.map(fn, args, chunksize=max(1, len(X) // (8 * int(n_workers)))))
        except Exception as exc:  # never fail scoring; but say so
            import sys as _sys
            print(f"[rv] parallel predict failed ({exc!r}); running serially", file=_sys.stderr)
            out = None
    if out is None:
        out = [_fit_one(a) for a in args]
    out = np.stack(out).astype(np.float64)
    out[:, 2] = np.clip(out[:, 2], 0.0, E_MAX)
    return out


if __name__ == "__main__":  # quick self-test on a synthetic circular orbit sampled at night
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rng = np.random.default_rng(0)
    tt = np.sort(rng.choice(np.arange(800), 40, replace=False) + rng.normal(0, 0.08, 40))
    vv = 20 * np.cos(2 * np.pi * tt / 37.0 + 1.0) + rng.normal(0, 2, 40)
    xs = np.zeros((4, 60), np.float32)
    xs[0, :40], xs[1, :40], xs[2, :40], xs[3, :40] = tt - tt[0], vv, 2.0, 1.0
    print(10 ** fit_star(xs)[:2])
