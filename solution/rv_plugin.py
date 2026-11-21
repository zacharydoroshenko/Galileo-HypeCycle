"""Self-contained RV orbit-recovery algorithm plugin (Galileo hackathon, exoplanet-orbits).

Contract: predict(X, n_workers=14) with X of shape [N, 4, 60] -> [N, 3] = [log10 P, log10 K, e].
Only numpy + scipy (available in the SIDERIUS sandbox). No network, no extra files.

Pipeline per star:
  1. parse the 4 channels (time, rv, err, mask/padding)          -> CHANNELS / parse_star
  2. generalised Lomb-Scargle on a log-uniform-ish frequency grid  -> gls
  3. take top peaks + their daily/yearly aliases + harmonics      -> candidate_periods
  4. for each candidate: dense grid over (P, e, mean anomaly) with the linear
     parameters (K cos w, K sin w, gamma) solved exactly          -> grid_fit
  5. nonlinear polish with jitter in the likelihood               -> polish
  6. choose the candidate with the best penalised likelihood

!!! CONFIG below is overwritten by train/train.py (hyperparameter tuning on the TRAIN split).
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------- config
CONFIG = {
    # channel layout of X[:, c, :]. VERIFY against TASK_CARD.md when the data arrives.
    "ch_t": 0,
    "ch_rv": 1,
    "ch_err": 2,
    "ch_mask": 3,          # set to -1 if there is no mask channel
    "time_scale": 1.0,     # multiply times by this to get days
    # period search range (days)
    "p_min": 1.0,
    "p_max_factor": 3.0,   # p_max = factor * baseline
    "p_max_abs": 1.0e5,
    "oversample": 10.0,
    "n_peaks": 6,
    "add_aliases": True,
    "add_harmonics": True,
    # grid fit
    "n_e": 10,
    "e_max": 0.95,
    "n_phase": 36,
    "p_local": 7,          # local period refinements per candidate
    "p_local_width": 0.5,  # +/- fraction of a peak width (1/baseline) in frequency
    "n_polish": 3,         # candidates kept for nonlinear polish
    # Keplerian periodogram (finds eccentric planets that GLS misses)
    "kep_on": True,
    "kep_oversample": 3.0,
    "kep_e": (0.5, 0.75, 0.9),
    "kep_phase": 16,
    "kep_peaks": 5,
    # (log10 P, log10 K) prior learned from TRAIN labels by train/train.py; None = flat.
    # Deliberately no prior on e: the test set is more eccentric than training.
    "prior_weight": 0.0,
    "prior_lp_edges": None,
    "prior_lk_edges": None,
    "prior_logpdf": None,
    # model selection
    "e_penalty": 0.0,      # added to neg-log-likelihood * e^2 (shrinks spurious high e)
    "jitter_init": 3.0,    # m/s
}

# ----------------------------------------------------------------------------- parsing

def parse_star(x: np.ndarray, cfg: dict = CONFIG):
    """x: [4, L] -> (t, y, s) arrays of valid points, t in days, sorted."""
    t = np.asarray(x[cfg["ch_t"]], float) * cfg["time_scale"]
    y = np.asarray(x[cfg["ch_rv"]], float)
    s = np.asarray(x[cfg["ch_err"]], float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    if cfg["ch_mask"] >= 0:
        m = np.asarray(x[cfg["ch_mask"]], float)
        ok &= np.isfinite(m) & (m > 0.5)
    t, y, s = t[ok], y[ok], s[ok]
    o = np.argsort(t)
    return t[o], y[o], s[o]


# ----------------------------------------------------------------------------- physics

def kepler_E(M: np.ndarray, e: np.ndarray, n_iter: int = 12) -> np.ndarray:
    """Solve E - e sin E = M (vectorised, broadcasting). Newton with Danby start."""
    M = np.mod(M, 2 * np.pi)
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(n_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E = E - f / fp
    return E


def true_anomaly(M, e):
    E = kepler_E(M, e)
    return 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


_NU_TABLE = {}
_NU_N = 8192


def true_anomaly_table(M, e_values):
    """Fast nu(M, e) for a fixed set of e by linear interpolation in a cached table.
    M: [G, n], e_values: [G] (each row uses its own e). Coarse-grid use only."""
    key = tuple(np.round(np.unique(e_values), 10))
    tab = _NU_TABLE.get(key)
    if tab is None:
        grid = np.linspace(0, 2 * np.pi, _NU_N + 1)
        es = np.array(key)
        nu = np.unwrap(np.mod(true_anomaly(grid[None, :], es[:, None]), 2 * np.pi), axis=1)
        tab = (es, nu)
        _NU_TABLE[key] = tab
    es, nu = tab
    ie = np.searchsorted(es, np.round(e_values, 10))
    u = np.mod(M, 2 * np.pi) * (_NU_N / (2 * np.pi))
    i0 = np.minimum(u.astype(np.int64), _NU_N - 1)
    fr = u - i0
    rows = nu[ie]                                     # [G, N+1]
    a = np.take_along_axis(rows, i0, 1)
    b = np.take_along_axis(rows, i0 + 1, 1)
    return a + fr * (b - a)


def rv_model(t, P, K, e, w, M0, gamma):
    nu = true_anomaly(2 * np.pi * t / P + M0, e)
    return K * (np.cos(nu + w) + e * np.cos(w)) + gamma


# ----------------------------------------------------------------------------- periodogram

def gls(t, y, s, freqs):
    """Generalised Lomb-Scargle (Zechmeister & Kurster 2009). Returns power in [0, 1]."""
    w = 1.0 / s**2
    w = w / w.sum()
    yb = np.sum(w * y)
    yy = np.sum(w * (y - yb) ** 2) + 1e-300
    power = np.empty(len(freqs))
    chunk = 4096
    for i in range(0, len(freqs), chunk):
        f = freqs[i:i + chunk, None]
        arg = 2 * np.pi * f * t[None, :]
        c, sn = np.cos(arg), np.sin(arg)
        C = c @ w
        S = sn @ w
        wy = w * (y - yb)
        YC = c @ wy
        YS = sn @ wy
        CC = (c * c) @ w - C * C
        SS = (sn * sn) @ w - S * S
        CS = (c * sn) @ w - C * S
        D = CC * SS - CS**2 + 1e-300
        power[i:i + chunk] = (SS * YC**2 + CC * YS**2 - 2 * CS * YC * YS) / (yy * D)
    return power


def freq_grid(t, cfg):
    T = max(t.max() - t.min(), 1.0)
    p_max = min(cfg["p_max_factor"] * T, cfg["p_max_abs"])
    fmin, fmax = 1.0 / p_max, 1.0 / cfg["p_min"]
    df = 1.0 / (cfg["oversample"] * T)
    n = int(min((fmax - fmin) / df, 200_000))
    return np.linspace(fmin, fmax, max(n, 100)), T


def kepler_periodogram(t, y, s, cfg, jitter):
    """Coarse chi2 scan over the full frequency range with eccentric templates.
    Returns the frequencies of the best (separated) minima."""
    T = max(t.max() - t.min(), 1.0)
    p_max = min(cfg["p_max_factor"] * T, cfg["p_max_abs"])
    fmin, fmax = 1.0 / p_max, 1.0 / cfg["p_min"]
    n = int(min((fmax - fmin) * T * cfg["kep_oversample"], 60_000))
    freqs = np.linspace(fmin, fmax, max(n, 50))
    es = np.asarray(cfg["kep_e"], float)
    M0s = np.linspace(0, 2 * np.pi, cfg["kep_phase"], endpoint=False)
    EE, MM = np.meshgrid(es, M0s, indexing="ij")
    EE, MM = EE.ravel(), MM.ravel()                    # [G]
    wts = 1.0 / (s**2 + jitter**2)
    best = np.empty(len(freqs))
    chunk = max(1, 20000 // len(EE))
    for i in range(0, len(freqs), chunk):
        f = freqs[i:i + chunk]
        M = (2 * np.pi * f[:, None, None] * t[None, None, :] + MM[None, :, None])  # [F, G, n]
        F, G, nn = M.shape
        Mf = M.reshape(F * G, nn)
        ef = np.tile(EE, F)
        nu = true_anomaly_table(Mf, ef)
        _, chi2 = _linear_solve(nu, ef[:, None], y[None, :], wts[None, :])
        best[i:i + chunk] = chi2.reshape(F, G).min(1)
    lm = np.where((best[1:-1] < best[:-2]) & (best[1:-1] <= best[2:]))[0] + 1
    lm = lm[np.argsort(best[lm])]
    out = []
    for j in lm:
        if all(abs(freqs[j] - g) > 1.0 / T for g in out):
            out.append(freqs[j])
        if len(out) >= cfg["kep_peaks"]:
            break
    return out


def candidate_periods(t, y, s, cfg):
    freqs, T = freq_grid(t, cfg)
    pw = gls(t, y, s, freqs)
    # local maxima
    lm = np.where((pw[1:-1] > pw[:-2]) & (pw[1:-1] >= pw[2:]))[0] + 1
    lm = lm[np.argsort(pw[lm])[::-1]]
    peaks = []
    for i in lm:  # keep peaks separated by > one resolution element
        if all(abs(freqs[i] - freqs[j]) > 1.0 / T for j in peaks):
            peaks.append(i)
        if len(peaks) >= cfg["n_peaks"]:
            break
    fs = [freqs[i] for i in peaks]
    out = list(fs)
    if cfg["add_aliases"]:
        for f in fs[:3]:
            for fa in (1.0, 1.0 / 365.25, 1.0 / 29.53):
                out += [abs(f - fa), f + fa]
    if cfg["add_harmonics"]:
        for f in fs[:2]:
            out += [f / 2, f / 3, 2 * f]
    if cfg.get("kep_on", False):
        out += kepler_periodogram(t, y, s, cfg, cfg["jitter_init"])
    fmin, fmax = freqs[0], freqs[-1]
    out = [f for f in out if fmin * 0.5 <= f <= fmax * 1.05 and f > 0]
    # dedupe in frequency
    out = sorted(set(np.round(np.array(out) * T * cfg["oversample"]) / (T * cfg["oversample"])))
    return np.array([1.0 / f for f in out if f > 0]), T


# ----------------------------------------------------------------------------- fitting

def _linear_solve(nu, e, y, w):
    """For model y ~ c1 (cos nu + e) + c2 sin nu + c3, weighted LS, batched over leading dims.
    nu: [..., n], e broadcastable to [..., 1]. Returns coeffs [..., 3] and chi2 [...]."""
    a1 = np.cos(nu) + e
    a2 = np.sin(nu)
    a3 = np.ones_like(nu)
    A = np.stack([a1, a2, a3], -1)                 # [..., n, 3]
    Aw = A * w[..., :, None]
    ATA = np.einsum("...ni,...nj->...ij", Aw, A) + 1e-9 * np.eye(3)
    ATy = np.einsum("...ni,...n->...i", Aw, y)
    c = np.linalg.solve(ATA, ATy[..., None])[..., 0]
    r = y - np.einsum("...ni,...i->...n", A, c)
    chi2 = np.sum(w * r * r, -1)
    return c, chi2


def _coeffs_to_params(c, e):
    c1, c2, c3 = c[..., 0], c[..., 1], c[..., 2]
    K = np.hypot(c1, c2)
    w = np.arctan2(-c2, c1)
    return K, w, c3


def grid_fit(t, y, s, P0, T, cfg, jitter):
    """Dense grid around period P0 over (P, e, M0). Returns best (P, e, M0, chi2)."""
    f0 = 1.0 / P0
    dfs = np.linspace(-cfg["p_local_width"], cfg["p_local_width"], cfg["p_local"]) / T
    Ps = 1.0 / np.clip(f0 + dfs, 1e-8, None)
    es = np.linspace(0.0, cfg["e_max"], cfg["n_e"]) ** 1.5 / cfg["e_max"] ** 0.5  # denser at low e
    M0s = np.linspace(0, 2 * np.pi, cfg["n_phase"], endpoint=False)
    wts = 1.0 / (s**2 + jitter**2)
    PP, EE, MM = np.meshgrid(Ps, es, M0s, indexing="ij")
    PP, EE, MM = PP.ravel(), EE.ravel(), MM.ravel()
    M = 2 * np.pi * t[None, :] / PP[:, None] + MM[:, None]
    nu = true_anomaly_table(M, EE)
    _, chi2 = _linear_solve(nu, EE[:, None], y[None, :], wts[None, :])
    k = int(np.argmin(chi2))
    return PP[k], EE[k], MM[k], chi2[k]


def nll(t, y, s, P, e, M0, jitter):
    var = s**2 + jitter**2
    nu = true_anomaly(2 * np.pi * t / P + M0, e)
    c, _ = _linear_solve(nu, e, y, 1.0 / var)
    K, w, g = _coeffs_to_params(c, e)
    r = y - (K * (np.cos(nu + w) + e * np.cos(w)) + g)
    return 0.5 * np.sum(r * r / var + np.log(var)), K


def polish(t, y, s, P, e, M0, jitter, cfg):
    from scipy.optimize import minimize

    def unpack(th):
        return (math.exp(th[0]), 1.0 / (1.0 + math.exp(-th[1])) * 0.99, th[2], math.exp(th[3]))

    e = float(np.clip(e, 1e-3, 0.98))
    th0 = np.array([math.log(P), math.log(e / (0.99 - e)), M0, math.log(max(jitter, 0.05))])

    def obj(th):
        P_, e_, M_, j_ = unpack(th)
        v, _ = nll(t, y, s, P_, e_, M_, j_)
        return v + cfg["e_penalty"] * e_**2

    res = minimize(obj, th0, method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-7, "fatol": 1e-6})
    P_, e_, M_, j_ = unpack(res.x)
    v, K = nll(t, y, s, P_, e_, M_, j_)
    return dict(P=P_, e=e_, M0=M_, jitter=j_, K=float(K), score=v + cfg["e_penalty"] * e_**2)


def log_prior(P, K, cfg):
    """Smoothed 2-D histogram log-density of (log10 P, log10 K); outside the range -> lowest bin."""
    if cfg.get("prior_logpdf") is None or cfg.get("prior_weight", 0.0) == 0.0:
        return 0.0
    lp_e = np.asarray(cfg["prior_lp_edges"])
    lk_e = np.asarray(cfg["prior_lk_edges"])
    tab = np.asarray(cfg["prior_logpdf"])
    i = int(np.clip(np.searchsorted(lp_e, math.log10(max(P, 1e-9))) - 1, 0, tab.shape[0] - 1))
    j = int(np.clip(np.searchsorted(lk_e, math.log10(max(K, 1e-9))) - 1, 0, tab.shape[1] - 1))
    return float(tab[i, j])


def fit_star(x, cfg=CONFIG):
    t, y, s = parse_star(x, cfg)
    if len(t) < 4:
        return np.array([1.0, 1.0, 0.1]), {}
    t = t - t.min()
    cands, T = candidate_periods(t, y, s, cfg)
    jit = cfg["jitter_init"]
    coarse = []
    for P0 in cands:
        P, e, M0, chi2 = grid_fit(t, y, s, P0, T, cfg, jit)
        coarse.append((chi2, P, e, M0))
    coarse.sort(key=lambda z: z[0])
    best = None
    w = cfg.get("prior_weight", 0.0)
    for _, P, e, M0 in coarse[: cfg["n_polish"]]:
        r = polish(t, y, s, P, e, M0, jit, cfg)
        r["score"] -= w * log_prior(r["P"], r["K"], cfg)
        if best is None or r["score"] < best["score"]:
            best = r
    out = np.array([math.log10(best["P"]), math.log10(max(best["K"], 1e-3)),
                    float(np.clip(best["e"], 0.0, 0.99))])
    if not np.all(np.isfinite(out)):
        out = np.array([1.0, 1.0, 0.1])
    return out, best


def _fit_one(arg):
    x, cfg = arg
    try:
        return fit_star(x, cfg)[0]
    except Exception:  # never let one star kill the batch
        return np.array([1.0, 1.0, 0.1])


def _worker_init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _parallel(X, n_workers):
    """Process pool. Linux (the scoring sandbox): 'fork'; the evaluator may load this file via
    importlib without registering it in sys.modules, so we register it for pickling.
    macOS: fork crashes Accelerate-linked children, so use 'spawn' and import this file by its
    stem from its own directory. CONFIG is shipped with every task, so tuned values reach
    workers either way. Any failure -> caller falls back to serial."""
    import multiprocessing as mp
    import sys

    args = [(x, dict(CONFIG)) for x in X]
    chunk = max(1, len(X) // (8 * n_workers))
    if sys.platform.startswith("linux"):
        mod = sys.modules.get(__name__)
        if mod is None or getattr(mod, "_fit_one", None) is not _fit_one:
            import types
            mod = types.ModuleType(__name__)
            mod.__dict__.update(globals())
            sys.modules[__name__] = mod
        ctx, fn, init = mp.get_context("fork"), _fit_one, _worker_init
    else:
        import importlib
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        _m = importlib.import_module(os.path.splitext(os.path.basename(__file__))[0])
        fn, init = _m._fit_one, _m._worker_init
        ctx = mp.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor  # raises (no hang) if workers die
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=init) as ex:
        return list(ex.map(fn, args, chunksize=chunk))


def predict(X, n_workers: int = 1):
    X = np.asarray(X, dtype=float)
    out = None
    if n_workers and n_workers > 1 and len(X) > 1:
        try:
            out = _parallel(X, int(n_workers))
        except Exception as exc:  # never fail scoring; but say so
            import sys as _sys
            print(f"[rv] parallel predict failed ({exc!r}); running serially", file=_sys.stderr)
            out = None
    if out is None:
        out = [_fit_one((x, CONFIG)) for x in X]
    out = np.stack(out).astype(np.float64)
    # validity guard: scorer rejects non-finite / all-identical predictions
    out[~np.isfinite(out)] = 0.5
    return out
