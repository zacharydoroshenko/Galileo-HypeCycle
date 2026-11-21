"""GPU dense-posterior period choice (The-Joker-style), TRAIN-only diagnostic. Organizer hint 2: use the H100.

Per star, on the GPU: frequency grid (log-uniform P prior => weight 1/f) x e x omega x M0; K, gamma marginalised
analytically (flat priors); marginal likelihood summed over the shape parameters gives p(f | data).
Period choice = the posterior peak (+-1 resolution element) with the largest mass.
Compared with submission 16's period on the same stars: P-correct rate (|dP|/P < 1%) and paired gains.

    python tools/gpu_joker.py --set fresh_std2000 --n 1000      (on the GPU box)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "solution"), str(ROOT / "tools")]

E_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
N_OMEGA, N_PHASE, OVERSAMPLE = 8, 16, 5.0


def shape_grid(dev):
    es, oms, ms = [], [], []
    for e in E_VALUES:
        omegas = [0.0] if e == 0.0 else [2 * math.pi * k / N_OMEGA for k in range(N_OMEGA)]
        for om in omegas:
            for k in range(N_PHASE):
                es.append(e); oms.append(om); ms.append(2 * math.pi * k / N_PHASE)
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
    return t(es), t(oms), t(ms)


def kepler_shape(t, f, e, om, m0, iters=15):
    """cos(nu+om) + e cos(om) for t [n], f [F,1,1], e/om/m0 [1,G,1] -> [F,G,n]."""
    M = torch.remainder(2 * math.pi * f * t + m0, 2 * math.pi)
    E = M + 0.85 * e * torch.sign(torch.sin(M))
    for _ in range(iters):
        E = E - (E - e * torch.sin(E) - M) / (1 - e * torch.cos(E))
    nu = 2 * torch.atan2(torch.sqrt(1 + e) * torch.sin(E / 2), torch.sqrt(1 - e) * torch.cos(E / 2))
    return torch.cos(nu + om) + e * torch.cos(om)


@torch.no_grad()
def log_post_freq(t, y, w, freqs, grid, chunk=256):
    es, oms, ms = grid
    e, om, m0 = es.view(1, -1, 1), oms.view(1, -1, 1), ms.view(1, -1, 1)
    sw, yw = w.sum(), (w * y).sum()
    yy = (w * y * y).sum()
    out = []
    for i in range(0, len(freqs), chunk):
        f = freqs[i:i + chunk].view(-1, 1, 1)
        s = kepler_shape(t, f, e, om, m0)                       # [F,G,n]
        a11 = (w * s * s).sum(-1); a12 = (w * s).sum(-1); b1 = (w * s * y).sum(-1)
        det = torch.clamp(a11 * sw - a12 * a12, min=1e-20)
        # weighted LS of y ~ K s + gamma: chi2_min = yy - [b1 yw] A^-1 [b1 yw]^T
        q = (sw * b1 * b1 - 2 * a12 * b1 * yw + a11 * yw * yw) / det
        chi2 = yy - q
        # K marginalised under a PROPER prior: posterior width sqrt(sw/det) capped at the train K range (100 m/s);
        # a flat improper prior makes near-degenerate shapes (P >> span) win through -0.5 log det -> +inf.
        logwidth = torch.clamp(0.5 * (torch.log(sw) - torch.log(det)), max=math.log(100.0))
        terms = -0.5 * chi2 + logwidth
        out.append(torch.logsumexp(terms, dim=1) - math.log(terms.shape[1]))
    return torch.cat(out) - torch.log(freqs)                     # log-uniform P prior: p(f) ~ 1/f


def choose_period(t, y, sig, dev, grid):
    span = float(t.max() - t.min())
    df = 1.0 / (OVERSAMPLE * max(span, 1.0))
    freqs = torch.arange(1 / 1000.0, 1 / 1.0, df, device=dev, dtype=torch.float32)
    tt = torch.tensor(t - np.median(t), dtype=torch.float32, device=dev)
    yy = torch.tensor(y, dtype=torch.float32, device=dev)
    w = torch.tensor(1.0 / (sig**2 + 4.0), dtype=torch.float32, device=dev)
    lp = log_post_freq(tt, yy, w, freqs, grid)
    # peak masses: posterior mass within +-1 resolution element (1/span) of each grid point
    p = torch.exp(lp - lp.max())
    k = max(1, int(round(OVERSAMPLE)))
    mass = torch.nn.functional.conv1d(p.view(1, 1, -1), torch.ones(1, 1, 2 * k + 1, device=dev), padding=k).view(-1)
    best = int(torch.argmax(mass))
    imap = int(torch.argmax(lp))
    return 1.0 / float(freqs[best]), float(mass[best] / mass.sum()), 1.0 / float(freqs[imap])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fresh_std2000")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    import rv_v2
    from k_calib_aug import sign_p
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(ROOT / f"data/exo/{a.set}.npz")
    X, Y = d["X"][: a.n], d["Y"][: a.n]
    rv_v2.CONFIG.update(occam_weight=0.5, k_debias_c=6.95)
    t0 = time.time()
    pred = rv_v2.predict(X, n_workers=a.workers)                 # submission 16's choice
    t_cpu = time.time() - t0
    grid = shape_grid(dev)
    Pt = 10 ** Y[:, 0]
    okF = np.abs(10 ** pred[:, 0] - Pt) / Pt < 0.01
    okG = np.zeros(len(X), bool)
    okM = np.zeros(len(X), bool)
    t0 = time.time()
    for i, x in enumerate(X):
        m = x[3] > 0.5
        Pg, _, Pm = choose_period(x[0, m].astype(np.float64), x[1, m].astype(np.float64), x[2, m].astype(np.float64), dev, grid)
        # grid P is only accurate to the grid spacing; count it right if it lands in the true peak (+-1 resolution element)
        span = float(x[0, m].max() - x[0, m].min())
        okG[i] = abs(1 / Pg - 1 / Pt[i]) * span < 1.0
        okM[i] = abs(1 / Pm - 1 / Pt[i]) * span < 1.0
    t_gpu = time.time() - t0
    okF_peak = np.array([abs(10 ** -pred[i, 0] - 1 / Pt[i]) * float(X[i, 0][X[i, 3] > 0.5].max()) < 1.0 for i in range(len(X))])
    g, l = int((okG & ~okF_peak).sum()), int((~okG & okF_peak).sum())
    gm, lm = int((okM & ~okF_peak).sum()), int((~okM & okF_peak).sum())
    print(f"MAP frequency: right peak {okM.mean():.4f}  vs sub16 (+{gm}/-{lm}, p={sign_p(gm, lm):.3f})")
    print(f"{a.set} n={len(X)} | right PEAK: submission16 {okF_peak.mean():.4f}  GPU posterior-mass {okG.mean():.4f}"
          f"  (+{g}/-{l}, p={sign_p(g, l):.3f}) | sub16 P within 1%: {okF.mean():.4f}"
          f" | GPU {t_gpu / len(X) * 1000:.0f} ms/star, CPU pipeline {t_cpu:.0f}s")
