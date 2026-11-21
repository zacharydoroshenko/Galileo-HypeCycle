"""Report figures from the official analysis/headroom.py outputs (validation = held-out check).

    .venv/bin/python tools/make_figures.py [--baseline results/headroom_baseline.json]
                                           [--final results/headroom_final.json] [--out report/figs]

Writes:
  report/figs/success_vs_ceiling.pdf  success_rate by group: official baseline vs final, with the Fisher ceiling
  report/figs/failure_breakdown.pdf   failure counts by official failure class: official baseline vs final
Prints every plotted number so it can be checked against the JSON / the report tables.

Note: headroom.py names the analysed model's score `baseline_success` whatever model it was run on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GROUPS = [("all", "Overall"), ("in_distribution", "In-dist."), ("high_ecc", "High-e"), ("sparse", "Sparse")]
CLASSES = [("period_ok_k_off", "P right,\nK off"), ("yearly_alias", "Yearly\nalias"),
           ("one_day_alias", "1-day\nalias"), ("harmonic", "Harmonic"), ("other", "Other\n(no peak)")]

# Reference palette (light): categorical slots 1-2, ink and chrome tokens.
C_BASE, C_FINAL = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def success_vs_ceiling(base, final, out):
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(GROUPS))
    w = 0.34
    print("success_vs_ceiling: group  baseline  final  fisher  final/fisher")
    for i, (g, _) in enumerate(GROUPS):
        b, f, c = base[g]["baseline_success"], final[g]["baseline_success"], final[g]["fisher_ceiling"]
        print(f"  {g:16s} {b:.4f}  {f:.4f}  {c:.4f}  {f / c:.1%}")
        for dx, v, col in ((-w / 2, b, C_BASE), (w / 2, f, C_FINAL)):
            ax.bar(i + dx, v, w - 0.03, color=col, linewidth=0, zorder=2)
            ax.text(i + dx, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=5.8, color=INK_2)
        # Fisher (Cramer-Rao) ceiling: a tick spanning both bars.
        ax.plot([i - w - 0.02, i + w + 0.02], [c, c], color=INK, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=3)
        ax.text(i + w + 0.04, c, f"{c:.3f}", ha="left", va="center", fontsize=5.8, color=INK)
    ax.set_xticks(x, [lab for _, lab in GROUPS])
    ax.set_ylabel("success_rate (validation)")
    ax.set_ylim(0, 0.75)
    ax.set_xlim(-0.6, len(GROUPS) - 0.25)
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE), plt.Rectangle((0, 0), 1, 1, color=C_FINAL),
               plt.Line2D([], [], color=INK, linewidth=1.0, linestyle=(0, (3, 1.5)))]
    ax.legend(handles, ["Official baseline", "Final (Occam + K-debias)", "Fisher ceiling"],
              loc="upper right", frameon=False, ncol=1, handlelength=1.6, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def failure_breakdown(base, final, out, group="all"):
    def counts(d):
        n = d[group]["n_fail"]
        return np.array([round(n * d[group]["fail_share"][k]) for k, _ in CLASSES])

    cb, cf = counts(base), counts(final)
    print(f"failure_breakdown ({group}): class  baseline  final  change")
    for (k, _), b, f in zip(CLASSES, cb, cf):
        print(f"  {k:16s} {b:5d}  {f:5d}  {f - b:+d}")
    print(f"  {'total':16s} {base[group]['n_fail']:5d}  {final[group]['n_fail']:5d}")

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(CLASSES))
    w = 0.34
    for dx, v, col in ((-w / 2, cb, C_BASE), (w / 2, cf, C_FINAL)):
        ax.bar(x + dx, v, w - 0.03, color=col, linewidth=0, zorder=2)
        for xi, vi in zip(x + dx, v):
            ax.text(xi, vi + 12, f"{vi}", ha="center", va="bottom", fontsize=5.8, color=INK_2)
    ax.set_xticks(x, [lab for _, lab in CLASSES])
    n = base[group]["n"]
    ax.set_ylabel(f"failed stars (of {n:,}, validation)")
    ax.set_ylim(0, max(cb.max(), cf.max()) * 1.15)
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE), plt.Rectangle((0, 0), 1, 1, color=C_FINAL)]
    ax.legend(handles, [f"Official baseline ({base[group]['n_fail']:,} failures)",
                        f"Final ({final[group]['n_fail']:,} failures)"],
              loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / "results/headroom_baseline.json")
    ap.add_argument("--final", type=Path, default=ROOT / "results/headroom_final.json")
    ap.add_argument("--out", type=Path, default=ROOT / "report/figs")
    a = ap.parse_args()
    base, final = json.loads(a.baseline.read_text()), json.loads(a.final.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    success_vs_ceiling(base, final, a.out / "success_vs_ceiling.pdf")
    failure_breakdown(base, final, a.out / "failure_breakdown.pdf")
    print(f"wrote {a.out / 'success_vs_ceiling.pdf'} and {a.out / 'failure_breakdown.pdf'}")


if __name__ == "__main__":
    main()
