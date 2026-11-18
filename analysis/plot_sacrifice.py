"""Figure for the sacrifice demo: share of deaths that are self-sacrifice, over time.

    .venv/bin/python analysis/plot_sacrifice.py results/demo_curves docs/demo_sacrifice.png

Reads the per-run CSVs written by `kin_dots.experiments.sweep --out DIR` for the
conditions `sacrifice-local` and `sacrifice-strangers` (any seeds).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CONDS = {
    "sacrifice-local": ("gift goes to a neighbour (usually a relative)", "#d9a400"),
    "sacrifice-strangers": ("gift goes to a random dot anywhere (a stranger)", "#2a9fb0"),
}


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    step = np.array([float(r["step"]) for r in rows])
    sac = np.array([float(r["total_sacrificed"]) for r in rows])
    died = np.array([float(r["total_deaths"]) for r in rows])
    ds, dd = np.diff(np.r_[0, sac]), np.diff(np.r_[0, died])
    frac = np.where(ds + dd > 0, ds / np.maximum(ds + dd, 1), np.nan)
    b = np.array([float(r.get("mean_share_w2") or "nan") for r in rows])
    pop = np.array([float(r["pop_mean_win"]) for r in rows])
    return step, frac, b, pop


def main(src: str, out: str) -> None:
    src_p = Path(src)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2, 1, 1]})
    for cond, (label, color) in CONDS.items():
        runs = [load(p) for p in sorted(src_p.glob(f"{cond}_seed*.csv"))]
        if not runs:
            continue
        n = min(len(r[0]) for r in runs)
        for k, ax in enumerate(axes):
            data = np.array([r[k + 1][:n] for r in runs])
            if k == 0:
                data = data * 100
            for d in data:
                ax.plot(runs[0][0][:n], d, color=color, alpha=0.3, lw=1)
            ax.plot(runs[0][0][:n], np.nanmean(data, axis=0), color=color, lw=2.5,
                    label=f"{label} ({len(runs)} runs)")
    axes[0].set_title("Share of deaths that are self-sacrifice (all energy given away)")
    axes[0].set_ylabel("% of all deaths")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set_title("Gene b: how the urge depends on own energy\n(< 0: the weak sacrifice themselves)")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[2].set_title("Mean population size (per year)")
    for ax in axes:
        ax.set_xlabel("simulation step")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
