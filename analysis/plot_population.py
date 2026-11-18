"""Figure: raw population over time in the two demo worlds.

    .venv/bin/python analysis/plot_population.py results/population.npz docs/demo_population.png

Input from analysis/population_runs.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kin_dots.core import Params  # noqa: E402

COND = {
    "neighbour": ("gift goes to a neighbour (usually a relative)", "#d9a400"),
    "stranger": ("gift goes to a random dot anywhere (a stranger)", "#2a9fb0"),
}
ZOOM = (24000, 30000)


def shade_winters(ax, lo, hi, period):
    """Grey bands where food regrows slower than normal (sin < 0: second half of each period)."""
    start = (lo // period) * period
    for t in range(start, hi, period):
        ax.axvspan(max(t + period / 2, lo), min(t + period, hi), color="#000000", alpha=0.06, lw=0)


def main(src: str, out: str) -> None:
    d = np.load(src)
    seeds = sorted({int(k.split("_s")[1].split("_")[0]) for k in d.files if k.endswith("_pop")})
    period = Params.from_yaml("configs/demo_sacrifice.yaml").season_period

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.38, wspace=0.12)
    top = fig.add_subplot(gs[0, :])
    for cond, (label, color) in COND.items():
        k = f"{cond}_s{seeds[0]}"
        top.plot(d[f"{k}_step"], d[f"{k}_pop"], color=color, lw=0.8, label=label)
    top.set_title(f"Population size at every step, whole run (seed {seeds[0]}). "
                  f"Each wave is one year ({period} steps).")
    top.set_ylabel("number of dots")
    top.set_xlabel("simulation step")
    top.set_ylim(bottom=0)
    top.set_xlim(left=0)
    top.legend(loc="upper left", frameon=False)
    top.axvspan(*ZOOM, color="#888888", alpha=0.12, lw=0)
    top.text(ZOOM[0] + 100, top.get_ylim()[1] * 0.92, "zoomed in below", color="#555555")

    ymax = 0
    axes = []
    for col, (cond, (label, color)) in enumerate(COND.items()):
        ax = fig.add_subplot(gs[1, col])
        axes.append(ax)
        shade_winters(ax, *ZOOM, period)
        for i, s in enumerate(seeds):
            k = f"{cond}_s{s}"
            st, pop = d[f"{k}_step"], d[f"{k}_pop"]
            m = (st >= ZOOM[0]) & (st <= ZOOM[1])
            ax.plot(st[m], pop[m], color=color, lw=1.4, alpha=[1.0, 0.6, 0.35][i % 3], label=f"seed {s}")
            ymax = max(ymax, pop[m].max() if m.any() else 0)
        ax.set_title(f"{label}\nsteps {ZOOM[0]}–{ZOOM[1]} (grey bands = winter)")
        ax.set_xlabel("simulation step")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.08)
    axes[0].set_ylabel("number of dots")
    for ax in [top] + axes:
        ax.spines[["top", "right"]].set_visible(False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
