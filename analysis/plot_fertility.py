"""Figure: self-sacrifice on a patchy landscape (output of analysis/fertility_runs.py).

    .venv/bin/python analysis/plot_fertility.py results/fertility docs/fertility.png [results/fertility_long]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kin_dots.core.landscape import Landscape  # noqa: E402
from kin_dots.core.params import Params  # noqa: E402

CONDS = [("uniform", "uniform map", "#444444"), ("redraw", "new map every year", "#2a7fd4"),
         ("drift", "map drifts (memory 0.7)", "#3fa66b"), ("static", "fixed map, contrast 1.0", "#d98a00"),
         ("static_c1.5", "fixed map, contrast 1.5", "#c0392b")]
SHORT = {"uniform": "uniform", "redraw": "new map\nyearly", "drift": "drifting\nmap", "static": "fixed\nc = 1.0",
         "static_c1.5": "fixed\nc = 1.5"}


def load(d):
    return [json.loads(f.read_text()) for f in sorted(Path(d).glob("*.json"))]


def tail(r, k, last=10):
    v = [x[k] for x in r["rows"][-last:] if x.get(k) is not None]
    return float(np.mean(v)) if v else np.nan


def main(src, out, long=None):
    runs = load(src)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.15, 1.05], hspace=.34, wspace=.3)

    # row 1: what the landscapes look like
    base = Params.from_yaml(ROOT / "configs/demo_sacrifice.yaml").replace(fertility_map="noise", fertility_scale=30)
    maps = [("fixed map, contrast 1.0", Landscape(base.replace(fertility_contrast=1.0)).grid),
            ("fixed map, contrast 1.5", Landscape(base.replace(fertility_contrast=1.5)).grid)]
    L = Landscape(base.replace(fertility_contrast=1.0, fertility_memory=0.0)); L.update(600)
    maps.append(("new map every year: year 2", L.grid.copy()))
    L = Landscape(base.replace(fertility_contrast=1.0, fertility_memory=0.7)); L.update(600)
    maps.append(("drifting map: year 2", L.grid.copy()))
    for k, (title, g) in enumerate(maps):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(g.T, origin="lower", cmap="YlGn", vmin=0, vmax=3.5, extent=(0, 300, 0, 300))
        ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=fig.axes[:4], fraction=.015, pad=.01)
    cb.set_label("grass regrowth rate (map mean = 1)")

    # row 2, left: sacrifice share over time
    ax = fig.add_subplot(gs[1, :2])
    for key, label, col in CONDS:
        for world, ls, lw in (("neighbour", "-", 2.4), ("stranger", ":", 1.6)):
            rs = [r for r in runs if r["cond"] == key and r["world"] == world]
            if not rs:
                continue
            n = min(len(r["rows"]) for r in rs)
            y = np.nanmean([[x["share"] if x["share"] is not None else np.nan for x in r["rows"][:n]] for r in rs], axis=0)
            k = 3
            ys = np.convolve(np.nan_to_num(y), np.ones(k) / k, mode="valid")
            ax.plot(np.arange(k, n + 1) * 600, ys * 100, ls, color=col, lw=lw, label=label if world == "neighbour" else None)
    ax.set_xlabel("step"); ax.set_ylabel("deaths that are self-sacrifice (%)")
    ax.set_ylim(0, 80); ax.set_xlim(0, 30000); ax.grid(alpha=.25)
    ax.set_title("Self-sacrifice over time (solid: gift to a neighbour, dotted: to a stranger; mean of 3 runs)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    # row 2, middle: final share per condition, each run a dot
    ax = fig.add_subplot(gs[2, :2])
    for i, (key, label, col) in enumerate(CONDS):
        for world, off, mk in (("neighbour", -.15, "o"), ("stranger", .15, "s")):
            v = [100 * tail(r, "share") for r in runs if r["cond"] == key and r["world"] == world]
            if v:
                ax.scatter(np.full(len(v), i + off), v, color=col, marker=mk, s=40, alpha=.85 if world == "neighbour" else .5)
                ax.hlines(np.mean(v), i + off - .12, i + off + .12, color=col, lw=2.5)
    ax.set_xticks(range(len(CONDS)), [SHORT[c[0]] for c in CONDS], fontsize=9)
    ax.set_ylabel("sacrifice share, last 10 years (%)"); ax.set_ylim(0, 80); ax.grid(axis="y", alpha=.25)
    ax.set_title("After 30,000 steps\n(circles: neighbour, squares: stranger)", fontsize=11)

    # row 2, right: where the givers are, and how poor the recipients are
    ax = fig.add_subplot(gs[2, 2:])
    keys = [c for c in CONDS if c[0] != "uniform"]
    for i, (key, label, col) in enumerate(keys):
        rs = [r for r in runs if r["cond"] == key and r["world"] == "neighbour"]
        v = [tail(r, "gift_fert") / tail(r, "pop_fert") for r in rs]
        ax.bar(i, np.mean(v), color=col, alpha=.85)
        ax.scatter(np.full(len(v), i), v, color="k", s=12, zorder=3)
    ax.axhline(1, color="k", lw=.8, ls="--")
    ax.set_xticks(range(len(keys)), [SHORT[c[0]] for c in keys], fontsize=9)
    ax.set_ylabel("fertility under givers / under all dots")
    ax.set_title("Givers live on poorer ground\n(gift to a neighbour, last 10 years)", fontsize=11)
    ax.set_ylim(0, 1.3)

    # row 2, right: 150-year runs, uniform vs fixed map (neighbour world only)
    if long:
        ax = fig.add_subplot(gs[1, 2:])
        lr = load(long)
        for key, label, col in CONDS:
            rs = [r for r in lr if r["cond"] == key]
            for j, r in enumerate(rs):
                y = np.array([x["share"] if x["share"] is not None else np.nan for x in r["rows"]])
                ys = np.convolve(np.nan_to_num(y), np.ones(5) / 5, mode="valid")
                ax.plot(np.arange(5, len(y) + 1), ys * 100, color=col, lw=2, alpha=.9 if j == 0 else .55,
                        label=f"{label} (2 runs)" if j == 0 else None)
            if rs:
                end = np.mean([np.nanmean([x["share"] for x in r["rows"][-25:] if x["share"] is not None]) for r in rs])
                ax.annotate(f"{end * 100:.0f}%", (150, end * 100), xytext=(6, 0), textcoords="offset points", va="center", color=col, fontsize=12)
        ax.set_xlim(0, 160); ax.set_ylim(0, 80); ax.grid(alpha=.25)
        ax.set_xlabel("year (600 steps)"); ax.set_ylabel("deaths that are self-sacrifice (%)")
        ax.set_title("150 years, gift to a neighbour: a lower plateau, not just slower evolution", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Self-sacrifice on a patchy landscape: grass regrows at different rates in different places",
                 fontsize=14, y=.995)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
