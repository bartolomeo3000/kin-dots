"""How the sacrifice genes evolve: gene a, and p(sacrifice) as a function of energy over time.

Every dot decides with p = sigmoid(a + b * min(E / 20, 2)) per encounter. Snapshots of every
living dot's (a, b, energy) are taken every SNAP steps, so p(E) is the population mean of
each dot's own curve (not the curve of the mean genes).

    .venv/bin/python analysis/sacrifice_genes.py run  results/sacrifice_genes 30000 0 1 2
    .venv/bin/python analysis/sacrifice_genes.py plot results/sacrifice_genes docs/demo_sacrifice_genes.png
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kin_dots.core import genome as G  # noqa: E402
from kin_dots.core.params import Params  # noqa: E402
from kin_dots.core.world import World  # noqa: E402

CONDS = {
    "neighbour": ("configs/demo_sacrifice.yaml", "gift to a neighbour", "#d9a400"),
    "stranger": ("configs/demo_sacrifice_strangers.yaml", "gift to a stranger", "#2a9fb0"),
}
SNAP = 300
E_GRID = np.linspace(0, 20, 41)  # energy (birth threshold = 20)
E_BINS = np.linspace(0, 40, 81)  # for the population's energy distribution


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_one(job):
    cond, seed, steps, out = job
    p = Params.from_yaml(ROOT / CONDS[cond][0]).replace(seed=seed)
    w = World(p)
    a0, b0 = G.SHARE_W.start, G.SHARE_W.start + 2
    rec = {k: [] for k in ("step", "a_mean", "b_mean", "a_q", "b_q", "p_of_e", "e_hist", "pop")}
    for t in range(1, steps + 1):
        w.step()
        if w.population == 0:
            break
        if t % SNAP == 0:
            al = w.alive
            a, b, e = w.genome[al, a0], w.genome[al, b0], w.energy[al]
            x = np.minimum(E_GRID / p.birth_threshold, 2.0)
            rec["step"].append(t)
            rec["a_mean"].append(a.mean())
            rec["b_mean"].append(b.mean())
            rec["a_q"].append(np.quantile(a, [0.1, 0.5, 0.9]))
            rec["b_q"].append(np.quantile(b, [0.1, 0.5, 0.9]))
            rec["p_of_e"].append(sig(a[:, None] + b[:, None] * x[None, :]).mean(axis=0))
            rec["e_hist"].append(np.histogram(e, E_BINS)[0] / e.size)
            rec["pop"].append(e.size)
    np.savez_compressed(Path(out) / f"{cond}_seed{seed}.npz", **{k: np.array(v) for k, v in rec.items()})
    return f"{cond} seed {seed}: {len(rec['step'])} snapshots, final pop {w.population}"


def load(d: Path):
    runs = {c: [] for c in CONDS}
    for f in sorted(d.glob("*.npz")):
        runs[f.stem.split("_seed")[0]].append(dict(np.load(f)))
    return runs


def yearly(r, period=600):
    """The same run, keeping only snapshots at one point of the year (start of the year),
    so that seasonal swings in who is alive do not show up as stripes."""
    keep = r["step"] % period == 0
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in r.items()}


def plot(runs, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.45, wspace=0.42,
                          width_ratios=[1.25, 1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1:])
    ax_h = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_e = fig.add_subplot(gs[1, 2])

    # gene a over time
    for cond, rs in runs.items():
        _, label, col = CONDS[cond]
        L = min(r["step"].size for r in rs)
        for r in rs:
            ax_a.plot(r["step"], r["a_mean"], color=col, lw=0.7, alpha=0.4)
        ax_a.plot(rs[0]["step"][:L], np.mean([r["a_mean"][:L] for r in rs], axis=0), color=col, lw=2.4,
                  label=f"{label}: mean a")
        q = np.mean([r["a_q"][:L] for r in rs], axis=0)
        ax_a.fill_between(rs[0]["step"][:L], q[:, 0], q[:, 2], color=col, alpha=0.12, lw=0)
    ax_a.axhline(-9, color="k", lw=0.6, ls=":")
    ax_a.text(300, -9.25, "start: a = -9", fontsize=8, va="top")
    ax_a.set_xlabel("step")
    ax_a.set_ylabel("gene a")
    ax_a.set_title("Gene a: the overall urge to sacrifice\n(thin: runs, thick: mean, band: 10-90% of dots)")
    ax_a.legend(fontsize=8, loc="upper left")

    # p(E) at several moments, neighbour world; control at the end
    yr = {c: [yearly(r) for r in rs] for c, rs in runs.items()}
    nb, st = yr["neighbour"], yr["stranger"]
    L = min(r["step"].size for r in nb)
    steps = nb[0]["step"][:L]
    P = np.mean([r["p_of_e"][:L] for r in nb], axis=0)  # (time, E)
    moments = [3000, 6000, 9000, 12000, 15000, 20000, 25000, 30000]
    cmap = plt.get_cmap("viridis")
    for k, m in enumerate(moments):
        i = int(np.argmin(np.abs(steps - m)))
        ax_c.plot(E_GRID, 100 * P[i], color=cmap(k / (len(moments) - 1)), lw=2,
                  label=f"neighbour, step {int(steps[i]):,}")
    Ls = min(r["step"].size for r in st)
    Ps = np.mean([r["p_of_e"][:Ls] for r in st], axis=0)
    ax_c.plot(E_GRID, 100 * Ps[-1], color=CONDS["stranger"][2], lw=2, ls="--",
              label=f"stranger (control), step {int(st[0]['step'][Ls - 1]):,}")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("own energy (20 = enough to reproduce)")
    ax_c.set_ylabel("p(sacrifice) per encounter, % (log scale)")
    ax_c.set_title("Chance to sacrifice as a function of own energy, as evolution goes on\n"
                   "(mean over all dots of each dot's own curve; at the start of each year)")
    ax_c.legend(fontsize=7.5, ncol=2, loc="lower left")
    ax_c.set_xlim(0, 20)

    # heatmaps p(E, t)
    vmin, vmax = 1e-3, max(P.max(), Ps.max()) * 100
    for ax, (cond, PP, ss) in zip(ax_h, (("neighbour", P, steps), ("stranger", Ps, st[0]["step"][:Ls]))):
        _, label, col = CONDS[cond]
        im = ax.pcolormesh(ss, E_GRID, 100 * PP.T, cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax),
                           shading="nearest")
        # where the dots actually are: mean energy of the population
        rs = yr[cond]
        LL = len(ss)
        eh = np.mean([r["e_hist"][:LL] for r in rs], axis=0)
        centres = (E_BINS[:-1] + E_BINS[1:]) / 2
        ax.plot(ss, eh @ centres, color="w", lw=1.2, ls="--", label="mean energy of the dots")
        ax.set_ylim(0, 20)
        ax.set_xlabel("step")
        ax.set_xticks([0, 10000, 20000, 30000])
        ax.set_ylabel("own energy")
        ax.set_title(f"{label}:\np(sacrifice) by own energy and time", color=col, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
    cb = fig.colorbar(im, ax=ax_h[1], fraction=0.06, pad=0.03)
    cb.set_label("p(sacrifice) per encounter, %")

    # how much stronger the urge of weak dots is than of strong ones
    for cond, rs in yr.items():
        _, label, col = CONDS[cond]
        LL = min(r["step"].size for r in rs)
        ratio = np.mean([r["p_of_e"][:LL, 10] / r["p_of_e"][:LL, 38] for r in rs], axis=0)  # E=5 vs E=19
        ax_e.plot(rs[0]["step"][:LL], ratio, color=col, lw=2, label=label)
    ax_e.axhline(1, color="k", lw=0.6, ls=":")
    ax_e.set_yscale("log")
    ax_e.set_xticks([0, 10000, 20000, 30000])
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("p(sacrifice) at energy 5 / at energy 19")
    ax_e.set_title("Weak vs well-fed: how many times more likely\nthe weak dot is to sacrifice (> 1: the weak)")
    ax_e.legend(fontsize=8)

    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    for cond, rs in runs.items():
        LL = min(r["step"].size for r in rs)
        PP = np.mean([r["p_of_e"][:LL] for r in rs], axis=0)
        for m in (3000, 6000, 12000, 20000, 30000):
            i = int(np.argmin(np.abs(rs[0]["step"][:LL] - m)))
            a = np.mean([r["a_mean"][i] for r in rs])
            b = np.mean([r["b_mean"][i] for r in rs])
            print(f"{cond:9s} step {int(rs[0]['step'][i]):6d}  a {a:6.2f}  b {b:6.2f}  "
                  f"p% E=5 {100 * PP[i, 10]:.3f}  E=10 {100 * PP[i, 20]:.3f}  E=19 {100 * PP[i, 38]:.3f}")


def main():
    mode, d = sys.argv[1], Path(sys.argv[2])
    if mode == "run":
        steps = int(sys.argv[3])
        seeds = [int(s) for s in sys.argv[4:]] or [0, 1, 2]
        d.mkdir(parents=True, exist_ok=True)
        jobs = [(c, s, steps, str(d)) for c in CONDS for s in seeds]
        with Pool(min(len(jobs), 8)) as pool:
            for msg in pool.imap_unordered(run_one, jobs):
                print(msg, flush=True)
    else:
        plot(load(d), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
