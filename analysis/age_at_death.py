"""Age at death in the sacrifice demo, by cause of death.

Runs both demo worlds (gift to a neighbour / gift to a stranger) and records every death:
step, age, cause (sacrifice / hunger / old age), energy just before, and how many children
the dot had had. Measurement only: the recording subclass does not change the simulation.

    .venv/bin/python analysis/age_at_death.py run  results/age_at_death 30000 0 1 2 3
    .venv/bin/python analysis/age_at_death.py plot results/age_at_death docs/demo_age_at_death.png
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kin_dots.core.params import Params  # noqa: E402
from kin_dots.core.world import World  # noqa: E402

CONDS = {
    "neighbour": ("configs/demo_sacrifice.yaml", "gift to a neighbour", "#d9a400"),
    "stranger": ("configs/demo_sacrifice_strangers.yaml", "gift to a stranger", "#2a9fb0"),
}
SACRIFICE, HUNGER, OLD_AGE = 0, 1, 2
CAUSES = {SACRIFICE: ("self-sacrifice", "#e8590c"), HUNGER: ("hunger", "#6c757d"),
          OLD_AGE: ("old age", "#7b5ea7")}


class RecordingWorld(World):
    """World that logs each death with its cause. Same RNG calls, so the same run."""

    def __init__(self, params: Params):
        super().__init__(params)
        self.rec = []  # (step, slot ids, age, cause, energy before)

    def _log(self, slots, cause, energy):
        if slots.size:
            self.rec.append((np.full(slots.size, self.step_count), self.id[slots].copy(),
                             self.age[slots].copy(), np.full(slots.size, cause), energy.copy()))

    def _interact(self, idx, nbr_start, nbr):
        e = self.energy[idx].copy()
        out = super()._interact(idx, nbr_start, nbr)
        gone = ~self.alive[idx]  # no fights or predators in the demo: only sacrifices
        self._log(idx[gone], SACRIFICE, e[gone])
        return out

    def _die(self, idx):
        e = self.energy[idx]
        starved = e <= 0
        old = ~starved & (self.age[idx] >= self.p.max_age) if self.p.max_age > 0 else np.zeros_like(starved)
        self._log(idx[starved], HUNGER, e[starved])
        self._log(idx[old], OLD_AGE, e[old])
        return super()._die(idx)


def run_one(job):
    cond, seed, steps, out = job
    p = Params.from_yaml(ROOT / CONDS[cond][0]).replace(seed=seed)
    w = RecordingWorld(p)
    pop = np.zeros(steps, dtype=np.int32)
    for t in range(steps):
        w.step()
        pop[t] = w.population
        if w.population == 0:
            break
    step, ids, age, cause, energy = (np.concatenate(c) for c in zip(*w.rec))
    assert (cause == SACRIFICE).sum() == w.total_sacrificed and (cause == HUNGER).sum() == w.total_starved, \
        "death log does not match the world's counters"
    g = w.genealogy
    par = g.parent[: g.n]
    children = np.bincount(par[par >= 0], minlength=g.n)
    np.savez_compressed(Path(out) / f"{cond}_seed{seed}.npz", step=step.astype(np.int32),
                        age=age.astype(np.int16), cause=cause.astype(np.int8),
                        energy=energy.astype(np.float32), children=children[ids].astype(np.int16),
                        pop=pop, max_age=p.max_age, season=p.season_period,
                        birth_threshold=p.birth_threshold)
    return f"{cond} seed {seed}: {step.size} deaths, final pop {w.population}"


def load(d: Path):
    runs = {c: [] for c in CONDS}
    for f in sorted(d.glob("*.npz")):
        cond = f.stem.split("_seed")[0]
        runs[cond].append(dict(np.load(f)))
    return runs


def summary(runs, t0):
    for cond, rs in runs.items():
        print(f"== {CONDS[cond][1]}  (deaths after step {t0}, {len(rs)} runs pooled)")
        age = np.concatenate([r["age"][r["step"] >= t0] for r in rs])
        cause = np.concatenate([r["cause"][r["step"] >= t0] for r in rs])
        kids = np.concatenate([r["children"][r["step"] >= t0] for r in rs])
        en = np.concatenate([r["energy"][r["step"] >= t0] for r in rs])
        print(f"   all deaths: n={age.size:8d}  mean age {age.mean():6.1f}  median {np.median(age):5.0f}"
              f"  childless {np.mean(kids == 0):5.1%}  mean children {kids.mean():.2f}")
        for c, (name, _) in CAUSES.items():
            m = cause == c
            if m.any():
                print(f"   {name:15s} {m.mean():6.1%} of deaths  mean age {age[m].mean():6.1f}"
                      f"  median {np.median(age[m]):5.0f}  age<100: {np.mean(age[m] < 100):5.1%}"
                      f"  childless {np.mean(kids[m] == 0):5.1%}  mean children {kids[m].mean():.2f}"
                      f"  energy before {en[m].mean():5.2f}")
        per_run = [r["age"][r["step"] >= t0].mean() for r in rs]
        print("   mean age per run:", " ".join(f"{a:.1f}" for a in per_run))


def plot(runs, out: Path, t0: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.38, wspace=0.28)
    ax_t = fig.add_subplot(gs[0, :2])
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_h = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_k = fig.add_subplot(gs[1, 2])

    # 1. mean age at death over time, per year
    for cond, rs in runs.items():
        _, label, col = CONDS[cond]
        curves, medians = [], []
        for r in rs:
            year = r["step"] // r["season"]
            n_years = int(year.max()) + 1
            s = np.bincount(year, weights=r["age"], minlength=n_years)
            n = np.bincount(year, minlength=n_years)
            m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            curves.append(m)
            order = np.argsort(year, kind="stable")
            parts = np.split(r["age"][order], np.cumsum(n)[:-1])
            medians.append(np.array([np.median(a) if a.size else np.nan for a in parts]))
            ax_t.plot(np.arange(n_years) * r["season"], m, color=col, lw=0.6, alpha=0.3)
        L = min(len(c) for c in curves)
        x = np.arange(L) * rs[0]["season"]
        ax_t.plot(x, np.nanmean([c[:L] for c in curves], axis=0), color=col, lw=2.2, label=f"{label}: mean")
        ax_t.plot(x, np.nanmean([c[:L] for c in medians], axis=0), color=col, lw=1.6, ls="--",
                  label=f"{label}: median")
    ax_t.axvline(t0, color="k", lw=0.8, ls=":")
    ax_t.text(t0, 60, f"  panels below: deaths after step {t0}", fontsize=9, va="bottom")
    ax_t.set_ylim(0, None)
    ax_t.set_xlabel("step")
    ax_t.set_ylabel("mean age at death (steps)")
    ax_t.set_title("Age at death, per year (thin: single runs; thick: mean of runs; dashed: median)")
    ax_t.legend(loc="lower left", fontsize=8, ncol=2)

    # 2. mean age by cause, both worlds
    names = [CAUSES[c][0] for c in CAUSES] + ["all deaths"]
    xs = np.arange(len(names))
    for k, (cond, rs) in enumerate(runs.items()):
        _, label, col = CONDS[cond]
        age = np.concatenate([r["age"][r["step"] >= t0] for r in rs])
        cause = np.concatenate([r["cause"][r["step"] >= t0] for r in rs])
        vals = [age[cause == c].mean() if (cause == c).any() else np.nan for c in CAUSES] + [age.mean()]
        share = [np.mean(cause == c) for c in CAUSES] + [1.0]
        meds = [np.median(age[cause == c]) if (cause == c).any() else np.nan for c in CAUSES] + [np.median(age)]
        bars = ax_bar.bar(xs + (k - 0.5) * 0.38, vals, 0.38, color=col, label=label)
        ax_bar.scatter(xs + (k - 0.5) * 0.38, meds, marker="_", s=260, color="k", lw=2,
                       label="median" if k == 0 else None, zorder=3)
        for b, v, s in zip(bars, vals, share):
            if v == v:
                ax_bar.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}\n({s:.0%})", ha="center",
                            va="bottom", fontsize=8)
    ax_bar.set_xticks(xs, names, rotation=15)
    ax_bar.set_ylim(0, 360)
    ax_bar.set_ylabel("mean age at death (steps)")
    ax_bar.set_title("Age at death by cause: bar = mean, black = median\n(in brackets: share of all deaths)")
    ax_bar.legend(fontsize=8, loc="upper left")

    # 3. distribution of age at death, stacked by cause, one panel per world
    bins = np.arange(0, 310, 10)
    for ax, (cond, rs) in zip(ax_h, runs.items()):
        _, label, col = CONDS[cond]
        age = np.concatenate([r["age"][r["step"] >= t0] for r in rs])
        cause = np.concatenate([r["cause"][r["step"] >= t0] for r in rs])
        stack = [np.histogram(age[cause == c], bins)[0] / age.size for c in CAUSES]
        ax.stackplot(bins[:-1] + 5, stack, colors=[CAUSES[c][1] for c in CAUSES],
                     labels=[CAUSES[c][0] for c in CAUSES], alpha=0.9, step="mid")
        ax.set_xlim(0, 305)
        ax.set_xlabel("age at death (steps; max age 300)")
        ax.set_ylabel("share of all deaths per 10-step bin")
        ax.set_title(f"{label}\nmean age at death {age.mean():.0f}, median {np.median(age):.0f}",
                     color=col, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.97, 0.92))
        ax.text(296, 0.059, f"old age (300): {np.mean(cause == OLD_AGE):.0%} of deaths, off the scale \u2192",
                ha="right", va="top", fontsize=8, color=CAUSES[OLD_AGE][1])
    for a in ax_h:
        a.set_ylim(0, 0.06)

    # 4. number of children at death, by cause (left world) and all deaths (both)
    kmax = 6
    for k, (cond, rs) in enumerate(runs.items()):
        _, label, col = CONDS[cond]
        kids = np.concatenate([r["children"][r["step"] >= t0] for r in rs])
        cause = np.concatenate([r["cause"][r["step"] >= t0] for r in rs])
        for c, ls in ((SACRIFICE, "-"), (HUNGER, "--")):
            m = cause == c
            if m.sum() < 50:
                continue
            h = np.bincount(np.minimum(kids[m], kmax), minlength=kmax + 1) / m.sum()
            ax_k.plot(np.arange(kmax + 1), h, ls, color=col, marker="o", ms=4,
                      label=f"{label}: {CAUSES[c][0]}")
    ax_k.set_xticks(range(kmax + 1), [str(i) for i in range(kmax)] + [f"{kmax}+"])
    ax_k.set_xlabel("children had before dying")
    ax_k.set_ylabel("share of those deaths")
    ax_k.set_title("How many children had they had?")
    ax_k.legend(fontsize=7.5)

    fig.suptitle("Who dies, and how old? Age at death in the sacrifice demo", fontsize=14, y=0.995)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


def plot_detail(runs, out: Path, t0: int):
    """Densities of age at death per cause (both worlds overlaid, mean and median marked),
    and age at death against the time of year."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw=dict(hspace=0.42, wspace=0.38))
    bins = np.arange(0, 305, 5)

    def pooled(rs, key):
        return np.concatenate([r[key][r["step"] >= t0] for r in rs])

    # row 1: density of age at death, per cause and for all deaths
    for ax, (title, sel) in zip(axes[0], (("self-sacrifice", SACRIFICE), ("hunger", HUNGER),
                                           ("all deaths", None))):
        for cond, rs in runs.items():
            _, label, col = CONDS[cond]
            age, cause = pooled(rs, "age"), pooled(rs, "cause")
            a = age if sel is None else age[cause == sel]
            if a.size < 50:
                continue
            ax.hist(a, bins, density=True, histtype="stepfilled", color=col, alpha=0.25)
            ax.hist(a, bins, density=True, histtype="step", color=col, lw=1.5,
                    label=f"{label}  (n={a.size:,})")
            ax.axvline(a.mean(), color=col, lw=1.8)
            ax.axvline(np.median(a), color=col, lw=1.8, ls="--")
        ax.set_title(f"{title}: age at death\n(solid line = mean, dashed = median)")
        ax.set_xlabel("age at death (steps)")
        ax.set_ylabel("density")
        ax.set_xlim(0, 305)
        if sel is None:
            ax.set_ylim(0, 0.012)
            ax.text(296, 0.0115, "old-age spike at 300\noff the scale \u2192", ha="right", va="top", fontsize=8)
            ax.legend(fontsize=8, loc="center left")
        else:
            ax.legend(fontsize=8)

    # row 2, panels 1-2: age at death vs time of year, all deaths
    phase_bins = np.arange(0, 601, 20)
    for ax, (cond, rs) in zip(axes[1][:2], runs.items()):
        _, label, col = CONDS[cond]
        age, step = pooled(rs, "age"), pooled(rs, "step")
        H, _, _ = np.histogram2d(step % rs[0]["season"], age, (phase_bins, bins))
        H = H / H.sum()
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("black")
        im = ax.pcolormesh(phase_bins, bins, np.ma.masked_equal(H.T, 0), cmap=cmap,
                           norm=LogNorm(vmin=1e-5, vmax=H.max()))
        ax.axvspan(300, 600, color="#5fa8ff", alpha=0.12)
        ax.set_title(f"{label}: when in the year, at what age\n(colour: share of deaths, log scale)",
                     color=col)
        ax.set_xlabel("step within the year (shaded: winter half)")
        ax.set_ylabel("age at death (steps)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # row 2, panel 3: deaths through the year, by cause
    ax = axes[1][2]
    for cond, rs in runs.items():
        _, label, col = CONDS[cond]
        step, cause = pooled(rs, "step"), pooled(rs, "cause")
        ph = step % rs[0]["season"]
        tot = step.size
        for c, ls in ((SACRIFICE, "-"), (HUNGER, ":"), (OLD_AGE, "--")):
            h = np.histogram(ph[cause == c], phase_bins)[0] / tot
            ax.plot(phase_bins[:-1] + 10, h, ls, color=col, lw=1.6, label=f"{label}: {CAUSES[c][0]}")
    ax.axvspan(300, 600, color="#5fa8ff", alpha=0.08)
    ax.set_xlabel("step within the year (grass grows fastest at 150,\nslowest at 450; shaded: winter half)")
    ax.set_ylabel("share of deaths per 20 steps")
    ax.set_title("When in the year they die, by cause")
    ax.legend(fontsize=7)

    fig.suptitle(f"Age at death in detail (deaths after step {t0}, all runs pooled)", fontsize=14, y=0.995)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


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
        runs = load(d)
        t0 = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
        summary(runs, t0)
        if len(sys.argv) > 3:
            out = Path(sys.argv[3])
            plot(runs, out, t0)
            plot_detail(runs, out.with_name(out.stem + "_detail" + out.suffix), t0)


if __name__ == "__main__":
    main()
