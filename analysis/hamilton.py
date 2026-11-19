"""Hamilton's rule, measured: r, B and C of the self-sacrifices in the demo, in offspring.

C, the cost: the children the giver would still have had. B, the benefit: the extra children the
recipient gets thanks to the gift. Both are read from a "reproductive value" table
V(energy, age, time of year) = the mean number of future children of a dot in that state, measured
on thousands of random dots of the same run (children counted from the family tree). Then

    C = V(giver's energy, age, time of year)
    B = V(recipient's energy + gift, ...) - V(recipient's energy, ...)
    r = genetic relatedness of givers and recipients (kinship metrics, r_share)

As a check, B is also measured directly: the recipients' actual future children minus V of their
state before the gift.

    .venv/bin/python analysis/hamilton.py run  results/hamilton 30000 0 1
    .venv/bin/python analysis/hamilton.py plot results/hamilton docs/hamilton.png

Simplifications: offspring are counted as children (not grandchildren), every child counts the
same whenever it is born, and effects on third parties (e.g. relatives competing for the same grass)
are left out.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kin_dots.core.kinship import kinship_report  # noqa: E402
from kin_dots.core.params import Params  # noqa: E402
from kin_dots.core.world import World  # noqa: E402

WORLDS = {"neighbour": "configs/demo_sacrifice.yaml", "stranger": "configs/demo_sacrifice_strangers.yaml"}
SAMPLE_EVERY, SAMPLE_FRAC = 20, 0.3
E_BINS = np.arange(0, 21, 1.0)       # energy grid 0..20 (at the start of a step no dot has 20 or more)
BIRTH, BIRTH_COST = 20.0, 1.0
AGE_BINS = np.arange(0, 330, 30)
PHASES = 12                          # 50-step slices of the 600-step year


class HamiltonWorld(World):
    def __init__(self, p):
        super().__init__(p)
        self.rs = np.random.default_rng([p.seed, 4242])  # sampling only; does not touch the run
        self.samples, self.pairs = [], []

    def _interact(self, idx, nbr_start=None, nbr=None):
        t = self.step_count
        e0, a0 = self.energy.copy(), self.age.copy()
        if t % SAMPLE_EVERY == 0 and idx.size:
            pick = idx[self.rs.random(idx.size) < SAMPLE_FRAC]
            self.samples.append(np.stack([self.id[pick], np.full(pick.size, t), e0[pick], a0[pick]], axis=1))
        out = super()._interact(idx, nbr_start, nbr)
        n = int(self._log_n[0])
        sh = self._log_kind[:n] == 1
        gi, gj = self._log_i[:n][sh], self._log_j[:n][sh]
        if gi.size:
            self.pairs.append(np.stack([np.full(gi.size, t), self.id[gi], e0[gi], a0[gi],
                                        self.id[gj], e0[gj], a0[gj]], axis=1))
        return out


def run_one(job):
    world, seed, steps, out = job
    p = Params.from_yaml(ROOT / WORLDS[world]).replace(seed=seed)
    w = HamiltonWorld(p)
    r_rows = []
    for _ in range(steps):
        w.step()
        if w.step_count % 600 == 0:
            k = kinship_report(w)
            r_rows.append((w.step_count, k.get("r_share", np.nan), k.get("n_share", 0), k.get("r_met", np.nan)))
        if w.population == 0:
            break
    g = w.genealogy
    np.savez_compressed(Path(out) / f"{world}_s{seed}.npz",
                        samples=np.concatenate(w.samples).astype(np.float64),
                        pairs=np.concatenate(w.pairs).astype(np.float64) if w.pairs else np.zeros((0, 7)),
                        parent=g.parent[: g.n], birth=g.birth[: g.n], r=np.array(r_rows, dtype=float),
                        steps=w.step_count, eff=p.share_benefit / p.share_cost, max_age=p.max_age)
    return f"{world} seed {seed}: {sum(len(s) for s in w.samples)} samples, {sum(len(x) for x in w.pairs)} gifts"


def future_children(parent, birth, ids, t):
    """For each (id, t): the number of children of id born at step >= t."""
    has = parent >= 0
    BIG = int(birth.max()) + 10
    key = np.sort(parent[has].astype(np.int64) * BIG + birth[has])
    ids, t = ids.astype(np.int64), t.astype(np.int64)
    return np.searchsorted(key, (ids + 1) * BIG) - np.searchsorted(key, ids * BIG + t)


class ValueTable:
    """V(energy, age, phase): mean future children, linear in energy between bin centres."""

    def __init__(self, e, age, phase, kids):
        ei = np.clip(np.digitize(e, E_BINS) - 1, 0, len(E_BINS) - 1)
        ai = np.clip(np.digitize(age, AGE_BINS) - 1, 0, len(AGE_BINS) - 2)
        s = np.zeros((len(E_BINS), len(AGE_BINS) - 1, PHASES)); n = np.zeros_like(s)
        np.add.at(s, (ei, ai, phase), kids); np.add.at(n, (ei, ai, phase), 1)
        # cells with few samples borrow from the same energy row across all ages (a coarser estimate)
        row_s, row_n = s.sum(axis=1, keepdims=True), n.sum(axis=1, keepdims=True)
        w = np.minimum(n / 30.0, 1.0)
        fine = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
        coarse = np.divide(row_s, row_n, out=np.zeros_like(row_s), where=row_n > 0)
        self.V = w * fine + (1 - w) * coarse
        self.n = n
        # a dot at the birth threshold splits at once: one child now, then it continues with (E - 1) / 2
        self.V[-1] = 1.0 + self._interp(np.full(self.V.shape[1:], (BIRTH - BIRTH_COST) / 2), slice(None))

    def _interp(self, x, ai, phase=None):
        i0 = np.clip(np.floor(x).astype(int), 0, len(E_BINS) - 2)
        f = x - E_BINS[i0]
        if phase is None:  # whole table: x has shape (ages, phases)
            a, ph = np.indices(x.shape)
            return (1 - f) * self.V[i0, a, ph] + f * self.V[i0 + 1, a, ph]
        return (1 - f) * self.V[i0, ai, phase] + f * self.V[i0 + 1, ai, phase]

    def __call__(self, e, age, phase):
        """Expected future children. Energy of 20 or more means splitting at once (one child now,
        then (E - 1) / 2 left), applied repeatedly for very large gifts."""
        e = np.asarray(e, dtype=float).copy()
        age, phase = np.broadcast_to(age, e.shape), np.broadcast_to(phase, e.shape)
        ai = np.clip(np.digitize(age, AGE_BINS) - 1, 0, len(AGE_BINS) - 2)
        extra = np.zeros_like(e)
        while (e >= BIRTH).any():
            m = e >= BIRTH
            extra[m] += 1.0
            e[m] = (e[m] - BIRTH_COST) / 2
        return extra + self._interp(np.clip(e, 0, BIRTH - 1e-9), ai, phase)


def analyse(f, t0, t1):
    d = dict(np.load(f))
    end = int(d["steps"]) - int(d["max_age"]) - 1  # futures after this are cut off by the end of the run
    S, P = d["samples"], d["pairs"]
    S = S[(S[:, 1] >= t0) & (S[:, 1] < min(t1, end))]
    P = P[(P[:, 0] >= t0) & (P[:, 0] < min(t1, end))]
    kids = future_children(d["parent"], d["birth"], S[:, 0], S[:, 1])
    V = ValueTable(S[:, 2], S[:, 3], ((S[:, 1] % 600) // 50).astype(int), kids)
    ph = ((P[:, 0] % 600) // 50).astype(int)
    C = V(P[:, 2], P[:, 3], ph)
    gift = d["eff"] * P[:, 2]
    B = V(P[:, 5] + gift, P[:, 6], ph) - V(P[:, 5], P[:, 6], ph)
    actual = future_children(d["parent"], d["birth"], P[:, 4], P[:, 0]) - V(P[:, 5], P[:, 6], ph)
    rr = d["r"]
    sel = (rr[:, 0] > t0) & (rr[:, 0] <= t1) & (rr[:, 2] > 0)
    r = float(np.average(rr[sel, 1], weights=rr[sel, 2])) if sel.any() else np.nan
    return dict(C=C, B=B, B_actual=actual, r=r, e_donor=P[:, 2], e_recip=P[:, 5], phase=ph, n=len(P), V=V)


def summary(d, periods=((2000, 8000), (20000, 30000))):
    rows = {}
    for world in WORLDS:
        files = sorted(Path(d).glob(f"{world}_s*.npz"))
        for t0, t1 in periods:
            res = [analyse(f, t0, t1) for f in files]
            C = np.concatenate([x["C"] for x in res]); B = np.concatenate([x["B"] for x in res])
            Ba = np.concatenate([x["B_actual"] for x in res]); r = np.nanmean([x["r"] for x in res])
            rows[(world, t0)] = dict(r=r, B=B.mean(), B_actual=Ba.mean(), C=C.mean(), n=len(C),
                                     rB_C=r * B.mean() - C.mean(), res=res)
            print(f"{world:9s} steps {t0:5d}-{t1:5d}: n={len(C):6d}  r={r:5.2f}  B={B.mean():.3f} (actual {Ba.mean():.3f})"
                  f"  C={C.mean():.3f}  rB-C={r * B.mean() - C.mean():+.3f}")
    return rows


def plot(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary(d)
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=.38, wspace=.28)
    col = {"neighbour": "#d9a400", "stranger": "#2a9fb0"}

    # 1. reproductive value: future children by energy, summer vs winter (evolved neighbour world)
    ax = fig.add_subplot(gs[0, 0])
    V = rows[("neighbour", 20000)]["res"][0]["V"]
    E = np.linspace(0, 30, 121)
    for phase, label, c in ((2, "spring (steps 100-150 of the year)", "#3fa66b"), (6, "early winter (300-350)", "#6a8db8"),
                            (9, "deep winter (450-500)", "#2c3e75")):
        ax.plot(E, V(E, np.full(E.size, 100), np.full(E.size, phase)), color=c, lw=2.4, label=label)
    ax.axvline(20, color="k", lw=.8, ls=":"); ax.text(20.3, .05, "splits at 20", fontsize=8)
    ax.set_xlabel("own energy"); ax.set_ylabel("expected future children")
    ax.set_title("What a dot is worth: future children\nby energy and time of year (age 90-120)", fontsize=11)
    ax.legend(fontsize=8); ax.set_ylim(0, None); ax.grid(alpha=.25)

    # 2. r, B, C, rB - C per world and period
    ax = fig.add_subplot(gs[0, 1:])
    labels, keys = [], [("neighbour", 2000), ("neighbour", 20000), ("stranger", 2000), ("stranger", 20000)]
    x = np.arange(len(keys))
    w = .2
    for k, (name, c) in enumerate((("r·B", "#d9a400"), ("C", "#c0392b"), ("r·B − C", "#333333"))):
        vals = [rows[key]["r"] * rows[key]["B"] if name == "r·B" else rows[key]["C"] if name == "C" else rows[key]["rB_C"] for key in keys]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=c, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (.01 if v >= 0 else -.03), f"{v:+.2f}" if name == "r·B − C" else f"{v:.2f}",
                    ha="center", fontsize=8, va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x, [f"gift to a {w_}\n{'early (steps 2-8k)' if t0 == 2000 else 'evolved (steps 20-30k)'}\n"
                      f"r = {rows[(w_, t0)]['r']:.2f}, B = {rows[(w_, t0)]['B']:.2f}" for w_, t0 in keys], fontsize=9)
    ax.set_ylabel("children")
    ax.set_title("Hamilton's rule for the actual sacrifices (mean per sacrifice, in children)", fontsize=11)
    ax.legend(fontsize=9)

    # 3. by giver's energy: who should give?
    ax = fig.add_subplot(gs[1, :2])
    bins = np.arange(0, 22, 2)
    for world in WORLDS:
        res = rows[(world, 20000)]["res"]
        e = np.concatenate([x["e_donor"] for x in res]); C = np.concatenate([x["C"] for x in res]); B = np.concatenate([x["B"] for x in res])
        r = rows[(world, 20000)]["r"]
        idx = np.digitize(e, bins) - 1
        m = [(bins[i] + 1, np.mean(r * B[idx == i] - C[idx == i]), (idx == i).sum()) for i in range(len(bins) - 1) if (idx == i).sum() > 30]
        ax.plot([a for a, _, _ in m], [b for _, b, _ in m], "o-", color=col[world], lw=2.4, label=f"gift to a {world} (r = {r:.2f})")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("giver's energy at the moment of the gift")
    ax.set_ylabel("r·B − C  (children)")
    ax.set_title("Does the sacrifice pay for the gene? By the giver's energy (evolved phase)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=.25)

    # 4. B: predicted from V vs actually observed
    ax = fig.add_subplot(gs[1, 2])
    for i, key in enumerate(keys):
        ax.bar(i - .18, rows[key]["B"], .36, color=col[key[0]], alpha=.9, label="B from V" if i == 0 else None)
        ax.bar(i + .18, rows[key]["B_actual"], .36, color=col[key[0]], alpha=.45, hatch="//", label="B observed" if i == 0 else None)
    ax.set_xticks(range(len(keys)), [f"{w_[:5]}.\n{'early' if t0 == 2000 else 'evolved'}" for w_, t0 in keys], fontsize=9)
    ax.set_ylabel("extra children of the recipient")
    ax.set_title("Check: B from the table vs\nrecipients' actual extra children", fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle("Hamilton's rule, measured in the self-sacrifice demo (2 runs per world)", fontsize=14, y=.995)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("saved", out)


def main():
    if sys.argv[1] == "run":
        out, steps, seeds = Path(sys.argv[2]), int(sys.argv[3]), [int(s) for s in sys.argv[4:]] or [0, 1]
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(w, s, steps, str(out)) for w in WORLDS for s in seeds]
        with Pool(len(jobs)) as pool:
            for msg in pool.imap_unordered(run_one, jobs):
                print(msg, flush=True)
    elif sys.argv[1] == "plot":
        plot(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        summary(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
