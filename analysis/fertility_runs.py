"""Self-sacrifice on a patchy landscape: run the demo and its control with a fertility map.

Each condition is a set of parameter overrides applied to BOTH worlds (gift to a neighbour,
gift to a stranger), so the two still differ only in who receives the gift:

    .venv/bin/python analysis/fertility_runs.py results/fertility 30000 0 1 2 \\
        --cond uniform \\
        --cond static fertility_map=noise fertility_contrast=1.0 \\
        --cond redraw fertility_map=noise fertility_contrast=1.0 fertility_memory=0.0

Records per year, for each run: sacrifice share of deaths, population (mean and minimum),
relatedness of the neighbours a dot meets (r_met, from the kinship metrics), mean genes a
and b, and the fertility of the ground under the givers compared with under all dots.
Writes one JSON per run; `summary DIR` prints a table of the last years.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kin_dots.core import genome as G  # noqa: E402
from kin_dots.core.kinship import kinship_report  # noqa: E402
from kin_dots.core.params import Params, parse_overrides  # noqa: E402
from kin_dots.core.world import World  # noqa: E402

WORLDS = {"neighbour": "configs/demo_sacrifice.yaml", "stranger": "configs/demo_sacrifice_strangers.yaml"}


def run_one(job):
    name, overrides, world, seed, steps, out = job
    p = Params.from_yaml(ROOT / WORLDS[world]).replace(seed=seed, **overrides)
    w = World(p)
    year = p.season_period or 600
    rows, acc = [], None

    def fresh():
        return dict(sac=0, died=0, pop=[], gift_fert=0.0, gifts=0, pop_fert=[], recip_fert=0.0,
                    donor_e=0.0, recip_e=0.0, shares=0, pop_e=[])

    acc = fresh()
    for _ in range(steps):
        w.step()
        L = w.last
        acc["sac"] += int(L.get("sacrifices", 0))
        acc["died"] += int(L.get("deaths", 0) + L.get("sacrifices", 0))
        acc["pop"].append(w.population)
        acc["donor_e"] += float(L.get("donor_energy", 0.0))
        acc["recip_e"] += float(L.get("recipient_energy", 0.0))
        acc["shares"] += int(L.get("shares", 0))
        grid = w.landscape.grid
        if grid is not None and len(w.last_gifts):
            cs = p.cell_size
            gx = (w.last_gifts[:, 0] // cs).astype(int) % p.grid_cells
            gy = (w.last_gifts[:, 1] // cs).astype(int) % p.grid_cells
            acc["gift_fert"] += float(grid[gx, gy].sum())
            rx = (w.last_gifts[:, 2] // cs).astype(int) % p.grid_cells
            ry = (w.last_gifts[:, 3] // cs).astype(int) % p.grid_cells
            acc["recip_fert"] += float(grid[rx, ry].sum())
            acc["gifts"] += len(gx)
        if w.step_count % 50 == 0 and w.population:
            acc["pop_e"].append(float(w.energy[w.alive].mean()))
        if grid is not None and w.step_count % 50 == 0 and w.population:
            idx = np.flatnonzero(w.alive)
            cx, cy = w.cell_of(idx)
            acc["pop_fert"].append(float(grid[cx, cy].mean()))
        if w.step_count % year == 0:
            kin = kinship_report(w)
            idx = np.flatnonzero(w.alive)
            g = w.genome[idx]
            rows.append(dict(
                year=w.step_count // year, share=acc["sac"] / acc["died"] if acc["died"] else None,
                pop_mean=float(np.mean(acc["pop"])), pop_min=int(np.min(acc["pop"])),
                r_met=kin.get("r_met"), r_share=kin.get("r_share"),
                a=float(g[:, G.SHARE_W.start].mean()) if idx.size else None,
                b=float(g[:, G.SHARE_W.start + 2].mean()) if idx.size else None,
                gift_fert=acc["gift_fert"] / acc["gifts"] if acc["gifts"] else None,
                recip_fert=acc["recip_fert"] / acc["gifts"] if acc["gifts"] else None,
                donor_e=acc["donor_e"] / acc["shares"] if acc["shares"] else None,
                recip_e=acc["recip_e"] / acc["shares"] if acc["shares"] else None,
                pop_e=float(np.mean(acc["pop_e"])) if acc["pop_e"] else None,
                pop_fert=float(np.mean(acc["pop_fert"])) if acc["pop_fert"] else None))
            acc = fresh()
        if w.population == 0:
            break
    res = dict(cond=name, world=world, seed=seed, overrides=overrides, extinct=w.population == 0, rows=rows)
    (Path(out) / f"{name}__{world}__s{seed}.json").write_text(json.dumps(res))
    last = [r["share"] for r in rows[-10:] if r["share"] is not None]
    return f"{name:14s} {world:9s} seed {seed}: " + ("EXTINCT at year %d" % len(rows) if res["extinct"] else
                                                       f"sacrifice {100 * np.mean(last):4.0f}% (last 10 y), pop {rows[-1]['pop_mean']:.0f}")


def load(d: Path):
    return [json.loads(f.read_text()) for f in sorted(Path(d).glob("*.json"))]


def summary(d: Path, last=10):
    runs = load(d)
    conds = sorted({r["cond"] for r in runs}, key=lambda c: min(i for i, r in enumerate(runs) if r["cond"] == c))
    print(f"last {last} years, mean over seeds [min..max];  r_met = relatedness of met neighbours;"
          f"  fert gift/pop = fertility under givers / under all dots")
    print(f"{'condition':14s} {'world':9s} {'n':>2s} {'sacrifice':>18s} {'pop':>6s} {'pop min':>8s} {'r_met':>6s} {'a':>6s} {'b':>6s} {'fert gift/pop':>13s}  extinct")
    for c in conds:
        for wname in ("neighbour", "stranger"):
            rs = [r for r in runs if r["cond"] == c and r["world"] == wname]
            if not rs:
                continue
            def tail(r, k):
                v = [x[k] for x in r["rows"][-last:] if x.get(k) is not None]
                return float(np.mean(v)) if v else np.nan
            alive = [r for r in rs if not r["extinct"]]
            sh = [tail(r, "share") for r in alive]
            ratio = [tail(r, "gift_fert") / tail(r, "pop_fert") for r in alive]
            fmt = lambda v: f"{np.nanmean(v):6.2f}" if len(v) else "   nan"
            print(f"{c:14s} {wname:9s} {len(rs):2d} "
                  + (f"{100 * np.mean(sh):5.0f}% [{100 * min(sh):3.0f}..{100 * max(sh):3.0f}]" if sh else f"{'—':>18s}")
                  + f" {np.mean([tail(r, 'pop_mean') for r in alive]) if alive else np.nan:6.0f}"
                  + f" {np.mean([min(x['pop_min'] for x in r['rows'][-last:]) for r in alive]) if alive else np.nan:8.0f}"
                  + f" {fmt([tail(r, 'r_met') for r in alive])} {fmt([tail(r, 'a') for r in alive])} {fmt([tail(r, 'b') for r in alive])}"
                  + f" {fmt(ratio):>13s}  {sum(r['extinct'] for r in rs)}/{len(rs)}")


def main():
    if sys.argv[1] == "summary":
        summary(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 10)
        return
    out, steps = Path(sys.argv[1]), int(sys.argv[2])
    rest = sys.argv[3:]
    only = None
    if "--only" in rest:  # --only neighbour / --only stranger: run just one of the two worlds
        k = rest.index("--only"); only = rest[k + 1]; del rest[k:k + 2]
    seeds = []
    while rest and rest[0] != "--cond":
        seeds.append(int(rest.pop(0)))
    conds, cur = [], None
    for tok in rest:
        if tok == "--cond":
            cur = [None, []]; conds.append(cur)
        elif cur[0] is None:
            cur[0] = tok
        else:
            cur[1].append(tok)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(name, parse_overrides(kv), world, s, steps, str(out))
            for name, kv in conds for world in WORLDS if only in (None, world) for s in seeds or [0]]
    with Pool(min(len(jobs), 8)) as pool:
        for msg in pool.imap_unordered(run_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
