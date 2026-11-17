"""Run several conditions x seeds in parallel and compare how behaviour evolves.

Each condition is a name plus Params overrides:

    python -m kin_dots.experiments.sweep --steps 30000 --seeds 0 1 \\
        --cond base \\
        --cond seasons season_period=600 season_amplitude=0.9 \\
        --cond seasons-global season_period=600 season_amplitude=0.9 dispersal_sd=-1

Prints a table per condition (values averaged over each reporting window) and,
with --out DIR, writes one CSV per run.
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from kin_dots.core import Params, World
from kin_dots.core.kinship import PriceTracker, clans_and_cheaters, kinship_report

RATE_KEYS = ("share_rate", "attack_rate", "sim_encounter", "sim_share", "sim_attack",
             "sim_random", "assortment", "donor_energy", "recipient_energy", "alarm_rate", "sim_warned")
SHOW = ("population", "share_rate", "attack_rate", "mean_share_w0", "mean_share_w1",
        "mean_share_w2", "mean_share_w3", "assortment", "sim_encounter", "sim_share",
        "donor_energy", "recipient_energy", "mean_energy")
KIN_SHOW = ("r_met", "r_share", "r_attack", "close_kin_share", "close_kin_met", "share_to_child",
            "hamilton_rb_minus_c", "sel_share_w0", "sel_share_w1", "sel_share_w2", "sel_share_w3",
            "sel_attack_w1", "n_clans", "frac_cheaters")
PRED_SHOW = ("population", "alarm_rate", "mean_alarm_w0", "mean_alarm_w1", "mean_alarm_w2",
             "mean_alarm_w3", "r_met", "r_alarm", "close_kin_alarm", "sel_alarm_w0", "sel_alarm_w1",
             "predated_win", "share_rate")
PRICE_WINDOW = 600  # steps over which the Price decomposition is measured, ending at each report


def _parse_value(v: str):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def run_one(job):
    name, overrides, seed, steps, windows, out_dir = job
    w = World(Params(**overrides).replace(seed=seed))
    every = max(1, steps // windows)
    if w.p.season_period > 0:  # report at the same season phase every time
        every = max(w.p.season_period, every // w.p.season_period * w.p.season_period)
    rows, acc, deaths0, pred0 = [], {}, (0, 0, 0), 0
    price = None
    for s in range(1, steps + 1):
        if s % every == (every - PRICE_WINDOW) % every and w.population > 0:
            price = PriceTracker(w)
        w.step()
        c = w.census()
        for k in RATE_KEYS:
            if c[k] == c[k]:
                acc.setdefault(k, []).append(c[k])
        if s % every == 0 or w.population == 0:
            row = dict(c)
            for k in RATE_KEYS:
                row[k] = float(np.mean(acc[k])) if acc.get(k) else float("nan")
            d = (c["total_starved"], c["total_deaths"] - c["total_starved"], c["total_killed"])
            row["starved_win"], row["old_win"], row["killed_win"] = (a - b for a, b in zip(d, deaths0))
            if w.population > 0:
                row.update(kinship_report(w))
                row.update(clans_and_cheaters(w))
                if price is not None:
                    row.update(price.finish(w))
                    price = None
            row["predated_win"] = c["total_predated"] - pred0
            deaths0, acc, pred0 = d, {}, c["total_predated"]
            rows.append(row)
        if w.population == 0:
            break
    if out_dir:
        path = Path(out_dir) / f"{name}_seed{seed}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return name, seed, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", nargs="+", action="append", required=True, metavar="NAME [KEY=VAL ...]")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--windows", type=int, default=5, help="number of reporting windows")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=Path, help="directory for per-run CSVs")
    ap.add_argument("--table", choices=("behaviour", "kin", "pred", "both"), default="both",
                    help="which summary table(s) to print")
    args = ap.parse_args(argv)

    jobs = []
    for cond in args.cond:
        name, kvs = cond[0], [t for kv in cond[1:] for t in kv.split()]
        overrides = {}
        for kv in kvs:
            k, v = kv.split("=", 1)
            overrides[k] = _parse_value(v)
        defaults = Params()
        for k, v in overrides.items():  # fail fast on typos and wrong types
            if not hasattr(defaults, k):
                raise SystemExit(f"unknown parameter: {k}")
            ref = getattr(defaults, k)
            if isinstance(ref, (int, float)) and not isinstance(v, (int, float)) or \
                    isinstance(ref, bool) and not isinstance(v, bool):
                raise SystemExit(f"bad value for {k}: {v!r}")
        for seed in args.seeds:
            jobs.append((name, overrides, seed, args.steps, args.windows, args.out))

    with Pool(min(args.procs, len(jobs))) as pool:
        results = pool.map(run_one, jobs)

    def table(cols, extra):
        print("   step " + " ".join(f"{k.replace('mean_', '')[:11]:>11}" for k in cols) + extra[0])
        for r in rows:
            vals = " ".join(f"{float(r.get(k, float('nan'))):11.3f}" for k in cols)
            print(f"{r['step']:7d} {vals}" + extra[1](r))

    for name, seed, rows in results:
        print(f"== {name}  seed {seed}")
        if args.table in ("behaviour", "both"):
            table(SHOW, ("  starv/old/kill", lambda r: f"  {r['starved_win']}/{r['old_win']}/{r['killed_win']}"))
        if args.table in ("kin", "both"):
            table(KIN_SHOW, ("", lambda r: ""))
        if args.table == "pred":
            table(PRED_SHOW, ("", lambda r: ""))
        if rows and rows[-1]["population"] == 0:
            print("   EXTINCT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
