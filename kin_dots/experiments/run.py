"""Headless runner: simulate without graphics and write a census time series.

Examples:
    python -m kin_dots.experiments.run --steps 5000
    python -m kin_dots.experiments.run --config configs/default.yaml --seed 3 --out results/run.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from kin_dots.core import Params, World


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML file with Params overrides")
    ap.add_argument("--seed", type=int, help="override the seed from the config")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--every", type=int, default=100, help="record a census every N steps")
    ap.add_argument("--out", type=Path, help="CSV output path (default: print to stdout)")
    ap.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    args = ap.parse_args(argv)

    params = Params.from_yaml(args.config) if args.config else Params()
    if args.seed is not None:
        params = params.replace(seed=args.seed)

    world = World(params)
    rows = [world.census()]
    t0 = time.perf_counter()
    for _ in range(args.steps):
        world.step()
        if world.step_count % args.every == 0:
            row = world.census()
            rows.append(row)
            if not args.quiet:
                print(f"step {row['step']:>7}  pop {row['population']:>6}  "
                      f"food {row['total_food']:>9.0f}  "
                      f"harvest {row.get('mean_harvest', float('nan')):.3f}", file=sys.stderr)
        if world.population == 0:
            print(f"extinct at step {world.step_count}", file=sys.stderr)
            rows.append(world.census())
            break
    dt = time.perf_counter() - t0
    if not args.quiet:
        print(f"{world.step_count} steps in {dt:.1f}s ({world.step_count / max(dt, 1e-9):.0f} steps/s)",
              file=sys.stderr)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        f = open(args.out, "w", newline="")
    else:
        f = sys.stdout
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    if args.out:
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
