"""Run the demo presets and record the raw population (and sacrifices) every few steps.

    .venv/bin/python analysis/population_runs.py results/population.npz

Fast: no kinship metrics, only the step ledger. Output: an .npz with, per run,
arrays `<cond>_s<seed>_step`, `_pop` and `_sacrifice_frac` (per recording interval).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

from kin_dots.core import Params, World

PRESETS = {
    "neighbour": "configs/demo_sacrifice.yaml",
    "stranger": "configs/demo_sacrifice_strangers.yaml",
}
SEEDS = (0, 1, 2)
STEPS = 30000
EVERY = 10


def run(job):
    cond, seed = job
    w = World(Params.from_yaml(PRESETS[cond]).replace(seed=seed))
    steps, pops, fracs = [], [], []
    sac = died = 0.0
    for _ in range(STEPS):
        w.step()
        L = w.last
        sac += L.get("sacrifices", 0.0)
        died += L.get("deaths", 0) + L.get("sacrifices", 0.0)
        if w.step_count % EVERY == 0:
            steps.append(w.step_count)
            pops.append(w.population)
            fracs.append(sac / died if died else np.nan)
            sac = died = 0.0
        if w.population == 0:
            break
    return cond, seed, np.array(steps), np.array(pops), np.array(fracs)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/population.npz"
    jobs = [(c, s) for c in PRESETS for s in SEEDS]
    data = {}
    with Pool(len(jobs)) as pool:
        for cond, seed, st, pop, fr in pool.imap_unordered(run, jobs):
            key = f"{cond}_s{seed}"
            data[f"{key}_step"], data[f"{key}_pop"], data[f"{key}_sacrifice_frac"] = st, pop, fr
            print(f"done {key}: {len(st)} samples, final population {pop[-1] if len(pop) else 0}", flush=True)
    np.savez_compressed(out, **data)
    print("saved", out)
