"""Data for the presentation in presentation/: chart series and family-map sprite sheets.

Reads the saved results of the analysis scripts and writes presentation/assets/data.js and
presentation/assets/family_*.png:

    .venv/bin/python analysis/deck_data.py

Needs results/demo_curves (sweep), results/sacrifice_genes, results/population.npz,
results/age_at_death and results/tag_movie_cache.pkl (see the README for the commands).
"""
import csv, glob, json, pickle, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
out = {}
r2 = lambda a, n=3: [None if (x != x) else round(float(x), n) for x in a]

# 1. sacrifice share of deaths, per window, 3 runs per world (ratio of counts)
sac = {}
for cond, key in (("sacrifice-local", "neighbour"), ("sacrifice-strangers", "stranger")):
    runs = []
    for f in sorted(glob.glob(f"results/demo_curves/{cond}_seed*.csv")):
        rows = list(csv.DictReader(open(f)))
        st = np.array([float(r["step"]) for r in rows]); s = np.array([float(r["total_sacrificed"]) for r in rows])
        d = np.array([float(r["total_deaths"]) for r in rows])
        ds, dd = np.diff(np.r_[0, s]), np.diff(np.r_[0, d])
        fr = np.where(ds + dd > 0, ds / np.maximum(ds + dd, 1), np.nan)
        runs.append({"step": [int(x) for x in st], "frac": r2(fr)})
    sac[key] = runs
out["sacrifice"] = sac

# 2. genes: a over time (snapshots), p(E) yearly (mean over runs)
G = {}
for cond in ("neighbour", "stranger"):
    rs = [dict(np.load(f)) for f in sorted(glob.glob(f"results/sacrifice_genes/{cond}_seed*.npz"))]
    L = min(r["step"].size for r in rs)
    G[cond] = {"step": [int(x) for x in rs[0]["step"][:L]],
               "a": [r2(r["a_mean"][:L], 2) for r in rs],
               "b": [r2(r["b_mean"][:L], 2) for r in rs]}
    keep = rs[0]["step"][:L] % 600 == 0
    P = np.mean([r["p_of_e"][:L] for r in rs], axis=0)[keep]
    G[cond]["pe_step"] = [int(x) for x in rs[0]["step"][:L][keep]]
    G[cond]["pe"] = [r2(row * 100, 4) for row in P]   # percent, E grid 0..20 step 0.5
out["genes"] = G

# 3. raw population, 3 runs each, every 10 steps -> keep every 20 steps
z = np.load("results/population.npz")
pop = {}
for cond in ("neighbour", "stranger"):
    pop[cond] = []
    for s in range(3):
        k = f"{cond}_s{s}_pop"
        if k in z:
            pop[cond].append([int(x) for x in z[k][::2]])
out["population"] = {"every": 20, **pop}

# 4. age at death by cause (pooled after step 20000), 10-step bins, as share of all deaths
A = {}
for cond in ("neighbour", "stranger"):
    rs = [dict(np.load(f)) for f in sorted(glob.glob(f"results/age_at_death/{cond}_seed*.npz"))]
    age = np.concatenate([r["age"][r["step"] >= 20000] for r in rs])
    cause = np.concatenate([r["cause"][r["step"] >= 20000] for r in rs])
    bins = np.arange(0, 310, 10)
    A[cond] = {"bins": bins.tolist(),
               "sacrifice": r2(np.histogram(age[cause == 0], bins)[0] / age.size, 4),
               "hunger": r2(np.histogram(age[cause == 1], bins)[0] / age.size, 4),
               "old": round(float(np.mean(cause == 2)), 3),
               "mean": round(float(age.mean()), 1), "median": float(np.median(age)),
               "share": {c: round(float(np.mean(cause == i)), 3) for i, c in enumerate(["sacrifice", "hunger", "old"])}}
out["age"] = A

# 5. family maps: yearly frames -> sprite sheets, hue histograms -> kymograph data
c = pickle.load(open("results/tag_movie_cache.pkl", "rb"))
fam = {}
for k, (label, fast, yearly, stats) in enumerate(c):
    frames = yearly[:150]
    cols = 15; rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * 75, cols * 75, 3), np.uint8)
    for i, (step, popn, hm, hist) in enumerate(frames):
        hm = np.asarray(hm)
        if hm.shape[0] != 75: hm = np.array(Image.fromarray(hm).resize((75, 75), Image.NEAREST))
        y, x = divmod(i, cols)
        sheet[y*75:(y+1)*75, x*75:(x+1)*75] = hm
    name = ["neighbour", "stranger"][k]
    Image.fromarray(sheet).save(f"presentation/assets/family_{name}.png", optimize=True)
    fam[name] = {"n": len(frames), "cols": cols, "pop": [int(f[1]) for f in frames],
                 "hist": [r2(np.asarray(f[3]), 3) for f in frames]}
out["families"] = fam
out["numbers"] = {"ancestor_dist": {"neighbour": 11.6, "stranger": 21.1}, "r": 0.75,
                  "donor_energy": 9.6, "mean_energy": 12.5}
js = "window.DECK_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n"
open("presentation/assets/data.js", "w").write(js)
print(len(js) // 1024, "KB data.js")
