"""GIFs of the two demo worlds as maps of family colour (tag), plus family-turnover numbers.

    .venv/bin/python analysis/tag_movie.py docs/demo_tags.gif docs/demo_tags_yearly.gif

Each map pixel block shows the mean inherited colour (tag) of the dots in that patch of
the world; brightness = how many dots are there (dark = empty). Families show up as
patches of one colour.

- the first GIF has a frame every FRAME_EVERY steps (the seasons are visible),
- the second has one frame per year, always at the same moment of the year
  (when the population peaks), so slow vs fast change of the family map stands out.

Also prints, per year: how far (median) a dot lives from where its ancestor lived one
year earlier, and what share of last year's dots left any living descendants.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kin_dots.core import Params, World
from kin_dots.core import genome as G
from kin_dots.core.kinship import _ancestor_index

PRESETS = [("gift to a neighbour", "configs/demo_sacrifice.yaml"),
           ("gift to a stranger", "configs/demo_sacrifice_strangers.yaml")]
SEED = 0
STEPS = 30000
FRAME_EVERY = 150
BINS = 75            # map resolution (world 300 -> 4 units per block)
FULL_DENSITY = 2     # dots per block that give full brightness
PEAK_PHASE = 300     # step within the 600-step year when the population peaks
PANEL = 330          # pixels per world in the GIF


def heatmap(w: World) -> np.ndarray:
    idx = np.flatnonzero(w.alive)
    img = np.zeros((BINS, BINS, 3))
    if idx.size == 0:
        return img.astype(np.uint8)
    bx = np.minimum((w.x[idx] / w.p.world_size * BINS).astype(int), BINS - 1)
    by = np.minimum((w.y[idx] / w.p.world_size * BINS).astype(int), BINS - 1)
    cnt = np.zeros((BINS, BINS))
    np.add.at(cnt, (by, bx), 1)
    tags = w.genome[idx][:, G.TAG]
    for c in range(3):
        np.add.at(img[..., c], (by, bx), tags[:, c])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt[..., None] > 0, img / cnt[..., None], 0)
    bright = np.minimum(cnt / FULL_DENSITY, 1.0)[..., None]
    rgb = 18 + (40 + 215 * mean - 18) * bright  # blend with the dark background
    return np.clip(rgb, 0, 255).astype(np.uint8)


def run(job):
    label, path = job
    w = World(Params.from_yaml(path).replace(seed=SEED))
    period = w.p.season_period
    frames, yearly, stats = [], [], []
    prev = None  # (ids, x, y) at last year's peak
    while w.step_count < STEPS and w.population > 0:
        w.step()
        if w.step_count % FRAME_EVERY == 0:
            frames.append((w.step_count, w.population, heatmap(w)))
        if w.step_count % period == PEAK_PHASE:
            a = np.flatnonzero(w.alive)
            ids, x, y = w.id[a], w.x[a], w.y[a]
            yearly.append((w.step_count, w.population, heatmap(w)))
            if prev is not None:
                pids, px, py = prev
                snap = np.full(w.next_id, -1, dtype=np.int64)
                snap[pids] = np.arange(pids.size)
                anc = _ancestor_index(ids, w.genealogy.parent, snap)
                ok = anc >= 0
                dx = np.abs(x[ok] - px[anc[ok]])
                dy = np.abs(y[ok] - py[anc[ok]])
                size = w.p.world_size
                dist = np.hypot(np.minimum(dx, size - dx), np.minimum(dy, size - dy))
                left_descendants = np.unique(anc[ok]).size / pids.size
                stats.append((w.step_count, float(np.median(dist)), left_descendants))
            prev = (ids, x, y)
    return label, frames, yearly, stats


def compose(results, which: int, title_fn, out: str, ms: int):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    n = min(len(r[which]) for r in results)
    gap, top = 14, 58
    W_ = 2 * PANEL + 3 * gap
    H_ = top + PANEL + gap
    images = []
    for f in range(n):
        im = Image.new("RGB", (W_, H_), (18, 18, 22))
        d = ImageDraw.Draw(im)
        for k, r in enumerate(results):
            step, pop, hm = r[which][f]
            x0 = gap + k * (PANEL + gap)
            color = (250, 210, 40) if k == 0 else (90, 220, 230)
            d.text((x0, 10), r[0], font=font, fill=color)
            d.text((x0, 33), title_fn(step) + f"   population {pop}", font=small, fill=(200, 200, 205))
            tile = Image.fromarray(hm, "RGB").resize((PANEL, PANEL), Image.NEAREST)
            im.paste(tile, (x0, top))
        images.append(im)
    images[0].save(out, save_all=True, append_images=images[1:], duration=ms, loop=0, optimize=True)
    print(f"saved {out}: {n} frames, {Path(out).stat().st_size / 1e6:.1f} MB")


def main(out_fast: str, out_yearly: str) -> None:
    with Pool(2) as pool:
        results = pool.map(run, PRESETS)
    compose(results, 1, lambda s: f"step {s}", out_fast, 60)
    compose(results, 2, lambda s: f"year {s // 600}", out_yearly, 220)
    for label, _, _, stats in results:
        st = np.array(stats)
        late = st[st[:, 0] > 15000]
        print(f"{label}: median distance to last-year ancestor {late[:, 1].mean():.1f}, "
              f"share of last year's dots with living descendants {late[:, 2].mean():.2f} "
              f"(years after step 15000)")
        np.savetxt(Path(out_yearly).with_suffix(f".{'left' if label == PRESETS[0][0] else 'right'}.csv"),
                   st, delimiter=",", header="step,median_dist_to_ancestor,share_with_descendants",
                   fmt="%.4f")


if __name__ == "__main__":
    main(*sys.argv[1:3])
