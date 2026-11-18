"""GIFs of the two demo worlds as maps of family colour (tag), plus family-turnover numbers.

    .venv/bin/python analysis/tag_movie.py docs/demo_tags.gif docs/demo_tags_yearly.gif [steps]

Each map pixel block shows the mean inherited colour (tag) of the dots in that patch of
the world; brightness = how many dots are there (dark = empty). Families show up as
patches of one colour.

- the first GIF has a frame every FRAME_EVERY steps (the seasons are visible),
- the second has one frame per year, always at the same moment of the year
  (when the population peaks), so slow vs fast change of the family map stands out.

Under each map: a histogram of the dots' colour hue right now (bars coloured by hue,
height = share of dots) and the hue history so far (x = time, y = hue, brightness =
share of dots): lasting families are horizontal bands.

Run data is cached in results/tag_movie_cache.pkl; pass `cache` instead of a step count
to re-render the GIFs without simulating again.

Also prints, per year: how far (median) a dot lives from where its ancestor lived one
year earlier, and what share of last year's dots left any living descendants.
"""
from __future__ import annotations

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
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
HUE_BINS = 60        # resolution of the hue histogram / history
HIST_H = 70          # height of the hue histogram strip
HIST_MAX = 0.15      # share of dots that fills a histogram bar (higher is clipped)
KYMO_H = 90          # height of the hue-history strip
CACHE = "results/tag_movie_cache.pkl"
HUE_RGB = (hsv_to_rgb(np.stack([(np.arange(HUE_BINS) + 0.5) / HUE_BINS,
                                np.full(HUE_BINS, 0.85), np.ones(HUE_BINS)], axis=1)) * 255)


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


def hue_hist(w: World) -> np.ndarray:
    """Share of living dots per hue bin of their colour tag."""
    tags = w.genome[w.alive][:, G.TAG]
    if tags.size == 0:
        return np.zeros(HUE_BINS)
    hue = rgb_to_hsv(np.clip(tags, 0, 1))[:, 0]
    h = np.bincount(np.minimum((hue * HUE_BINS).astype(int), HUE_BINS - 1), minlength=HUE_BINS)
    return h / h.sum()


def run(job):
    label, path, steps = job
    w = World(Params.from_yaml(path).replace(seed=SEED))
    period = w.p.season_period
    frames, yearly, stats = [], [], []
    prev = None  # (ids, x, y) at last year's peak
    while w.step_count < steps and w.population > 0:
        w.step()
        if w.step_count % FRAME_EVERY == 0:
            frames.append((w.step_count, w.population, heatmap(w), hue_hist(w)))
        if w.step_count % period == PEAK_PHASE:
            a = np.flatnonzero(w.alive)
            ids, x, y = w.id[a], w.x[a], w.y[a]
            yearly.append((w.step_count, w.population, heatmap(w), hue_hist(w)))
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


def _hist_strip(hist: np.ndarray) -> Image.Image:
    im = Image.new("RGB", (PANEL, HIST_H), (24, 24, 30))
    d = ImageDraw.Draw(im)
    bw = PANEL / HUE_BINS
    for b, share in enumerate(hist):
        h = int(min(share / HIST_MAX, 1.0) * (HIST_H - 4))
        if h > 0:
            d.rectangle([int(b * bw), HIST_H - h, int((b + 1) * bw) - 1, HIST_H - 1],
                        fill=tuple(int(c) for c in HUE_RGB[b]))
    return im


def _kymograph(hists: np.ndarray, upto: int) -> Image.Image:
    """hists: (n_frames, HUE_BINS). Columns after `upto` stay dark."""
    n = hists.shape[0]
    bright = np.minimum(np.sqrt(hists.T / 0.15), 1.0)[..., None]  # (bins, n, 1)
    rgb = 24 + (HUE_RGB[:, None, :] - 24) * bright
    rgb[:, upto + 1:] = 24
    img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    return img.resize((PANEL, KYMO_H), Image.NEAREST)


def compose(results, which: int, title_fn, out: str, ms: int):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    n = min(len(r[which]) for r in results)
    gap, top, label_h = 14, 58, 18
    W_ = 2 * PANEL + 3 * gap
    y_hist = top + PANEL + gap + label_h
    y_kymo = y_hist + HIST_H + gap + label_h
    H_ = y_kymo + KYMO_H + gap
    all_hists = [np.array([fr[3] for fr in r[which][:n]]) for r in results]
    images = []
    for f in range(n):
        im = Image.new("RGB", (W_, H_), (18, 18, 22))
        d = ImageDraw.Draw(im)
        for k, r in enumerate(results):
            step, pop, hm, hist = r[which][f]
            x0 = gap + k * (PANEL + gap)
            color = (250, 210, 40) if k == 0 else (90, 220, 230)
            d.text((x0, 10), r[0], font=font, fill=color)
            d.text((x0, 33), title_fn(step) + f"   population {pop}", font=small, fill=(200, 200, 205))
            im.paste(Image.fromarray(hm, "RGB").resize((PANEL, PANEL), Image.NEAREST), (x0, top))
            d.text((x0, y_hist - label_h), f"dots by colour hue now (bar full = {HIST_MAX:.0%})",
                   font=small, fill=(150, 150, 160))
            im.paste(_hist_strip(hist), (x0, y_hist))
            d.text((x0, y_kymo - label_h), "hue over time (bands = lasting families)",
                   font=small, fill=(150, 150, 160))
            im.paste(_kymograph(all_hists[k], f), (x0, y_kymo))
        images.append(im)
    images[0].save(out, save_all=True, append_images=images[1:], duration=ms, loop=0, optimize=True)
    print(f"saved {out}: {n} frames, {Path(out).stat().st_size / 1e6:.1f} MB")


def main(out_fast: str, out_yearly: str, steps: str = "") -> None:
    if steps == "cache":
        with open(CACHE, "rb") as f:
            results = pickle.load(f)
    else:
        n = int(steps) if steps else STEPS
        with Pool(2) as pool:
            results = pool.map(run, [(label, path, n) for label, path in PRESETS])
        Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(results, f)
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
    main(*sys.argv[1:4])
