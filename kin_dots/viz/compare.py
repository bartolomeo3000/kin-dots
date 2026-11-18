"""Side-by-side viewer: two worlds that differ in one rule, same seed, one window.

    python -m kin_dots.viz.compare --left configs/kin_altruism.yaml \\
        --right configs/kin_altruism_random_recipient.yaml --metric share_rate

Both worlds advance in lockstep. Below them, one plot shows the chosen metric
for both (left = yellow, right = cyan): its mean over the last full season.

Keys: SPACE pause, UP/DOWN max steps per frame, C colour mode, F food layer, G energy gifts
(white lines donor -> recipient), H side panels, R restart, ESC quit.

Side panels (next to each world): the dots' family colours reduced to their hue, now
(histogram) and over the whole run (time goes down: lasting families are vertical bands).
Colour mode 'family map (yearly)' replaces the dots by the mean family colour of each
patch of the world, refreshed once a year when the population peaks.

The window fits the screen (or use --view). Each frame simulates as many steps as fit
into --budget-ms (and at most the current max steps per frame), so the window stays
responsive however slow the simulation is.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from kin_dots.core import Params, World  # noqa: E402
from kin_dots.core import genome as G  # noqa: E402

VIEW = 540          # world view size in pixels (recomputed to fit the screen in main())
GAP = 16
PLOT_H = 170
TEXT_H = 72
INFO_H = 24         # help line under the plots
BG = (18, 18, 22)
FG = (225, 225, 230)
MUTED = (140, 140, 150)
LEFT_C = (250, 210, 40)
RIGHT_C = (90, 220, 230)
RECORD_EVERY = 10   # steps between plotted samples (the whole run is always shown)
METRIC_TITLES = {  # readable names for the plot and the header
    "sacrifice_frac": "deaths that are self-sacrifice",
    "share_rate": "encounters with an energy gift",
}
COLOR_MODES = ("p(give)", "energy", "tag", "family map (yearly)")  # tag = inherited colour (families)
SIDE = 150          # width of the side panel (hue histogram + hue history)
HUE_BINS = 60
HUE_EVERY = 100     # steps between rows of the hue history
HIST_H = 70
HIST_MAX = 0.15     # share of dots that fills a histogram bar
FAMILY_BINS = 75    # resolution of the family map
FULL_DENSITY = 2    # dots per family-map block that give full brightness
PEAK_PHASE = 300    # step within the year when the family map is refreshed (population peak)


def _hue(rgb: np.ndarray) -> np.ndarray:
    """Hue in [0, 1) of RGB rows in [0, 1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    d = np.where(mx > mn, mx - mn, 1.0)
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return np.where(mx > mn, h / 6.0, 0.0)


def _hsv_row(h: np.ndarray, s=0.85, v=1.0) -> np.ndarray:
    """(n,) hues -> (n, 3) uint8 RGB at the given saturation and value."""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = np.stack([np.stack(c, axis=1) for c in
                      ((np.full_like(h, v), t, np.full_like(h, p)), (q, np.full_like(h, v), np.full_like(h, p)),
                       (np.full_like(h, p), np.full_like(h, v), t), (np.full_like(h, p), q, np.full_like(h, v)),
                       (t, np.full_like(h, p), np.full_like(h, v)), (np.full_like(h, v), np.full_like(h, p), q))])
    return (table[i, np.arange(h.size)] * 255).astype(np.uint8)


HUE_RGB = _hsv_row((np.arange(HUE_BINS) + 0.5) / HUE_BINS).astype(float)


def hue_hist(w: World) -> np.ndarray:
    tags = w.genome[w.alive][:, G.TAG]
    if tags.size == 0:
        return np.zeros(HUE_BINS)
    h = np.bincount(np.minimum((_hue(np.clip(tags, 0, 1)) * HUE_BINS).astype(int), HUE_BINS - 1),
                    minlength=HUE_BINS)
    return h / h.sum()


def family_map(w: World) -> np.ndarray:
    """(FAMILY_BINS, FAMILY_BINS, 3) uint8, indexed [x, y]: mean family colour per patch,
    brightness = number of dots there."""
    idx = np.flatnonzero(w.alive)
    img = np.zeros((FAMILY_BINS, FAMILY_BINS, 3))
    if idx.size == 0:
        return np.full((FAMILY_BINS, FAMILY_BINS, 3), BG, dtype=np.uint8)
    bx = np.minimum((w.x[idx] / w.p.world_size * FAMILY_BINS).astype(int), FAMILY_BINS - 1)
    by = np.minimum((w.y[idx] / w.p.world_size * FAMILY_BINS).astype(int), FAMILY_BINS - 1)
    cnt = np.zeros((FAMILY_BINS, FAMILY_BINS))
    np.add.at(cnt, (bx, by), 1)
    tags = w.genome[idx][:, G.TAG]
    for c in range(3):
        np.add.at(img[..., c], (bx, by), tags[:, c])
    mean = img / np.maximum(cnt, 1)[..., None]
    bright = np.minimum(cnt / FULL_DENSITY, 1.0)[..., None]
    return np.clip(BG[0] + (40 + 215 * mean - BG[0]) * bright, 0, 255).astype(np.uint8)


def _ramp(t):
    t = np.clip(t, 0.0, 1.0)[:, None]
    lo = np.array([40, 90, 220])
    hi = np.array([250, 210, 40])
    return (lo + (hi - lo) * t).astype(np.uint8)


def _metric(w: World, metric: str) -> float:
    """Cheap per-step metrics from the step ledger; anything else via the full census."""
    L = w.last
    if metric == "sacrifice_frac":
        died = L.get("deaths", 0) + L.get("killed", 0) + L.get("predated", 0) + L.get("sacrifices", 0)
        return L.get("sacrifices", 0) / died if died else float("nan")
    if metric == "share_rate":
        enc = L.get("encounters", 0)
        return L.get("shares", 0) / enc if enc else float("nan")
    return float(w.census().get(metric, float("nan")))


class Side:
    def __init__(self, params: Params, label: str):
        self.params, self.label = params, label
        self.reset()

    def reset(self):
        self.world = World(self.params)
        self.hist = []
        self.pop_hist = []  # raw population every RECORD_EVERY steps (not averaged)
        self.hue_rows = []  # hue histogram every HUE_EVERY steps
        self.family = None  # last yearly family map
        self.family_step = 0
        # plotted value = mean of the metric over the last full season (or 200 steps)
        self.window = deque(maxlen=self.params.season_period or 200)

    def step(self, metric: str):
        w = self.world
        if w.population == 0:
            return
        w.step()
        v = _metric(w, metric)
        if v == v:
            self.window.append(v)
        if w.step_count % HUE_EVERY == 0:
            self.hue_rows.append(hue_hist(w))
        period = self.params.season_period or 600
        if w.step_count % period == PEAK_PHASE % period or self.family is None:
            self.family, self.family_step = family_map(w), w.step_count
        if w.step_count % RECORD_EVERY == 0:
            self.pop_hist.append(w.population)
            if len(self.window) == self.window.maxlen:
                self.hist.append(float(np.mean(self.window)))


def draw_series(screen, font, x0, y0, w, h, series, title, x_len=None):
    """Line plot of several series [(values, colour)] sharing a y axis from 0, whole history
    squeezed into the width. x_len: number of samples the x axis spans (default: longest)."""
    pygame.draw.rect(screen, (40, 40, 48), (x0, y0, w, h), 1)
    screen.blit(font.render(title, True, MUTED), (x0 + 6, y0 + 4))
    vals = [v for vs, _ in series for v in vs if v == v]
    if not vals:
        return
    hi = max(vals) * 1.1 or 1.0
    n_max = x_len or max(len(vs) for vs, _ in series)
    for vs, col in series:
        if len(vs) < 2:
            continue
        v = np.asarray(vs, dtype=float)
        stride = max(1, len(v) // w)
        v = v[::stride]
        xs = np.arange(len(v)) * stride * (w - 1) / max(n_max - 1, 1)
        pts = [(x0 + int(x), y0 + h - 4 - int(val / hi * (h - 28))) for x, val in zip(xs, v) if val == val]
        if len(pts) >= 2:
            pygame.draw.lines(screen, col, False, pts, 2)
    screen.blit(font.render(f"{hi / 1.1:.3g}", True, MUTED), (x0 + w - 60, y0 + 4))


def draw_world(screen, side: Side, x0: int, y0: int, mode: str, show_food: bool, show_gifts: bool = True):
    w = side.world
    n = w.p.grid_cells
    cell = np.arange(VIEW) * n // VIEW  # view pixel -> food cell
    if show_food:
        t = np.sqrt(w.food / w.p.food_capacity)[..., None]
        rgb = (np.array(BG) * (1 - t) + np.array([30, 95, 45]) * t).astype(np.uint8)
        canvas = rgb[cell][:, cell].copy()  # indexed [x, y] like pygame.surfarray
    else:
        canvas = np.full((VIEW, VIEW, 3), 10, dtype=np.uint8)
    if mode.startswith("family map"):
        fm = side.family if side.family is not None else family_map(w)
        surf = pygame.transform.scale(pygame.surfarray.make_surface(fm), (VIEW, VIEW))
        screen.blit(surf, (x0, y0))
        return
    idx = np.flatnonzero(w.alive)
    if idx.size == 0:
        screen.blit(pygame.surfarray.make_surface(canvas), (x0, y0))
        return
    g = w.genome[idx]
    if mode == "tag":
        cols = (40 + 215 * g[:, G.TAG]).astype(np.uint8)
    elif mode == "energy":
        cols = _ramp(w.energy[idx] / w.p.birth_threshold)
    else:
        # per-step probability to give, at the agent's CURRENT energy, using only the inputs
        # the decision can see; log scale 1e-4 .. 1e-1 (blue = never, yellow = often)
        p = w.p
        wv = g[:, G.SHARE_W]
        z = wv[:, 0].copy()
        if not p.tag_blind_share:
            z += wv[:, 1]
        if p.share_sees_self:
            z += wv[:, 2] * np.minimum(w.energy[idx] / p.birth_threshold, 2.0)
        if p.share_sees_other:
            z += wv[:, 3] * 0.6
        prob = 1.0 / (1.0 + np.exp(-z))
        cols = _ramp((np.log10(np.maximum(prob, 1e-6)) + 4.0) / 3.0)
    s = VIEW / w.p.world_size
    # all agents at once: 2x2-pixel squares written into the pixel array of the view
    px = np.minimum((w.x[idx] * s).astype(int), VIEW - 2)
    py = np.minimum((w.y[idx] * s).astype(int), VIEW - 2)
    for dx in (0, 1):
        for dy in (0, 1):
            canvas[px + dx, py + dy] = cols
    screen.blit(pygame.surfarray.make_surface(canvas), (x0, y0))
    if show_gifts and len(w.last_gifts):
        half = w.p.world_size / 2
        for xa, ya, xb, yb in w.last_gifts.tolist():
            if abs(xa - xb) < half and abs(ya - yb) < half:
                pygame.draw.line(screen, (255, 255, 255), (x0 + int(xa * s), y0 + int(ya * s)),
                                 (x0 + int(xb * s), y0 + int(yb * s)), 1)


def draw_side(screen, font, side: Side, x0: int, y0: int, h: int):
    """Hue histogram now (top) and hue history over the whole run (time goes down)."""
    screen.blit(font.render("hue now", True, MUTED), (x0, y0))
    hy = y0 + 18
    pygame.draw.rect(screen, (24, 24, 30), (x0, hy, SIDE, HIST_H))
    hist = hue_hist(side.world)
    bw = SIDE / HUE_BINS
    for b, share in enumerate(hist):
        bh = int(min(share / HIST_MAX, 1.0) * (HIST_H - 2))
        if bh > 0:
            pygame.draw.rect(screen, HUE_RGB[b].astype(int).tolist(),
                             (x0 + int(b * bw), hy + HIST_H - bh, max(1, int(bw)), bh))
    ky = hy + HIST_H + 24
    screen.blit(font.render("hue over time", True, MUTED), (x0, ky - 20))
    kh = h - (ky - y0)
    pygame.draw.rect(screen, (24, 24, 30), (x0, ky, SIDE, kh))
    if len(side.hue_rows) >= 2:
        rows = np.asarray(side.hue_rows)                              # (time, bins)
        bright = np.minimum(np.sqrt(rows / 0.15), 1.0)[..., None]
        rgb = (24 + (HUE_RGB[None, :, :] - 24) * bright).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))  # [x=hue, y=time]
        screen.blit(pygame.transform.scale(surf, (SIDE, kh)), (x0, ky))
        steps = len(side.hue_rows) * HUE_EVERY
        screen.blit(font.render(f"0..{steps}", True, MUTED), (x0 + 2, ky + kh - 18))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--left-label", default=None)
    ap.add_argument("--right-label", default=None)
    ap.add_argument("--metric", default="share_rate", help="census key to plot for both worlds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=int, default=64, help="max simulation steps per frame")
    ap.add_argument("--budget-ms", type=float, default=50.0,
                    help="max time per frame spent simulating (keeps the window responsive)")
    ap.add_argument("--view", type=int, default=0, help="world view size in pixels (0 = fit the screen)")
    ap.add_argument("--color", choices=COLOR_MODES, default=COLOR_MODES[0])
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--screenshot", type=Path)
    args = ap.parse_args(argv)

    sides = [Side(Params.from_yaml(args.left).replace(seed=args.seed),
                  args.left_label or args.left.stem),
             Side(Params.from_yaml(args.right).replace(seed=args.seed),
                  args.right_label or args.right.stem)]
    pygame.init()
    pygame.display.set_caption("kin-dots: compare")
    global VIEW
    if args.view > 0:
        VIEW = args.view
    else:
        info = pygame.display.Info()
        if info.current_w > 0 and info.current_h > 0:
            avail_h = info.current_h - 130  # menu bar, title bar, dock
            VIEW = max(200, min(560, avail_h - (3 * GAP + TEXT_H + PLOT_H + INFO_H),
                                (info.current_w - 60 - 5 * GAP - 2 * SIDE) // 2))
    show_side = True

    def layout():
        block = VIEW + (GAP + SIDE if show_side else 0)
        return 2 * block + 3 * GAP, block

    W_, BLOCK = layout()
    H_ = GAP + TEXT_H + VIEW + GAP + PLOT_H + INFO_H + GAP
    screen = pygame.display.set_mode((W_, H_))
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
    big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 18, bold=True)
    clock = pygame.time.Clock()
    paused, speed, mode, show_food, frame = False, max(1, args.speed), COLOR_MODES.index(args.color), True, 0
    show_gifts = True
    sim_rate = deque(maxlen=30)  # (steps, seconds) of recent frames, for the steps/s readout

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_UP:
                    speed = min(speed * 2, 128)
                elif ev.key == pygame.K_DOWN:
                    speed = max(speed // 2, 1)
                elif ev.key == pygame.K_c:
                    mode = (mode + 1) % len(COLOR_MODES)
                elif ev.key == pygame.K_f:
                    show_food = not show_food
                elif ev.key == pygame.K_g:
                    show_gifts = not show_gifts
                elif ev.key == pygame.K_h:
                    show_side = not show_side
                    W_, BLOCK = layout()
                    screen = pygame.display.set_mode((W_, H_))
                elif ev.key == pygame.K_r:
                    for s in sides:
                        s.reset()
        if not paused:
            t0 = time.perf_counter()
            done = 0
            while done < speed and (time.perf_counter() - t0) * 1000 < args.budget_ms:
                for s in sides:
                    s.step(args.metric)
                done += 1
            sim_rate.append((done, time.perf_counter() - t0))

        screen.fill(BG)
        for k, s in enumerate(sides):
            x0 = GAP + k * (BLOCK + GAP)
            w = s.world
            col = LEFT_C if k == 0 else RIGHT_C
            screen.blit(big.render(s.label, True, col), (x0, GAP))
            cur = s.hist[-1] if s.hist else float("nan")
            screen.blit(font.render(f"step {w.step_count}   population {w.population}", True, FG),
                        (x0, GAP + 26))
            pretty = METRIC_TITLES.get(args.metric, args.metric)
            val = f"{cur:.0%}" if args.metric in METRIC_TITLES and cur == cur else f"{cur:.3f}"
            screen.blit(font.render(f"{pretty}: {val}", True, col), (x0, GAP + 46))
            draw_world(screen, s, x0, GAP + TEXT_H, COLOR_MODES[mode], show_food, show_gifts)
            if COLOR_MODES[mode].startswith("family map"):
                screen.blit(font.render(f"map from step {s.family_step}", True, MUTED),
                            (x0 + 6, GAP + TEXT_H + VIEW - 20))
            if show_side:
                draw_side(screen, font, s, x0 + VIEW + GAP, GAP + TEXT_H, VIEW)

        # plots: chosen metric (season mean) and raw population, both for the whole run
        py0 = GAP + TEXT_H + VIEW + GAP
        half = (W_ - 3 * GAP) // 2
        n_pop = max(len(s.pop_hist) for s in sides)
        steps = n_pop * RECORD_EVERY
        cols = (LEFT_C, RIGHT_C)
        # the metric starts one season late: pad it so both plots share the step axis
        draw_series(screen, font, GAP, py0, half, PLOT_H,
                    [([float("nan")] * (len(s.pop_hist) - len(s.hist)) + s.hist, c)
                     for s, c in zip(sides, cols)],
                    f"{METRIC_TITLES.get(args.metric, args.metric)} (mean over a year)", x_len=n_pop)
        draw_series(screen, font, 2 * GAP + half, py0, half, PLOT_H,
                    [(s.pop_hist, c) for s, c in zip(sides, cols)],
                    f"population, step 0..{steps}")
        st = sum(a for a, _ in sim_rate)
        rate = st / max(sum(b for _, b in sim_rate), 1e-9) if st else 0.0
        info = (f"~{rate:.0f} steps/s (max {speed}/frame)  colour: {COLOR_MODES[mode]}  |  "
                f"SPACE pause  UP/DOWN  C colour  F food  G gifts  H side panels  R")
        screen.blit(font.render(info, True, MUTED), (GAP + 6, py0 + PLOT_H + 4))
        pygame.display.flip()
        clock.tick(60)
        frame += 1
        if args.frames and frame >= args.frames:
            break
    if args.screenshot:
        pygame.image.save(screen, str(args.screenshot))
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
