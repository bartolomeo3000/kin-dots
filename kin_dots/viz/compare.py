"""Side-by-side viewer: two worlds that differ in one rule, same seed, one window.

    python -m kin_dots.viz.compare --left configs/kin_altruism.yaml \\
        --right configs/kin_altruism_random_recipient.yaml --metric share_rate

Both worlds advance in lockstep. Below them, one plot shows the chosen metric
for both (left = yellow, right = cyan): its mean over the last full season.

Keys: SPACE pause, UP/DOWN speed, C colour mode, F food layer, G energy gifts (white
lines donor -> recipient), R restart, ESC quit.
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from kin_dots.core import Params, World  # noqa: E402
from kin_dots.core import genome as G  # noqa: E402

VIEW = 540
GAP = 20
PLOT_H = 200
TEXT_H = 92
BG = (18, 18, 22)
FG = (225, 225, 230)
MUTED = (140, 140, 150)
LEFT_C = (250, 210, 40)
RIGHT_C = (90, 220, 230)
RECORD_EVERY = 10   # steps between plotted samples (the whole run is always shown)
COLOR_MODES = ("p(give)", "energy", "tag", "p(alarm|kin)", "garden")


def _ramp(t):
    t = np.clip(t, 0.0, 1.0)[:, None]
    lo = np.array([40, 90, 220])
    hi = np.array([250, 210, 40])
    return (lo + (hi - lo) * t).astype(np.uint8)


class Side:
    def __init__(self, params: Params, label: str):
        self.params, self.label = params, label
        self.reset()

    def reset(self):
        self.world = World(self.params)
        self.hist = []
        # plotted value = mean of the metric over the last full season (or 200 steps)
        self.window = deque(maxlen=self.params.season_period or 200)

    def step(self, metric: str):
        w = self.world
        if w.population == 0:
            return
        w.step()
        v = w.census().get(metric, float("nan"))
        if v == v:
            self.window.append(v)
        if len(self.window) == self.window.maxlen and w.step_count % RECORD_EVERY == 0:
            self.hist.append(float(np.mean(self.window)))


def draw_world(screen, side: Side, x0: int, y0: int, mode: str, show_food: bool, show_gifts: bool = True):
    w = side.world
    if show_food:
        t = np.sqrt(w.food / w.p.food_capacity)[..., None]
        rgb = (np.array(BG) * (1 - t) + np.array([30, 95, 45]) * t).astype(np.uint8)
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(rgb), (VIEW, VIEW)), (x0, y0))
    else:
        pygame.draw.rect(screen, (10, 10, 12), (x0, y0, VIEW, VIEW))
    idx = np.flatnonzero(w.alive)
    if idx.size == 0:
        return
    g = w.genome[idx]
    if mode == "tag":
        cols = (40 + 215 * g[:, G.TAG]).astype(np.uint8)
    elif mode == "energy":
        cols = _ramp(w.energy[idx] / w.p.birth_threshold)
    elif mode == "p(alarm|kin)":
        cols = _ramp(G.propensity(g, G.ALARM_W, 1.0, 0.5, 0.3))
    elif mode == "garden":
        cols = _ramp(g[:, G.GARDEN])
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
    px = (w.x[idx] * s).astype(int) + x0
    py = (w.y[idx] * s).astype(int) + y0
    circle = pygame.draw.circle
    for x, y, c in zip(px.tolist(), py.tolist(), cols.tolist()):
        circle(screen, c, (x, y), 2)
    if show_gifts and len(w.last_gifts):
        half = w.p.world_size / 2
        for xa, ya, xb, yb in w.last_gifts.tolist():
            if abs(xa - xb) < half and abs(ya - yb) < half:
                pygame.draw.line(screen, (255, 255, 255), (x0 + int(xa * s), y0 + int(ya * s)),
                                 (x0 + int(xb * s), y0 + int(yb * s)), 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--left-label", default=None)
    ap.add_argument("--right-label", default=None)
    ap.add_argument("--metric", default="share_rate", help="census key to plot for both worlds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=int, default=8)
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
    W_ = 2 * VIEW + 3 * GAP
    H_ = GAP + TEXT_H + VIEW + GAP + PLOT_H + GAP
    screen = pygame.display.set_mode((W_, H_))
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
    big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 18, bold=True)
    clock = pygame.time.Clock()
    paused, speed, mode, show_food, frame = False, max(1, args.speed), COLOR_MODES.index(args.color), True, 0
    show_gifts = True

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
                elif ev.key == pygame.K_r:
                    for s in sides:
                        s.reset()
        if not paused:
            for _ in range(speed):
                for s in sides:
                    s.step(args.metric)

        screen.fill(BG)
        for k, s in enumerate(sides):
            x0 = GAP + k * (VIEW + GAP)
            w = s.world
            col = LEFT_C if k == 0 else RIGHT_C
            screen.blit(big.render(s.label, True, col), (x0, GAP))
            c = w.census()
            cur = s.hist[-1] if s.hist else float("nan")
            screen.blit(font.render(f"step {w.step_count}   population {c['population']}", True, FG),
                        (x0, GAP + 28))
            screen.blit(font.render(f"{args.metric} {cur:.4f}", True, col), (x0, GAP + 50))
            draw_world(screen, s, x0, GAP + TEXT_H, COLOR_MODES[mode], show_food, show_gifts)

        # shared plot
        py0 = GAP + TEXT_H + VIEW + GAP
        pw = W_ - 2 * GAP
        pygame.draw.rect(screen, (40, 40, 48), (GAP, py0, pw, PLOT_H), 1)
        vals = [v for s in sides for v in s.hist]
        if vals:
            hi = max(vals) * 1.1 or 1.0
            n_max = max(len(s.hist) for s in sides)
            for k, s in enumerate(sides):
                if len(s.hist) < 2:
                    continue
                h = np.asarray(s.hist)
                stride = max(1, len(h) // pw)
                h = h[::stride]
                xs = np.arange(len(h)) * stride * (pw - 1) / max(n_max - 1, 1)
                pts = [(GAP + int(x), py0 + PLOT_H - 4 - int(v / hi * (PLOT_H - 30))) for x, v in zip(xs, h)]
                pygame.draw.lines(screen, LEFT_C if k == 0 else RIGHT_C, False, pts, 2)
            steps = n_max * RECORD_EVERY
            screen.blit(font.render(f"{args.metric} (mean over the last season): whole run, {steps} steps",
                                    True, MUTED),
                        (GAP + 6, py0 + 4))
        info = (f"speed {speed}/frame   colour: {COLOR_MODES[mode]}   "
                f"SPACE pause  UP/DOWN speed  C colour  F food  G gifts  R restart")
        screen.blit(font.render(info, True, MUTED), (GAP + 6, py0 + PLOT_H - 22))
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
