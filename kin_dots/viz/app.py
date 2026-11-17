"""Live viewer (pygame).

    python -m kin_dots.viz.app [--config configs/default.yaml] [--seed 3]

Keys:
    SPACE        pause / resume
    RIGHT        one step (while paused)
    UP / DOWN    more / fewer simulation steps per frame
    C            colour agents by: tag -> harvest gene -> energy
    F            show / hide the food layer
    R            restart with the same seed
    N            restart with the next seed
    ESC / Q      quit
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

VIEW = 800          # world view is VIEW x VIEW pixels
PANEL = 320         # side panel width
BG = (18, 18, 22)
FG = (225, 225, 230)
MUTED = (140, 140, 150)
COLOR_MODES = ("tag", "harvest", "energy")
HISTORY = 600       # samples kept for the side-panel plots


def _ramp(t: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to a blue -> yellow ramp, returned as uint8 RGB."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    lo = np.array([40, 90, 220])
    hi = np.array([250, 210, 40])
    return (lo + (hi - lo) * t).astype(np.uint8)


class Viewer:
    def __init__(self, params: Params):
        self.params = params
        pygame.init()
        pygame.display.set_caption("kin-dots")
        self.screen = pygame.display.set_mode((VIEW + PANEL, VIEW))
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 14)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.steps_per_frame = 1
        self.color_mode = 0
        self.show_food = True
        self.reset(params.seed)

    def reset(self, seed: int) -> None:
        self.params = self.params.replace(seed=seed)
        self.world = World(self.params)
        self.hist = {k: deque(maxlen=HISTORY) for k in ("population", "mean_harvest", "total_food")}
        self._record()

    def _record(self) -> None:
        c = self.world.census()
        for k in self.hist:
            self.hist[k].append(c.get(k, 0.0))

    # ---------------------------------------------------------------- input

    def handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type != pygame.KEYDOWN:
                continue
            k = ev.key
            if k in (pygame.K_ESCAPE, pygame.K_q):
                return False
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_RIGHT and self.paused:
                self.advance(1)
            elif k == pygame.K_UP:
                self.steps_per_frame = min(self.steps_per_frame * 2, 64)
            elif k == pygame.K_DOWN:
                self.steps_per_frame = max(self.steps_per_frame // 2, 1)
            elif k == pygame.K_c:
                self.color_mode = (self.color_mode + 1) % len(COLOR_MODES)
            elif k == pygame.K_f:
                self.show_food = not self.show_food
            elif k == pygame.K_r:
                self.reset(self.params.seed)
            elif k == pygame.K_n:
                self.reset(self.params.seed + 1)
        return True

    def advance(self, n: int) -> None:
        for _ in range(n):
            if self.world.population == 0:
                break
            self.world.step()
            self._record()

    # ---------------------------------------------------------------- drawing

    def draw(self) -> None:
        self.screen.fill(BG)
        self._draw_food()
        self._draw_agents()
        self._draw_panel()
        pygame.display.flip()

    def _draw_food(self) -> None:
        if not self.show_food:
            return
        w = self.world
        # sqrt makes the (usually low) food levels visible
        t = np.sqrt(w.food / w.p.food_capacity)[..., None]
        rgb = (np.array(BG) * (1 - t) + np.array([30, 95, 45]) * t).astype(np.uint8)
        surf = pygame.surfarray.make_surface(rgb)  # indexed [x, y], same as world.food
        self.screen.blit(pygame.transform.scale(surf, (VIEW, VIEW)), (0, 0))

    def _agent_colors(self, idx: np.ndarray) -> np.ndarray:
        w, mode = self.world, COLOR_MODES[self.color_mode]
        if mode == "tag":
            return (40 + 215 * w.genome[idx, G.TAG]).astype(np.uint8)
        if mode == "harvest":
            p = w.p
            return _ramp((w.genome[idx, G.HARVEST] - p.harvest_min) / (p.harvest_max - p.harvest_min))
        return _ramp(w.energy[idx] / w.p.birth_threshold)

    def _draw_agents(self) -> None:
        w = self.world
        idx = np.flatnonzero(w.alive)
        if idx.size == 0:
            return
        scale = VIEW / w.p.world_size
        px = (w.x[idx] * scale).astype(int)
        py = (w.y[idx] * scale).astype(int)
        cols = self._agent_colors(idx)
        circle = pygame.draw.circle
        for x, y, c in zip(px.tolist(), py.tolist(), cols.tolist()):
            circle(self.screen, c, (x, y), 2)

    def _text(self, s: str, x: int, y: int, color=FG) -> None:
        self.screen.blit(self.font.render(s, True, color), (x, y))

    def _plot(self, series, x: int, y: int, w: int, h: int, label: str, color) -> None:
        pygame.draw.rect(self.screen, (40, 40, 48), (x, y, w, h), 1)
        vals = list(series)
        self._text(f"{label}: {vals[-1]:.3g}" if vals else label, x + 4, y + 2, MUTED)
        if len(vals) < 2:
            return
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        n = len(vals)
        pts = [(x + int(i * (w - 1) / (HISTORY - 1)), y + h - 2 - int((v - lo) / span * (h - 22)))
               for i, v in enumerate(vals)]
        if n >= 2:
            pygame.draw.lines(self.screen, color, False, pts, 1)

    def _draw_panel(self) -> None:
        x0 = VIEW + 16
        w = self.world
        c = w.census()
        lines = [
            f"step        {w.step_count}",
            f"population  {c['population']}",
            f"mean energy {c.get('mean_energy', 0):.2f}",
            f"mean age    {c.get('mean_age', 0):.0f}",
            f"births      {c['total_births']}",
            f"deaths      {c['total_deaths']}",
            "",
            f"seed        {w.p.seed}",
            f"speed       {self.steps_per_frame} step/frame",
            f"colour by   {COLOR_MODES[self.color_mode]}",
            f"fps         {self.clock.get_fps():.0f}",
            "PAUSED" if self.paused else "",
        ]
        for i, s in enumerate(lines):
            self._text(s, x0, 16 + i * 18)

        y = 16 + len(lines) * 18 + 10
        pw = PANEL - 32
        self._plot(self.hist["population"], x0, y, pw, 110, "population", (120, 200, 255))
        self._plot(self.hist["mean_harvest"], x0, y + 120, pw, 110, "mean harvest", (250, 210, 40))
        self._plot(self.hist["total_food"], x0, y + 240, pw, 110, "total food", (90, 190, 110))

        help_lines = ["SPACE pause  RIGHT step", "UP/DOWN speed  C colour", "F food  R restart  N new seed"]
        for i, s in enumerate(help_lines):
            self._text(s, x0, VIEW - 66 + i * 18, MUTED)

    # ---------------------------------------------------------------- loop

    def run(self, max_frames: int = 0, screenshot: Path = None) -> None:
        frame = 0
        running = True
        while running:
            running = self.handle_events()
            if not self.paused:
                self.advance(self.steps_per_frame)
            self.draw()
            self.clock.tick(60)
            frame += 1
            if max_frames and frame >= max_frames:
                break
        if screenshot:
            pygame.image.save(self.screen, str(screenshot))
        pygame.quit()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML file with Params overrides")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--speed", type=int, default=1, help="initial steps per frame")
    ap.add_argument("--frames", type=int, default=0, help="quit after N frames (0 = run until closed)")
    ap.add_argument("--screenshot", type=Path, help="save the last frame to this PNG on exit")
    args = ap.parse_args(argv)

    params = Params.from_yaml(args.config) if args.config else Params()
    if args.seed is not None:
        params = params.replace(seed=args.seed)
    v = Viewer(params)
    v.steps_per_frame = max(1, args.speed)
    v.run(max_frames=args.frames, screenshot=args.screenshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
