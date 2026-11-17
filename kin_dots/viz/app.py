"""Live viewer (pygame).

    python -m kin_dots.viz.app [--config configs/default.yaml] [--seed 3]

Keys:
    SPACE        pause / resume
    RIGHT        one step (while paused)
    UP / DOWN    more / fewer simulation steps per frame
    C            colour agents by: tag -> p(share) -> p(attack) -> p(alarm) -> harvest -> energy

Predators (if enabled) are red triangles; agents that called an alarm this step
get a white ring. With two species, species 0 is drawn as dots and species 1 as squares.
    F            show / hide the food layer
    G            show / hide energy gifts (white line from donor to recipient, last step)
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
from kin_dots.core.kinship import kinship_report  # noqa: E402

VIEW = 800          # world view is VIEW x VIEW pixels
PANEL = 340         # side panel width
BG = (18, 18, 22)
FG = (225, 225, 230)
MUTED = (140, 140, 150)
BLUE = (120, 200, 255)
YELLOW = (250, 210, 40)
RED = (240, 90, 80)
GREEN = (90, 190, 110)
CYAN = (90, 220, 230)
PRED_COLOR = (255, 60, 60)
COLOR_MODES = ("tag", "species", "p(share|kin)", "p(attack|stranger)", "p(alarm|kin)",
               "p(defend|kin)", "harvest", "energy")
SPECIES_COLORS = np.array([[80, 160, 255], [255, 150, 50]], dtype=np.uint8)
HISTORY = 600       # samples kept for the side-panel plots
SMOOTH = 0.03       # exponential smoothing of the noisy per-step rates
KIN_EVERY = 100     # steps between genetic-relatedness measurements

# Reference situation used to show an agent's propensities as one number:
# both agents at half the birth threshold; "kin" = identical tag, "stranger" = similarity 0.5.
_REF_E = 0.5


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _ramp(t: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to a blue -> yellow ramp, returned as uint8 RGB."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    lo = np.array([40, 90, 220])
    hi = np.array([250, 210, 40])
    return (lo + (hi - lo) * t).astype(np.uint8)


def propensity(genome: np.ndarray, which: slice, sim: float) -> np.ndarray:
    return G.propensity(genome, which, sim, _REF_E, _REF_E)


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
        self.show_gifts = True
        self.reset(params.seed)

    def reset(self, seed: int) -> None:
        self.params = self.params.replace(seed=seed)
        self.world = World(self.params)
        keys = ("population", "share_rate", "attack_rate", "alarm_rate", "defend_rate",
                "sim_encounter", "sim_share", "sim_attack")
        self.hist = {k: deque(maxlen=HISTORY) for k in keys}
        self.kin_hist = {k: deque(maxlen=HISTORY // 4)
                         for k in ("r_met", "r_share", "r_attack", "r_alarm", "r_defend")}
        self.smooth = {}
        self._record()

    def _record(self) -> None:
        c = self.world.census()
        for k in self.hist:
            v = c.get(k, float("nan"))
            if k == "population":
                self.hist[k].append(v)
                continue
            if v == v:  # not NaN
                prev = self.smooth.get(k)
                self.smooth[k] = v if prev is None else prev + SMOOTH * (v - prev)
            if k in self.smooth:
                self.hist[k].append(self.smooth[k])

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
            elif k == pygame.K_g:
                self.show_gifts = not self.show_gifts
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
            if self.world.step_count % KIN_EVERY == 0 and self.world.population > 1:
                rep = kinship_report(self.world)
                for k, d in self.kin_hist.items():
                    v = rep.get(k, float("nan"))
                    d.append(v if v == v else (d[-1] if d else 0.0))

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
        g = w.genome[idx]
        if mode == "tag":
            return (40 + 215 * g[:, G.TAG]).astype(np.uint8)
        if mode == "p(share|kin)":
            return _ramp(propensity(g, G.SHARE_W, 1.0))
        if mode == "species":
            return SPECIES_COLORS[np.clip(w.species[idx], 0, 1)]
        if mode == "p(defend|kin)":
            return _ramp(G.propensity(g, G.DEFEND_W, 1.0, _REF_E, _REF_E))
        if mode == "p(alarm|kin)":
            # reference: kin around (sim 1), half-full energy, 3 neighbours
            return _ramp(G.propensity(g, G.ALARM_W, 1.0, _REF_E, 0.3))
        if mode == "p(attack|stranger)":
            return _ramp(propensity(g, G.ATTACK_W, 0.5))
        if mode == "harvest":
            p = w.p
            return _ramp((g[:, G.HARVEST] - p.harvest_min) / (p.harvest_max - p.harvest_min))
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
        square = w.species[idx] == 1
        rect = pygame.draw.rect
        for x, y, c, sq in zip(px.tolist(), py.tolist(), cols.tolist(), square.tolist()):
            if sq:
                rect(self.screen, c, (x - 2, y - 2, 5, 5))
            else:
                circle(self.screen, c, (x, y), 2)
        if self.show_gifts and len(w.last_gifts):
            half = w.p.world_size / 2
            for xa, ya, xb, yb in w.last_gifts.tolist():
                if abs(xa - xb) < half and abs(ya - yb) < half:  # skip gifts across the torus edge
                    pygame.draw.line(self.screen, (255, 255, 255), (int(xa * scale), int(ya * scale)),
                                     (int(xb * scale), int(yb * scale)), 1)
        callers = idx[w.caller[idx] == 1]
        for x, y in zip((w.x[callers] * scale).astype(int).tolist(), (w.y[callers] * scale).astype(int).tolist()):
            circle(self.screen, (255, 255, 255), (x, y), 6, 1)
        for x, y, hd in zip((w.px * scale).tolist(), (w.py * scale).tolist(), w.pheading.tolist()):
            pts = [(x + 9 * np.cos(hd + a), y + 9 * np.sin(hd + a)) for a in (0.0, 2.5, -2.5)]
            pygame.draw.polygon(self.screen, PRED_COLOR, pts)

    def _text(self, s: str, x: int, y: int, color=FG) -> None:
        self.screen.blit(self.font.render(s, True, color), (x, y))

    def _plot(self, x: int, y: int, w: int, h: int, series, fixed=None, n_max=HISTORY) -> None:
        """series: list of (label, deque, colour). fixed: (lo, hi) or None for auto range."""
        pygame.draw.rect(self.screen, (40, 40, 48), (x, y, w, h), 1)
        lx = x + 4
        for label, d, col in series:
            txt = f"{label} {d[-1]:.3g}" if d else label
            self._text(txt, lx, y + 2, col)
            lx += self.font.size(txt)[0] + 12
        vals = [v for _, d, _ in series for v in d]
        if not vals:
            return
        lo, hi = fixed if fixed else (min(vals), max(vals))
        span = (hi - lo) or 1.0
        for _, d, col in series:
            if len(d) < 2:
                continue
            pts = [(x + int(i * (w - 1) / (n_max - 1)),
                    y + h - 2 - int((min(max(v, lo), hi) - lo) / span * (h - 22)))
                   for i, v in enumerate(d)]
            pygame.draw.lines(self.screen, col, False, pts, 1)

    def _draw_panel(self) -> None:
        x0 = VIEW + 16
        w = self.world
        c = w.census()
        a = w.alive
        g = w.genome[a]
        p_share = float(propensity(g, G.SHARE_W, 1.0).mean()) if a.any() else 0.0
        p_attack = float(propensity(g, G.ATTACK_W, 0.5).mean()) if a.any() else 0.0
        lines = [
            f"step        {w.step_count}",
            f"population  {c['population']}"
            + (f"  ({c['pop_s0']}/{c['pop_s1']})" if w.p.n_species > 1 else ""),
            f"mean energy {c.get('mean_energy', 0):.2f}",
            f"births      {c['total_births']}",
            f"died/killed/eaten {c['total_deaths']}/{c['total_killed']}/{c['total_predated']}",
            f"p(share|kin)       {p_share:.3f}",
            f"p(attack|stranger) {p_attack:.3f}" if w.p.attacks_enabled else "p(attack|stranger) off",
            (f"p(alarm|kin)       {float(G.propensity(g, G.ALARM_W, 1.0, _REF_E, 0.3).mean()):.3f}"
             if w.p.n_predators and a.any() else
             f"p(defend|kin)      {float(G.propensity(g, G.DEFEND_W, 1.0, _REF_E, _REF_E).mean()):.3f}"
             if w.p.defense_enabled and a.any() else "predators off"),
            f"harvest     {c.get('mean_harvest', 0):.3f}",
            "",
            f"seed {w.p.seed}   speed {self.steps_per_frame}/frame   fps {self.clock.get_fps():.0f}",
            f"colour by   {COLOR_MODES[self.color_mode]}",
            "PAUSED" if self.paused else "",
        ]
        for i, s in enumerate(lines):
            self._text(s, x0, 12 + i * 18)

        y = 12 + len(lines) * 18 + 6
        pw, ph = PANEL - 32, 100
        h = self.hist
        self._plot(x0, y, pw, ph, [("population", h["population"], BLUE)])
        self._plot(x0, y + ph + 8, pw, ph,
                   [("share", h["share_rate"], YELLOW), ("attack", h["attack_rate"], RED)]
                   + ([("alarm/det", h["alarm_rate"], CYAN)] if w.p.n_predators else [])
                   + ([("def", h["defend_rate"], GREEN)] if w.p.defense_enabled else []))
        atk = w.p.attacks_enabled
        self._plot(x0, y + 2 * (ph + 8), pw, ph,
                   [("sim met", h["sim_encounter"], MUTED), ("help", h["sim_share"], GREEN)]
                   + ([("atk", h["sim_attack"], RED)] if atk else []))
        k = self.kin_hist
        self._plot(x0, y + 3 * (ph + 8), pw, ph,
                   [("r met", k["r_met"], MUTED), ("help", k["r_share"], GREEN)]
                   + ([("atk", k["r_attack"], RED)] if atk else [])
                   + ([("warn", k["r_alarm"], CYAN)] if w.p.n_predators else [])
                   + ([("def", k["r_defend"], YELLOW)] if w.p.defense_enabled else []), n_max=HISTORY // 4)

        help_lines = ["SPACE pause  RIGHT step", "UP/DOWN speed  C colour", "F food  R restart  N new seed"]
        for i, s in enumerate(help_lines):
            self._text(s, x0, VIEW - 62 + i * 18, MUTED)

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
    ap.add_argument("--color", choices=COLOR_MODES, default="tag", help="initial colour mode")
    ap.add_argument("--frames", type=int, default=0, help="quit after N frames (0 = run until closed)")
    ap.add_argument("--screenshot", type=Path, help="save the last frame to this PNG on exit")
    args = ap.parse_args(argv)

    params = Params.from_yaml(args.config) if args.config else Params()
    if args.seed is not None:
        params = params.replace(seed=args.seed)
    v = Viewer(params)
    v.steps_per_frame = max(1, args.speed)
    v.color_mode = COLOR_MODES.index(args.color)
    v.run(max_frames=args.frames, screenshot=args.screenshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
