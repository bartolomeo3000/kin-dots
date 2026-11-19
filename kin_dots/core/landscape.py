"""A fertility landscape: grass regrows faster in some places than in others.

The map is a smoothed random field on the torus (white noise blurred with a periodic
Gaussian), turned into fertility = exp(contrast * z) and normalised to mean 1, so the
total regrowth potential of the world is the same as without the map; only its
distribution in space changes. It looks like a relief map: fertile lowlands, barren
ridges.

Once a year (or every `fertility_period` steps) the field can change:
    z_new = memory * z_old + sqrt(1 - memory**2) * z_fresh
memory = 1 keeps the map fixed, 0 draws a new map every year, values in between let
it drift. The field stays statistically the same at every step.

The map has its own random generator, seeded from the run's seed, so two worlds with the
same seed (the demo and its control) always share the same landscape, whatever else
differs between them.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def smooth_noise(n: int, sigma_cells: float, rng: np.random.Generator) -> np.ndarray:
    """(n, n) field on a torus with zero mean and unit standard deviation, correlated
    over about sigma_cells cells."""
    white = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)  # cycles per cell
    kx, ky = np.meshgrid(k, k, indexing="ij")
    gauss = np.exp(-2.0 * np.pi ** 2 * sigma_cells ** 2 * (kx ** 2 + ky ** 2))
    z = np.real(np.fft.ifft2(np.fft.fft2(white) * gauss))
    z -= z.mean()
    sd = z.std()
    return z / sd if sd > 0 else z


class Landscape:
    def __init__(self, p: Params):
        self.p = p
        self.grid = None  # None = uniform fertility (the default model)
        if p.fertility_map == "none":
            return
        if p.fertility_map != "noise":
            raise ValueError(f"unknown fertility_map: {p.fertility_map!r} (use 'none' or 'noise')")
        self.rng = np.random.default_rng([p.seed, 7919])
        self.sigma = p.fertility_scale / p.cell_size
        self.z = smooth_noise(p.grid_cells, self.sigma, self.rng)
        self.period = p.fertility_period or p.season_period
        self.changes = 0
        self._refresh()

    def _refresh(self) -> None:
        f = np.exp(self.p.fertility_contrast * self.z)
        self.grid = f / f.mean()

    def update(self, step: int) -> None:
        """Call at the start of each step; changes the map at the start of each period."""
        if self.grid is None or self.p.fertility_memory >= 1.0 or not self.period:
            return
        if step > 0 and step % self.period == 0:
            m = self.p.fertility_memory
            fresh = smooth_noise(self.p.grid_cells, self.sigma, self.rng)
            self.z = m * self.z + np.sqrt(max(0.0, 1.0 - m * m)) * fresh
            self.z = (self.z - self.z.mean()) / (self.z.std() or 1.0)
            self.changes += 1
            self._refresh()
