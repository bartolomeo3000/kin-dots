"""All simulation parameters in one place.

A run is fully defined by a ``Params`` instance (including ``seed``), so the
same params always reproduce the same simulation.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass
class Params:
    # --- reproducibility ---
    seed: int = 0

    # --- world ---
    world_size: float = 200.0      # side of the square torus (continuous units)
    grid_cells: int = 50           # food grid is grid_cells x grid_cells
    food_capacity: float = 10.0    # K: max food per cell
    food_growth: float = 0.05      # r: logistic regrowth rate per step
    food_inflow: float = 0.1       # constant regrowth per cell per step (lets empty cells recover; sets pop ~1800)
    food_init_frac: float = 1.0    # initial food as a fraction of capacity

    # --- population ---
    init_agents: int = 400
    max_agents: int = 20000        # array capacity; births beyond it are blocked (and counted)
    init_energy: float = 10.0

    # --- energy budget ---
    metabolism: float = 0.1        # paid by every agent every step
    move_speed: float = 1.0        # distance moved per step
    move_cost: float = 0.05        # energy per unit distance moved
    move_noise: float = 1.0        # std of the random component of the movement direction
    max_age: int = 0               # 0 disables death by old age

    # --- reproduction (asexual) ---
    birth_threshold: float = 20.0  # energy needed to reproduce
    birth_cost: float = 1.0        # energy lost (not passed on) per birth
    child_fraction: float = 0.5    # share of the parent's remaining energy given to the child
    dispersal_sd: float = 2.0      # std of child offset from parent; < 0 means "anywhere on the map"

    # --- mutation ---
    mutation_sd: float = 0.05      # gaussian mutation of behavioural genes
    tag_mutation_sd: float = 0.02  # gaussian mutation of the colour tag (kept separate on purpose)

    # --- genes clipped to a valid range ---
    harvest_min: float = 0.01
    harvest_max: float = 0.95

    @property
    def cell_size(self) -> float:
        return self.world_size / self.grid_cells

    def replace(self, **changes: Any) -> "Params":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Params":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Params":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
