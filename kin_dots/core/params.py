"""All simulation parameters in one place.

A run is fully defined by a ``Params`` instance (including ``seed``), so the
same params always reproduce the same simulation.
"""
from __future__ import annotations

import ast
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
    season_period: int = 0         # steps per full season cycle; 0 disables seasons
    season_amplitude: float = 0.0  # food growth and inflow are scaled by 1 + A*sin(2*pi*t/period)

    # --- population ---
    init_agents: int = 400
    max_agents: int = 20000        # array capacity; births beyond it are blocked (and counted)
    init_energy: float = 10.0

    # --- energy budget ---
    metabolism: float = 0.1        # paid by every agent every step
    move_speed: float = 1.0        # distance moved per step
    move_cost: float = 0.05        # energy per unit distance moved
    move_noise: float = 1.0        # std of the random component of the movement direction
    max_age: int = 300             # death by old age (keeps generations short); 0 disables

    # --- interactions (M2) ---
    share_enabled: bool = True          # False: nobody can donate energy
    share_all: bool = False             # True: "sacrifice" - the donor gives ALL its energy
                                        # (recipient gets share_benefit/share_cost of it) and dies
    share_sees_self: bool = True        # sharing decision may use the donor's own energy
    share_sees_other: bool = True       # ...and the recipient's energy
    social_movement: bool = True        # False: movement ignores neighbours (kin/stranger genes unused)
    interaction_radius: float = 4.0     # neighbours within this distance can be met / sensed
    share_cost: float = 1.0             # energy a donor pays
    share_benefit: float = 0.9          # energy the recipient gets. Keep <= share_cost: with
                                        # benefit > cost two agents sharing back and forth create
                                        # energy from nothing and the population explodes. The
                                        # Hamilton "b" then comes from *who* is helped (a starving
                                        # relative survives), not from the transfer itself.
    attacks_enabled: bool = True        # False: nobody can attack (sharing-only control)
    attack_cost: float = 0.2            # paid by an attacker whatever the outcome
    predation_efficiency: float = 0.5   # share of the loser's energy the winner gets
    tag_blind_share: bool = False       # E3 control: sharing decision cannot see tag similarity
    share_to_random: bool = False       # control: the recipient is a random agent from the whole
                                        # map instead of the neighbour met (same ecology, no kinship)
    weight_clip: float = 10.0           # decision/movement weights are kept in [-clip, clip]

    # --- gardening (a public good) ---
    garden_enabled: bool = False        # agents may spend energy to fertilise the cell below them
    garden_max: float = 0.05            # energy per step at garden gene = 1
    garden_efficiency: float = 1.0      # fertility added per unit of energy
    fertility_decay: float = 0.02       # fertility fades by this fraction per step
    garden_to_random: bool = False      # control: the fertiliser lands on a random cell of the map
    founder_garden: float = 0.2         # initial value of the garden gene (in [0, 1])

    # --- species (two competing species) ---
    n_species: int = 1                  # 2: agents carry an inherited, never-mutating species label
    species_init: str = "halves"        # "halves": species 0 on the left half, 1 on the right;
                                        # "mixed": random
    fights_within_species: bool = False # with 2 species: False = attacks only on the other species
                                        # (control: True). Sharing is always within species.
    defense_enabled: bool = False       # an attacked agent's neighbours (same species) may take
                                        # the fight on themselves (DEFEND genes)

    # --- predators (non-evolving hunters) and alarm calls ---
    n_predators: int = 0                # 0 disables predators
    predator_speed: float = 0.5         # distance per step
    predator_sight: float = 10.0        # predators chase prey within this distance
    predator_catch_radius: float = 1.0  # a strike is possible within this distance
    predator_handling: int = 20         # steps a predator rests after a kill
    catch_prob: float = 0.5             # chance a strike kills an unprotected prey
    hidden_catch_factor: float = 0.2    # catch_prob is multiplied by this for alerted (hiding) prey
    detect_radius: float = 6.0          # prey may notice predators within this distance...
    detect_prob: float = 0.3            # ...with this probability per step
    alert_duration: int = 5             # steps a detecting or warned agent stays hidden
    alarm_cost: float = 0.2             # energy paid for calling. A caller also cannot hide
                                        # and is chased first by predators that see it.
    predators_target_callers: bool = True  # predators that see a caller chase it first
    alarm_to_random: bool = False       # control: the warning reaches random agents on the map
                                        # instead of the caller's neighbours

    # --- reproduction (asexual) ---
    birth_threshold: float = 20.0  # energy needed to reproduce
    birth_cost: float = 1.0        # energy lost (not passed on) per birth
    child_fraction: float = 0.5    # share of the parent's remaining energy given to the child
    dispersal_sd: float = 2.0      # std of child offset from parent; < 0 means "anywhere on the map"

    # --- mutation ---
    mutation_sd: float = 0.05      # gaussian mutation of non-weight genes (harvest)
    weight_mutation_sd: float = 0.15  # gaussian mutation of decision/movement weights
    founder_harvest: float = 0.5      # initial value of the harvest gene
    founder_share_bias: float = -2.0  # initial share_w0 (p(share) = sigmoid of it, before inputs)
    freeze_harvest: bool = False      # True: the harvest gene never mutates
    founder_weight_sd: float = 0.5    # standing variation of weights in the founding population
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


def parse_overrides(items) -> Dict[str, Any]:
    """Turn ["key=value", ...] into {key: value}, checking names and types against Params.

    Values are Python literals (0.7, 300, True); anything else stays a string.
    Raises ValueError on an unknown name or a value of the wrong type."""
    defaults = Params()
    out: Dict[str, Any] = {}
    for kv in items:
        if "=" not in kv:
            raise ValueError(f"expected KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            val = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            val = v
        if not hasattr(defaults, k):
            raise ValueError(f"unknown parameter: {k}")
        ref = getattr(defaults, k)
        if isinstance(ref, bool):
            ok = isinstance(val, bool)
        elif isinstance(ref, (int, float)):
            ok = isinstance(val, (int, float)) and not isinstance(val, bool)
            if ok and isinstance(ref, int) and not isinstance(val, int):
                ok = float(val).is_integer()
                val = int(val)
        else:
            ok = isinstance(val, type(ref))
        if not ok:
            raise ValueError(f"bad value for {k}: {v!r} (default is {ref!r})")
        out[k] = val
    return out
