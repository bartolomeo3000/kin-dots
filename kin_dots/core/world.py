"""The simulation state and its update rule.

The state is kept as arrays (struct-of-arrays) instead of Agent objects, so
each phase of a step is a few NumPy operations over the whole population.
Arrays have fixed capacity ``params.max_agents``; ``alive`` marks used slots
and dead slots are reused by newborns.

Step order: find neighbours -> interact (attack / share) -> move -> eat
-> metabolism -> die -> reproduce -> age -> food regrows.
Neighbour lists are built once per step (before anyone moves) and used by both
the interaction and the movement phase.

Every step records an energy ledger in ``self.last`` so the energy budget
can be checked exactly:
    d(total agent energy) = eaten - metabolic - movement - birth_cost - death_removed
                            - share_given + share_received
                            - attack_cost - killed_energy + predation_gain
    d(total food)         = food_growth - eaten
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import genome as G
from . import kernels as K
from .params import Params

_LEDGER_KEYS = {
    "shares": K.L_SHARES, "share_given": K.L_SHARE_GIVEN, "share_received": K.L_SHARE_RECEIVED,
    "attacks": K.L_ATTACKS, "attack_cost": K.L_ATTACK_COST, "killed": K.L_KILLED,
    "killed_energy": K.L_KILLED_ENERGY, "predation_gain": K.L_PREDATION_GAIN,
    "attacker_lost": K.L_ATTACKER_LOST, "encounters": K.L_ENCOUNTERS,
    "sim_encounter": K.L_SIM_ENCOUNTER, "sim_share": K.L_SIM_SHARE, "sim_attack": K.L_SIM_ATTACK,
    "donor_energy": K.L_DONOR_E, "recipient_energy": K.L_RECIPIENT_E,
}

# The 8 neighbouring food cells and the unit vector pointing to each.
_OFFSETS = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])
_UNIT = _OFFSETS / np.linalg.norm(_OFFSETS, axis=1, keepdims=True)


class World:
    def __init__(self, params: Params):
        self.p = p = params
        self.rng = np.random.default_rng(p.seed)
        cap = p.max_agents
        if p.init_agents > cap:
            raise ValueError("init_agents exceeds max_agents")

        self.alive = np.zeros(cap, dtype=bool)
        self.x = np.zeros(cap)
        self.y = np.zeros(cap)
        self.energy = np.zeros(cap)
        self.age = np.zeros(cap, dtype=np.int64)
        self.id = np.full(cap, -1, dtype=np.int64)
        self.parent_id = np.full(cap, -1, dtype=np.int64)
        self.genome = np.zeros((cap, G.N_GENES))

        n = p.grid_cells
        self.food = np.full((n, n), p.food_capacity * p.food_init_frac)

        self.step_count = 0
        self.next_id = 0
        self.total_births = 0
        self.total_deaths = 0   # starvation + old age
        self.total_starved = 0  # starvation only
        self.total_killed = 0   # deaths in fights
        self.last: Dict[str, float] = {}

        self.nb = max(3, int(p.world_size // p.interaction_radius))  # hash buckets per side
        self._buf = np.empty(cap, dtype=np.int64)  # scratch space for neighbour lists

        k = p.init_agents
        slots = np.arange(k)
        self.alive[slots] = True
        self.x[slots] = self.rng.random(k) * p.world_size
        self.y[slots] = self.rng.random(k) * p.world_size
        self.energy[slots] = p.init_energy
        self.genome[slots] = G.founder_genomes(k, self.rng, p.founder_weight_sd)
        self.id[slots] = np.arange(k)
        self.next_id = k

    # ------------------------------------------------------------------ queries

    @property
    def population(self) -> int:
        return int(self.alive.sum())

    def total_energy(self) -> float:
        return float(self.energy[self.alive].sum())

    def total_food(self) -> float:
        return float(self.food.sum())

    def cell_of(self, idx: np.ndarray):
        n = self.p.grid_cells
        cs = self.p.cell_size
        cx = (self.x[idx] // cs).astype(np.int64) % n
        cy = (self.y[idx] // cs).astype(np.int64) % n
        return cx, cy

    # ------------------------------------------------------------------ update

    def step(self) -> None:
        idx = np.flatnonzero(self.alive)
        last = dict(eaten=0.0, metabolic=0.0, movement=0.0, birth_cost=0.0,
                    death_removed=0.0, food_growth=0.0, births=0, deaths=0, births_blocked=0)
        last.update({k: 0.0 for k in _LEDGER_KEYS})

        if idx.size:
            p = self.p
            nbr_start, nbr = K.neighbour_lists(idx, self.x, self.y, self.alive, p.world_size,
                                               self.nb, p.interaction_radius, self._buf)
            last.update(self._interact(idx, nbr_start, nbr))
            kin, strangers = K.social_vectors(idx, self.x, self.y, self.alive, self.genome,
                                              G.TAG.start, nbr_start, nbr, p.world_size)
            live = self.alive[idx]  # fights may have killed some
            idx, kin, strangers = idx[live], kin[live], strangers[live]
            last["movement"] = self._move(idx, kin, strangers)
            last["eaten"] = self._eat(idx)
            last["metabolic"] = self.p.metabolism * idx.size
            self.energy[idx] -= self.p.metabolism
            last["deaths"], last["death_removed"] = self._die(idx)
            last["births"], last["births_blocked"], last["birth_cost"] = self._reproduce()
            self.age[self.alive] += 1

        last["food_growth"] = self._regrow()
        self.total_births += last["births"]
        self.total_deaths += last["deaths"]
        self.total_killed += int(last["killed"])
        self.last = last
        self.step_count += 1

    def _move(self, idx: np.ndarray, kin: np.ndarray, strangers: np.ndarray) -> float:
        """kin / strangers: per-agent social vectors (see kernels.social_vectors)."""
        p, n = self.p, self.p.grid_cells
        cx, cy = self.cell_of(idx)
        own = self.food[cx, cy]

        # Food gradient: sum over neighbours of (their food - my food) * direction to them.
        grad = np.zeros((idx.size, 2))
        for (dx, dy), u in zip(_OFFSETS, _UNIT):
            diff = self.food[(cx + dx) % n, (cy + dy) % n] - own
            grad += diff[:, None] * u

        # Social pull: towards similar-tagged neighbours, away from dissimilar ones.
        g = self.genome[idx]
        d = (g[:, G.MOVE_FOOD, None] * grad
             + g[:, G.MOVE_KIN, None] * kin
             - g[:, G.MOVE_STRANGER, None] * strangers
             + self.rng.normal(scale=p.move_noise, size=(idx.size, 2)))
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.divide(d, norm, out=np.zeros_like(d), where=norm > 0)

        self.x[idx] = (self.x[idx] + d[:, 0] * p.move_speed) % p.world_size
        self.y[idx] = (self.y[idx] + d[:, 1] * p.move_speed) % p.world_size

        cost = p.move_cost * p.move_speed
        self.energy[idx] -= cost
        return cost * idx.size

    def _eat(self, idx: np.ndarray) -> float:
        """Agents in a cell each try to eat a fraction h of its food.

        Within a cell the result equals eating one after another (the cell keeps
        prod(1 - h_i) of its food), and the eaten amount is split in proportion
        to h_i, so the order of agents does not matter.
        """
        p, n = self.p, self.p.grid_cells
        cx, cy = self.cell_of(idx)
        cell = cx * n + cy
        h = np.clip(self.genome[idx, G.HARVEST], p.harvest_min, p.harvest_max)

        log_keep = np.bincount(cell, weights=np.log1p(-h), minlength=n * n)
        sum_h = np.bincount(cell, weights=h, minlength=n * n)

        food = self.food.reshape(-1)  # view: edits go into self.food
        eaten_cell = food * -np.expm1(log_keep)  # = food * (1 - prod(1 - h))
        share = eaten_cell[cell] * h / sum_h[cell]

        food -= eaten_cell
        self.energy[idx] += share
        return float(eaten_cell.sum())

    def _interact(self, idx: np.ndarray, nbr_start=None, nbr=None) -> Dict[str, float]:
        p = self.p
        if nbr_start is None:
            nbr_start, nbr = K.neighbour_lists(idx, self.x, self.y, self.alive, p.world_size,
                                               self.nb, p.interaction_radius, self._buf)
        acting_k = self.rng.permutation(idx.size)
        u = self.rng.random((idx.size, 5))
        ledger = np.zeros(K.N_LEDGER)
        K.interact(acting_k, idx, self.energy, self.alive, self.genome,
                   G.SHARE_W.start, G.ATTACK_W.start, G.TAG.start, p.tag_blind_share, p.attacks_enabled,
                   p.share_to_random,
                   nbr_start, nbr, p.birth_threshold, p.share_cost, p.share_benefit,
                   p.attack_cost, p.predation_efficiency, u, self._buf, ledger)
        return {k: float(ledger[i]) for k, i in _LEDGER_KEYS.items()}

    def _die(self, idx: np.ndarray):
        starved = self.energy[idx] <= 0
        dead = starved.copy()
        if self.p.max_age > 0:
            dead |= self.age[idx] >= self.p.max_age
        self.total_starved += int(starved.sum())
        d = idx[dead]
        removed = float(self.energy[d].sum())
        self.alive[d] = False
        self.energy[d] = 0.0
        return int(d.size), removed

    def _reproduce(self):
        p = self.p
        parents = np.flatnonzero(self.alive & (self.energy >= p.birth_threshold))
        if parents.size == 0:
            return 0, 0, 0.0

        free = np.flatnonzero(~self.alive)
        blocked = max(0, parents.size - free.size)
        if blocked:
            parents = self.rng.permutation(parents)[: free.size]
        kids = free[: parents.size]
        k = kids.size

        # Energy: pay the birth cost, then split the rest between parent and child.
        remaining = self.energy[parents] - p.birth_cost
        child_e = remaining * p.child_fraction
        self.energy[parents] = remaining - child_e
        self.energy[kids] = child_e

        # Position: next to the parent, or anywhere on the map (control condition).
        if p.dispersal_sd < 0:
            self.x[kids] = self.rng.random(k) * p.world_size
            self.y[kids] = self.rng.random(k) * p.world_size
        else:
            self.x[kids] = (self.x[parents] + self.rng.normal(scale=p.dispersal_sd, size=k)) % p.world_size
            self.y[kids] = (self.y[parents] + self.rng.normal(scale=p.dispersal_sd, size=k)) % p.world_size

        # Genome: copy with gaussian mutation; the tag mutates at its own rate.
        g = self.genome[parents].copy()
        g += self.rng.normal(scale=p.mutation_sd, size=g.shape)
        g[:, G.WEIGHT_GENES] += self.rng.normal(
            scale=np.sqrt(max(p.weight_mutation_sd ** 2 - p.mutation_sd ** 2, 0.0)),
            size=(k, G.WEIGHT_GENES.size))  # weights get weight_mutation_sd in total
        # Keep the gene itself in range so no hidden drift builds up past the limit.
        g[:, G.HARVEST] = np.clip(g[:, G.HARVEST], p.harvest_min, p.harvest_max)
        g[:, G.WEIGHT_GENES] = np.clip(g[:, G.WEIGHT_GENES], -p.weight_clip, p.weight_clip)
        g[:, G.TAG] = np.clip(
            self.genome[parents][:, G.TAG] + self.rng.normal(scale=p.tag_mutation_sd, size=(k, 3)), 0.0, 1.0
        )
        self.genome[kids] = g

        self.alive[kids] = True
        self.age[kids] = -1  # becomes 0 after this step's ageing
        self.id[kids] = np.arange(self.next_id, self.next_id + k)
        self.parent_id[kids] = self.id[parents]
        self.next_id += k
        return k, blocked, p.birth_cost * k

    def season_factor(self) -> float:
        """Multiplier on food regrowth: 1 without seasons, oscillates with them."""
        p = self.p
        if p.season_period <= 0 or p.season_amplitude == 0:
            return 1.0
        return max(0.0, 1.0 + p.season_amplitude * np.sin(2 * np.pi * self.step_count / p.season_period))

    def _regrow(self) -> float:
        p = self.p
        before = self.food.sum()
        f = self.food
        s = self.season_factor()
        f += s * (p.food_growth * f * (1.0 - f / p.food_capacity) + p.food_inflow)
        np.clip(f, 0.0, p.food_capacity, out=f)
        return float(self.food.sum() - before)

    # ------------------------------------------------------------------ summaries

    def random_pair_similarity(self, n_pairs: int = 500) -> float:
        """Mean tag similarity of random pairs of living agents (baseline for assortment).

        Uses its own generator seeded by the step number, so calling it never
        changes the simulation's random stream.
        """
        idx = np.flatnonzero(self.alive)
        if idx.size < 2:
            return float("nan")
        r = np.random.default_rng(self.step_count)
        i = idx[r.integers(idx.size, size=n_pairs)]
        j = idx[r.integers(idx.size, size=n_pairs)]
        d = np.linalg.norm(self.genome[i, G.TAG] - self.genome[j, G.TAG], axis=1)
        return float(np.mean(1.0 - d / np.sqrt(3.0)))

    def census(self) -> Dict[str, float]:
        a = self.alive
        out: Dict[str, float] = {
            "step": self.step_count,
            "population": int(a.sum()),
            "total_food": self.total_food(),
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "total_killed": self.total_killed,
            "total_starved": self.total_starved,
            "season": self.season_factor(),
        }
        # Behaviour actually shown in the last step (per encounter), and who it was aimed at.
        L = self.last
        enc = L.get("encounters", 0.0)
        nan = float("nan")
        out["share_rate"] = L.get("shares", 0.0) / enc if enc else nan
        out["attack_rate"] = L.get("attacks", 0.0) / enc if enc else nan
        out["sim_encounter"] = L.get("sim_encounter", 0.0) / enc if enc else nan
        out["sim_share"] = L["sim_share"] / L["shares"] if L.get("shares") else nan
        out["sim_attack"] = L["sim_attack"] / L["attacks"] if L.get("attacks") else nan
        out["donor_energy"] = L["donor_energy"] / L["shares"] if L.get("shares") else nan
        out["recipient_energy"] = L["recipient_energy"] / L["shares"] if L.get("shares") else nan
        out["sim_random"] = self.random_pair_similarity()
        out["assortment"] = out["sim_encounter"] - out["sim_random"]
        if a.any():
            g = self.genome[a]
            out["mean_energy"] = float(self.energy[a].mean())
            out["mean_age"] = float(self.age[a].mean())
            for i, name in enumerate(G.GENE_NAMES[: G.TAG.start]):
                out[f"mean_{name}"] = float(g[:, i].mean())
        return out
