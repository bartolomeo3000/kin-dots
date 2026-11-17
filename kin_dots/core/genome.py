"""Genome layout: each agent's genome is one row of a float array."""
from __future__ import annotations

import numpy as np

# Sharing decision: p = sigmoid(w0 + w1*tag_similarity + w2*my_energy + w3*their_energy)
# (energies divided by birth_threshold)
SHARE_W = slice(0, 4)
# Attack decision, same inputs
ATTACK_W = slice(4, 8)
# Fraction of the food in the current cell eaten each step (restraint channel)
HARVEST = 8
# Movement weights: towards food, towards similar tags, away from dissimilar tags
MOVE_FOOD = 9
MOVE_KIN = 10
MOVE_STRANGER = 11
# Alarm call when a predator is detected:
# p = sigmoid(w0 + w1*mean tag similarity of neighbours + w2*my_energy + w3*n_neighbours/10)
ALARM_W = slice(12, 16)
# Defending a neighbour attacked by another species (taking the fight on oneself):
# p = sigmoid(w0 + w1*tag similarity to the victim + w2*my_energy + w3*attacker's energy)
DEFEND_W = slice(16, 20)
# Heritable colour tag (RGB in [0, 1]), mutated independently of behaviour
TAG = slice(20, 23)

N_GENES = 23

GENE_NAMES = (
    [f"share_w{i}" for i in range(4)]
    + [f"attack_w{i}" for i in range(4)]
    + ["harvest", "move_food", "move_kin", "move_stranger"]
    + [f"alarm_w{i}" for i in range(4)]
    + [f"defend_w{i}" for i in range(4)]
    + ["tag_r", "tag_g", "tag_b"]
)
assert len(GENE_NAMES) == N_GENES


# Everything except the tag: the genes that shape behaviour (used for genetic relatedness).
BEHAVIOUR_GENES = np.arange(0, TAG.start)

# Genes that are decision/movement weights (clipped to [-weight_clip, weight_clip]).
WEIGHT_GENES = np.r_[0:8, MOVE_FOOD, MOVE_KIN, MOVE_STRANGER, 12:20]


def propensity(g: np.ndarray, which: slice, sim: float = 1.0,
               e_self: float = 0.5, e_other: float = 0.5) -> np.ndarray:
    """Probability of sharing (which=SHARE_W) or attacking (ATTACK_W) in a reference
    situation: given tag similarity and both energies (as fractions of birth_threshold).
    For ALARM_W the last input is the neighbour count / 10 instead of the other's energy."""
    w = g[:, which]
    z = w[:, 0] + w[:, 1] * sim + w[:, 2] * e_self + w[:, 3] * e_other
    return 1.0 / (1.0 + np.exp(-z))


def founder_genomes(n: int, rng: np.random.Generator, weight_sd: float = 0.0) -> np.ndarray:
    """Initial population: rare sharing and attacking, some standing variation in the
    weights (so selection has something to act on), and a random colour tag each.

    The biases sit where the sigmoid still has slope; starting deep in its flat tail
    (e.g. -5) leaves selection almost no signal on the other weights.
    """
    g = np.zeros((n, N_GENES))
    g[:, SHARE_W] = [-2.0, 0.0, 0.0, 0.0]   # p(share) ~ 12% per encounter
    g[:, ATTACK_W] = [-3.0, 0.0, 0.0, 0.0]  # p(attack) ~ 5% per encounter
    g[:, ALARM_W] = [-2.0, 0.0, 0.0, 0.0]   # p(alarm | predator detected) ~ 12%
    g[:, DEFEND_W] = [-3.0, 0.0, 0.0, 0.0]  # p(defend a neighbour) ~ 5% per candidate
    g[:, HARVEST] = 0.5
    g[:, MOVE_FOOD] = 1.0
    if weight_sd > 0:
        g[:, WEIGHT_GENES] += rng.normal(scale=weight_sd, size=(n, WEIGHT_GENES.size))
    g[:, TAG] = rng.random((n, 3))
    return g
