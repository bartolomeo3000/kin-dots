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
# Heritable colour tag (RGB in [0, 1]), mutated independently of behaviour
TAG = slice(12, 15)

N_GENES = 15

GENE_NAMES = (
    [f"share_w{i}" for i in range(4)]
    + [f"attack_w{i}" for i in range(4)]
    + ["harvest", "move_food", "move_kin", "move_stranger", "tag_r", "tag_g", "tag_b"]
)
assert len(GENE_NAMES) == N_GENES


# Everything except the tag: the genes that shape behaviour (used for genetic relatedness).
BEHAVIOUR_GENES = np.arange(0, 12)

# Genes that are decision/movement weights (clipped to [-weight_clip, weight_clip]).
WEIGHT_GENES = np.r_[0:8, MOVE_FOOD, MOVE_KIN, MOVE_STRANGER]


def propensity(g: np.ndarray, which: slice, sim: float = 1.0,
               e_self: float = 0.5, e_other: float = 0.5) -> np.ndarray:
    """Probability of sharing (which=SHARE_W) or attacking (ATTACK_W) in a reference
    situation: given tag similarity and both energies (as fractions of birth_threshold)."""
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
    g[:, HARVEST] = 0.5
    g[:, MOVE_FOOD] = 1.0
    if weight_sd > 0:
        g[:, WEIGHT_GENES] += rng.normal(scale=weight_sd, size=(n, WEIGHT_GENES.size))
    g[:, TAG] = rng.random((n, 3))
    return g
