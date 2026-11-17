"""Genome layout: each agent's genome is one row of a float array.

Only HARVEST and MOVE_FOOD are used in M1. The other genes already exist,
mutate and get inherited, so later milestones only add behaviour that reads them.
"""
from __future__ import annotations

import numpy as np

# Sharing decision: p = sigmoid(w0 + w1*tag_similarity + w2*my_energy + w3*their_energy)  (M2)
SHARE_W = slice(0, 4)
# Attack decision, same inputs  (M2)
ATTACK_W = slice(4, 8)
# Fraction of the food in the current cell eaten each step (restraint channel)
HARVEST = 8
# Movement weights: towards food, towards similar tags, away from dissimilar tags  (kin/stranger: M2)
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


def founder_genomes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Initial population: identical behaviour and a random colour tag each."""
    g = np.zeros((n, N_GENES))
    g[:, SHARE_W] = [-4.0, 0.0, 0.0, 0.0]   # sharing starts essentially off
    g[:, ATTACK_W] = [-4.0, 0.0, 0.0, 0.0]  # attacking starts essentially off
    g[:, HARVEST] = 0.5
    g[:, MOVE_FOOD] = 1.0
    g[:, TAG] = rng.random((n, 3))
    return g
