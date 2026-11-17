"""Who descends from whom, and a log of who interacted with whom.

Genealogy: for every agent id ever born, its parent id, generation depth and
birth step. Ids are handed out consecutively, so plain growable arrays indexed
by id are enough.

EventLog: a bounded ring buffer of interactions (who met / helped / attacked
whom, with their genetic distance at that moment). Metrics drain it
periodically; if nobody drains it, the oldest events are overwritten.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from numba import njit

NO_PARENT = -1

# Event kinds
EV_MET = 0      # an agent met a neighbour (whatever happened next)
EV_SHARE = 1    # actor donated energy to target
EV_ATTACK = 2   # actor attacked target
EV_RANDOM = 3   # a random pair from the whole population (baseline, not an interaction)
EV_ALARM = 4    # actor's alarm call warned target
EV_DEFEND = 5   # actor took a fight on itself to defend target (kernels.LOG_DEFEND)
EVENT_NAMES = {EV_MET: "met", EV_SHARE: "share", EV_ATTACK: "attack", EV_RANDOM: "random",
               EV_ALARM: "alarm", EV_DEFEND: "defend"}


class Genealogy:
    def __init__(self, capacity: int = 1 << 16):
        self.parent = np.full(capacity, NO_PARENT, dtype=np.int64)
        self.depth = np.zeros(capacity, dtype=np.int64)   # generations since a founder
        self.birth = np.zeros(capacity, dtype=np.int64)   # step of birth
        self.n = 0                                         # ids 0..n-1 are known

    def _grow(self, need: int) -> None:
        cap = self.parent.size
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name, fill in (("parent", NO_PARENT), ("depth", 0), ("birth", 0)):
            old = getattr(self, name)
            arr = np.full(new, fill, dtype=old.dtype)
            arr[:cap] = old
            setattr(self, name, arr)

    def add(self, ids: np.ndarray, parents: np.ndarray, step: int) -> None:
        """Register newborns. ids must be the next consecutive ids."""
        if ids.size == 0:
            return
        assert ids[0] == self.n and ids[-1] == self.n + ids.size - 1, "ids must be consecutive"
        self._grow(self.n + ids.size)
        self.parent[ids] = parents
        has_parent = parents >= 0
        self.depth[ids] = 0
        self.depth[ids[has_parent]] = self.depth[parents[has_parent]] + 1
        self.birth[ids] = step
        self.n += ids.size

    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Generations from a and from b up to their most recent common ancestor
        (summed). Parent-child = 1, siblings = 2, cousins = 4, same agent = 0,
        -1 when they descend from different founders."""
        return _mrca_distance(np.asarray(a, np.int64), np.asarray(b, np.int64), self.parent, self.depth)


@njit(cache=True)
def mrca_pair(x, y, parent, depth):
    """Summed generations from x and y to their most recent common ancestor; -1 if none."""
    steps = 0
    while depth[x] > depth[y]:
        x = parent[x]
        steps += 1
    while depth[y] > depth[x]:
        y = parent[y]
        steps += 1
    while x != y:
        x = parent[x]
        y = parent[y]
        steps += 2
        if x < 0 or y < 0:
            return -1
    return steps


@njit(cache=True)
def _mrca_distance(a, b, parent, depth):
    out = np.empty(a.size, np.int64)
    for k in range(a.size):
        out[k] = mrca_pair(a[k], b[k], parent, depth)
    return out


class EventLog:
    FIELDS = (("kind", np.int8), ("actor", np.int64), ("target", np.int64), ("d2", np.float64),
              ("sim", np.float64), ("actor_e", np.float64), ("target_e", np.float64))

    def __init__(self, capacity: int = 100_000):
        self.cap = capacity
        self.data = {name: np.zeros(capacity, dtype=dt) for name, dt in self.FIELDS}
        self.head = 0      # next write position
        self.size = 0      # number of valid entries (<= cap)

    def append(self, **cols: np.ndarray) -> None:
        n = len(cols["kind"])
        if n == 0:
            return
        if n >= self.cap:  # keep only the newest
            cols = {k: v[-self.cap:] for k, v in cols.items()}
            n = self.cap
        pos = (self.head + np.arange(n)) % self.cap
        for name, _ in self.FIELDS:
            self.data[name][pos] = cols[name]
        self.head = (self.head + n) % self.cap
        self.size = min(self.cap, self.size + n)

    def drain(self) -> Dict[str, np.ndarray]:
        """Return all buffered events (oldest first) and empty the buffer."""
        if self.size == 0:
            out = {name: np.zeros(0, dtype=dt) for name, dt in self.FIELDS}
        else:
            start = (self.head - self.size) % self.cap
            pos = (start + np.arange(self.size)) % self.cap
            out = {name: self.data[name][pos].copy() for name, _ in self.FIELDS}
        self.size = 0
        return out
