"""Kinship metrics (M3): is help really aimed at relatives, and what is selection doing?

All functions only read the World; none of them changes the simulation.

kinship_report(world)
    Drains the world's event log and, for each kind of interaction (met, share,
    attack, alarm = caller -> warned agent, random baseline pair), reports:
      r_<kind>       genetic relatedness: 1 - mean d2(pair) / mean d2(random pair),
                     d2 = squared distance of behaviour genes, each scaled by its
                     population variance. 1 = clones, 0 = like random pairs.
      gdist_<kind>   median generations to the most recent common ancestor (summed
                     over both sides; parent-child 1, siblings 2, cousins 4)
      parent_child_<kind>, kin2_<kind>, close_kin_<kind>
                     share of pairs at genealogical distance 1, 2, and 1..4
      tagsim_<kind>  mean tag similarity (the cue agents actually see)
    plus Hamilton's rule in energy units (r_share * benefit vs cost) and the
    share of donations going to the donor's own child / parent.

PriceTracker
    Snapshot the population, run, then decompose the change of every behaviour
    gene's mean into selection (cov(fitness, gene) / mean fitness, fitness =
    number of descendants alive at the end) and transmission (mutation etc.).

clans_and_cheaters(world)
    Clans = connected groups of neighbours that are close kin (genealogical
    distance <= 4). Cheaters = members of a sharing clan that share much less
    than their clan does.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from numba import njit

from . import genome as G
from . import kernels as K
from .genealogy import (EV_ALARM, EV_ATTACK, EV_DEFEND, EV_MET, EV_RANDOM, EV_SHARE, EVENT_NAMES,
                        mrca_pair)

MAX_PAIRS_FOR_GENEALOGY = 3000  # sample size for the (slower) genealogical distances
FATE_HORIZON = 50               # "dies soon" = dies within this many steps after the act
FATE_SAMPLE = 400               # acts sampled per report for the fate statistics


def _fate(world, ev, rng) -> Dict[str, float]:
    """Do donors die soon after donating, more than agents in general?

    Samples (actor, step) of donations and of ordinary encounters ('met', the
    baseline) are kept on the world between reports; once HORIZON steps have
    passed, each sample is resolved as died-within-horizon or not.
    """
    store = world.__dict__.setdefault("_fate_samples", {"share": [], "met": []})
    for name, k in (("share", EV_SHARE), ("met", EV_MET)):
        m = np.flatnonzero(ev["kind"] == k)
        if m.size > FATE_SAMPLE:
            m = rng.choice(m, FATE_SAMPLE, replace=False)
        store[name].append(np.stack([ev["actor"][m], ev["step"][m]], axis=1))
    out: Dict[str, float] = {}
    death = world.genealogy.death
    now = world.step_count
    for name, chunks in store.items():
        allv = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.int64)
        ripe = allv[:, 1] <= now - FATE_HORIZON
        res = allv[ripe]
        store[name] = [allv[~ripe]]
        if res.size == 0:
            out[f"dies_soon_{name}"] = float("nan")
            continue
        d = death[res[:, 0]]
        out[f"dies_soon_{name}"] = float(np.mean((d >= 0) & (d - res[:, 1] <= FATE_HORIZON)))
    return out


def kinship_report(world, events: Dict[str, np.ndarray] = None) -> Dict[str, float]:
    ev = world.events.drain() if events is None else events
    nan = float("nan")
    out: Dict[str, float] = {}
    kind = ev["kind"]
    rnd = kind == EV_RANDOM
    d2_rand = float(ev["d2"][rnd].mean()) if rnd.any() else nan
    gen = world.genealogy
    r = np.random.default_rng([world.p.seed, world.step_count, 31])

    for k in (EV_MET, EV_SHARE, EV_ATTACK, EV_ALARM, EV_DEFEND, EV_RANDOM):
        name = EVENT_NAMES[k]
        m = np.flatnonzero(kind == k)
        out[f"n_{name}"] = int(m.size)
        if m.size == 0:
            for key in ("r", "gdist", "parent_child", "kin2", "close_kin", "tagsim"):
                out[f"{key}_{name}"] = nan
            continue
        out[f"r_{name}"] = 1.0 - float(ev["d2"][m].mean()) / d2_rand if d2_rand > 0 else nan
        out[f"tagsim_{name}"] = float(ev["sim"][m].mean())
        if m.size > MAX_PAIRS_FOR_GENEALOGY:
            m = r.choice(m, MAX_PAIRS_FOR_GENEALOGY, replace=False)
        dist = gen.distance(ev["actor"][m], ev["target"][m])
        related = dist >= 0
        out[f"gdist_{name}"] = float(np.median(dist[related])) if related.any() else nan
        out[f"parent_child_{name}"] = float(np.mean(dist == 1))
        out[f"kin2_{name}"] = float(np.mean(dist == 2))
        out[f"close_kin_{name}"] = float(np.mean((dist >= 1) & (dist <= 4)))
        if k == EV_SHARE:
            a, t = ev["actor"][m], ev["target"][m]
            out["share_to_child"] = float(np.mean(gen.parent[t] == a))
            out["share_to_parent"] = float(np.mean(gen.parent[a] == t))

    out.update(_fate(world, ev, r))
    p = world.p
    out["hamilton_rb"] = out["r_share"] * p.share_benefit if out["n_share"] else nan
    out["hamilton_c"] = p.share_cost
    out["hamilton_rb_minus_c"] = out["hamilton_rb"] - p.share_cost
    return out


# ---------------------------------------------------------------------------- Price


@njit(cache=True)
def _ancestor_index(ids, parent, snap_index):
    """For each id, the index (into the snapshot) of its ancestor alive at the snapshot."""
    out = np.empty(ids.size, np.int64)
    n_map = snap_index.size
    for k in range(ids.size):
        x = ids[k]
        while x >= 0 and (x >= n_map or snap_index[x] < 0):
            x = parent[x]
        out[k] = snap_index[x] if x >= 0 else -1
    return out


class PriceTracker:
    """Price equation for every behaviour gene between start (now) and finish().

        mean_after - mean_before = selection + transmission
        selection    = cov(w, z) / mean(w)       w = descendants alive at the end
        transmission = E[w * (z_desc - z)] / mean(w)
    The identity is exact (up to float rounding).
    """

    def __init__(self, world):
        a = np.flatnonzero(world.alive)
        self.step0 = world.step_count
        self.ids = world.id[a].copy()
        self.z = world.genome[a][:, G.BEHAVIOUR_GENES].copy()
        self.n_ids = world.next_id

    def finish(self, world) -> Dict[str, float]:
        a = np.flatnonzero(world.alive)
        ids_end = world.id[a]
        z_end = world.genome[a][:, G.BEHAVIOUR_GENES]
        snap_index = np.full(self.n_ids, -1, dtype=np.int64)
        snap_index[self.ids] = np.arange(self.ids.size)
        anc = _ancestor_index(ids_end, world.genealogy.parent, snap_index)
        ok = anc >= 0
        anc, z_end = anc[ok], z_end[ok]

        n0 = self.ids.size
        w = np.bincount(anc, minlength=n0).astype(float)
        out: Dict[str, float] = {"price_steps": world.step_count - self.step0,
                                 "price_mean_fitness": float(w.mean()) if n0 else float("nan")}
        if n0 == 0 or w.sum() == 0:
            return out
        wbar = w.mean()
        names = [G.GENE_NAMES[i] for i in G.BEHAVIOUR_GENES]
        for gi, name in enumerate(names):
            z = self.z[:, gi]
            zsum_desc = np.bincount(anc, weights=z_end[:, gi], minlength=n0)
            selection = float(np.mean((w - wbar) * (z - z.mean())) / wbar)
            transmission = float(np.sum(zsum_desc - w * z) / w.sum())
            out[f"dz_{name}"] = float(z_end[:, gi].mean() - z.mean())
            out[f"sel_{name}"] = selection
            out[f"trans_{name}"] = transmission
        return out


# ---------------------------------------------------------------------------- clans


@njit(cache=True)
def _find(uf, x):
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


@njit(cache=True)
def _kin_components(idx, nbr_start, nbr, slot_pos, ids, parent, depth, max_dist):
    n = idx.size
    uf = np.arange(n)
    for k in range(n):
        for s in range(nbr_start[k], nbr_start[k + 1]):
            kj = slot_pos[nbr[s]]
            if kj <= k:
                continue
            d = mrca_pair(ids[idx[k]], ids[idx[kj]], parent, depth)
            if 0 <= d <= max_dist:
                a = _find(uf, k)
                b = _find(uf, kj)
                if a != b:
                    uf[a] = b
    labels = np.empty(n, np.int64)
    for k in range(n):
        labels[k] = _find(uf, k)
    return labels


def clans_and_cheaters(world, max_dist: int = 4, min_size: int = 5,
                       cheat_ratio: float = 0.5, min_clan_share: float = 0.02) -> Dict[str, float]:
    p = world.p
    idx = np.flatnonzero(world.alive)
    nan = float("nan")
    if idx.size == 0:
        return {"n_clans": 0, "mean_clan_size": nan, "frac_in_clans": nan, "frac_cheaters": nan}
    nbr_start, nbr = K.neighbour_lists(idx, world.x, world.y, world.alive, p.world_size,
                                       world.nb, p.interaction_radius, world._buf)
    slot_pos = np.full(world.alive.size, -1, dtype=np.int64)
    slot_pos[idx] = np.arange(idx.size)
    labels = _kin_components(idx, nbr_start, nbr, slot_pos, world.id,
                             world.genealogy.parent, world.genealogy.depth, max_dist)
    _, inv, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    member_size = sizes[inv]
    in_clan = member_size >= min_size

    ps = G.propensity(world.genome[idx], G.SHARE_W)  # reference: kin, both at half threshold
    clan_mean = np.bincount(inv, weights=ps) / sizes
    my_clan_mean = clan_mean[inv]
    sharing_clan = in_clan & (my_clan_mean >= min_clan_share)
    cheater = sharing_clan & (ps < cheat_ratio * my_clan_mean)
    big = sizes >= min_size
    return {
        "n_clans": int(big.sum()),
        "mean_clan_size": float(sizes[big].mean()) if big.any() else nan,
        "frac_in_clans": float(in_clan.mean()),
        "frac_cheaters": float(cheater.sum() / sharing_clan.sum()) if sharing_clan.any() else nan,
    }
