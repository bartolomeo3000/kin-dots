"""Numba kernels for everything that needs neighbours.

Once per step the World builds neighbour lists for all living agents:
1. a spatial hash (the torus split into nb x nb buckets of side >= radius, so all
   neighbours within `radius` are in the 3x3 buckets around an agent),
2. from it, CSR neighbour lists: neighbours of the k-th living agent are
   nbr[nbr_start[k]:nbr_start[k+1]] (agent slot numbers).
Both the interaction phase and the movement phase reuse these lists.

Kernels never draw random numbers themselves; all randomness is passed in from
the World's NumPy generator, which keeps runs deterministic.
Genome columns are passed as integer offsets so the genome is never copied.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Indices into the ledger array filled by `interact`.
L_SHARES = 0          # number of donations
L_SHARE_GIVEN = 1     # energy paid by donors (n * cost)
L_SHARE_RECEIVED = 2  # energy received by recipients (n * benefit)
L_ATTACKS = 3         # number of attacks
L_ATTACK_COST = 4     # energy paid by attackers for attacking
L_KILLED = 5          # number of agents killed in fights
L_KILLED_ENERGY = 6   # energy the killed agents had (removed from the world)
L_PREDATION_GAIN = 7  # energy winners got from the killed
L_ATTACKER_LOST = 8   # fights the attacker lost (and died)
L_ENCOUNTERS = 9      # interactions considered (agent had >= 1 neighbour)
L_SIM_ENCOUNTER = 10  # sum of tag similarity over encounters
L_SIM_SHARE = 11      # sum of tag similarity over donations
L_SIM_ATTACK = 12     # sum of tag similarity over attacks
L_DONOR_E = 13        # sum of donors' energy just before donating
L_RECIPIENT_E = 14    # sum of recipients' energy just before receiving
N_LEDGER = 15

INV_SQRT3 = 1.0 / np.sqrt(3.0)


@njit(cache=True)
def build_hash(x, y, idx, world_size, nb):
    """Counting sort of agents `idx` into nb*nb buckets.

    Returns (start, order): the agents in bucket b are order[start[b]:start[b+1]].
    """
    n = idx.size
    cs = world_size / nb
    bucket = np.empty(n, np.int64)
    counts = np.zeros(nb * nb + 1, np.int64)
    for k in range(n):
        i = idx[k]
        b = (int(x[i] / cs) % nb) * nb + int(y[i] / cs) % nb
        bucket[k] = b
        counts[b + 1] += 1
    start = np.cumsum(counts)
    fill = start[:-1].copy()
    order = np.empty(n, np.int64)
    for k in range(n):
        b = bucket[k]
        order[fill[b]] = idx[k]
        fill[b] += 1
    return start, order


@njit(cache=True, inline="always")
def _torus_delta(a, b, size):
    d = b - a
    if d > 0.5 * size:
        d -= size
    elif d < -0.5 * size:
        d += size
    return d


@njit(cache=True)
def gather_neighbours(i, x, y, alive, start, order, world_size, nb, radius, buf):
    """Write the living neighbours of agent i (within radius) into buf; return how many."""
    cs = world_size / nb
    bx = int(x[i] / cs) % nb
    by = int(y[i] / cs) % nb
    r2 = radius * radius
    cnt = 0
    for ox in range(nb - 1, nb + 2):          # == -1, 0, 1 but keeps the modulo positive
        cx = ((bx + ox) % nb) * nb
        for oy in range(nb - 1, nb + 2):
            b = cx + (by + oy) % nb
            for s in range(start[b], start[b + 1]):
                j = order[s]
                if j == i or not alive[j]:
                    continue
                dx = _torus_delta(x[i], x[j], world_size)
                dy = _torus_delta(y[i], y[j], world_size)
                if dx * dx + dy * dy <= r2:
                    buf[cnt] = j
                    cnt += 1
    return cnt


@njit(cache=True)
def neighbour_lists(idx, x, y, alive, world_size, nb, radius, buf):
    """CSR neighbour lists for every agent in idx (see module docstring)."""
    start, order = build_hash(x, y, idx, world_size, nb)
    n = idx.size
    nbr_start = np.zeros(n + 1, np.int64)
    total = 0
    cap = 8 * n + 64
    nbr = np.empty(cap, np.int64)
    for k in range(n):
        cnt = gather_neighbours(idx[k], x, y, alive, start, order, world_size, nb, radius, buf)
        if total + cnt > cap:
            cap = 2 * (total + cnt)
            grown = np.empty(cap, np.int64)
            grown[:total] = nbr[:total]
            nbr = grown
        nbr[total:total + cnt] = buf[:cnt]
        total += cnt
        nbr_start[k + 1] = total
    return nbr_start, nbr[:total]


@njit(cache=True, inline="always")
def tag_similarity(genome, tag0, i, j):
    """1 for identical tags, 0 for opposite corners of the RGB cube."""
    d0 = genome[i, tag0] - genome[j, tag0]
    d1 = genome[i, tag0 + 1] - genome[j, tag0 + 1]
    d2 = genome[i, tag0 + 2] - genome[j, tag0 + 2]
    return 1.0 - np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) * INV_SQRT3


@njit(cache=True, inline="always")
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@njit(cache=True)
def social_vectors(idx, x, y, alive, genome, tag0, nbr_start, nbr, world_size):
    """For each agent: mean unit vector towards neighbours, weighted by similarity
    (kin) and by dissimilarity (strangers). Returns two (n, 2) arrays."""
    n = idx.size
    kin = np.zeros((n, 2))
    strangers = np.zeros((n, 2))
    for k in range(n):
        i = idx[k]
        if not alive[i]:
            continue
        cnt = 0
        for s in range(nbr_start[k], nbr_start[k + 1]):
            j = nbr[s]
            if not alive[j]:
                continue
            dx = _torus_delta(x[i], x[j], world_size)
            dy = _torus_delta(y[i], y[j], world_size)
            dist = np.sqrt(dx * dx + dy * dy)
            if dist == 0.0:
                continue
            ux = dx / dist
            uy = dy / dist
            sim = tag_similarity(genome, tag0, i, j)
            kin[k, 0] += sim * ux
            kin[k, 1] += sim * uy
            strangers[k, 0] += (1.0 - sim) * ux
            strangers[k, 1] += (1.0 - sim) * uy
            cnt += 1
        if cnt:
            kin[k, 0] /= cnt
            kin[k, 1] /= cnt
            strangers[k, 0] /= cnt
            strangers[k, 1] /= cnt
    return kin, strangers


@njit(cache=True)
def interact(acting_k, idx, energy, alive, genome, share0, attack0, tag0, tag_blind_share, attacks_enabled,
             share_to_random,
             nbr_start, nbr, energy_scale, share_cost, share_benefit, attack_cost,
             predation_efficiency, u, buf, ledger):
    """Agents meet one random living neighbour each and may attack it or donate energy.

    acting_k: shuffled positions into idx (the order in which agents act).
    Decision inputs: [1, tag similarity, my energy / scale, their energy / scale],
    energies clipped to [0, 2] after scaling.
    Attack is considered first; if it does not happen, sharing is considered.
    Fight: attacker wins with probability e_a / (e_a + e_t); the loser dies and the
    winner gets predation_efficiency * the loser's energy.
    `alive` and `energy` are modified in place.
    u: (len(acting_k), 5) uniforms for [pick neighbour, attack?, share?, fight outcome, random recipient].
    share_to_random: the sharing decision and transfer use a random living agent from
    idx instead of the neighbour (control without kin structure).
    """
    for m in range(acting_k.size):
        k = acting_k[m]
        i = idx[k]
        if not alive[i]:
            continue  # killed earlier this step
        cnt = 0
        for s in range(nbr_start[k], nbr_start[k + 1]):
            if alive[nbr[s]]:
                buf[cnt] = nbr[s]
                cnt += 1
        if cnt == 0:
            continue
        j = buf[min(int(u[m, 0] * cnt), cnt - 1)]
        sim = tag_similarity(genome, tag0, i, j)
        ei = min(energy[i] / energy_scale, 2.0)  # clipped so large energies do not
        ej = min(energy[j] / energy_scale, 2.0)  # saturate the decision
        ledger[L_ENCOUNTERS] += 1
        ledger[L_SIM_ENCOUNTER] += sim

        z_att = (genome[i, attack0] + genome[i, attack0 + 1] * sim
                 + genome[i, attack0 + 2] * ei + genome[i, attack0 + 3] * ej)
        if attacks_enabled and u[m, 1] < _sigmoid(z_att):
            ledger[L_ATTACKS] += 1
            ledger[L_SIM_ATTACK] += sim
            energy[i] -= attack_cost
            ledger[L_ATTACK_COST] += attack_cost
            ea = max(energy[i], 0.0)
            et = max(energy[j], 0.0)
            p_win = ea / (ea + et) if ea + et > 0 else 0.0
            if u[m, 3] < p_win:
                winner, loser = i, j
            else:
                winner, loser = j, i
                ledger[L_ATTACKER_LOST] += 1
            el = energy[loser]
            gain = predation_efficiency * max(el, 0.0)
            energy[winner] += gain
            ledger[L_PREDATION_GAIN] += gain
            ledger[L_KILLED_ENERGY] += el
            ledger[L_KILLED] += 1
            energy[loser] = 0.0
            alive[loser] = False
            continue

        if share_to_random:
            j = idx[min(int(u[m, 4] * idx.size), idx.size - 1)]
            if j == i or not alive[j]:
                continue
            sim = tag_similarity(genome, tag0, i, j)
            ej = min(energy[j] / energy_scale, 2.0)
        s_in = 0.0 if tag_blind_share else sim
        z_sh = (genome[i, share0] + genome[i, share0 + 1] * s_in
                + genome[i, share0 + 2] * ei + genome[i, share0 + 3] * ej)
        if energy[i] > share_cost and u[m, 2] < _sigmoid(z_sh):
            ledger[L_DONOR_E] += energy[i]
            ledger[L_RECIPIENT_E] += energy[j]
            energy[i] -= share_cost
            energy[j] += share_benefit
            ledger[L_SHARES] += 1
            ledger[L_SHARE_GIVEN] += share_cost
            ledger[L_SHARE_RECEIVED] += share_benefit
            ledger[L_SIM_SHARE] += sim
