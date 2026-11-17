"""Predators and alarm calls (Numba).

Predators are simple, non-evolving hunters. Each step:

1. Detection and alarms (prey, in random order). A prey with a predator within
   `detect_radius` notices it with probability `detect_prob` and hides
   (`alert` > 0 makes it harder to catch). It may then call an alarm, decided by
   its ALARM genes from: mean tag similarity of its neighbours, its energy and the
   number of neighbours. A call warns every neighbour (they hide too) but the
   caller pays: energy, it cannot hide this step, and predators that see it chase
   it first.
2. Predators. A resting predator (after a kill) wanders. Otherwise it picks the
   nearest caller in sight, else the nearest prey in sight, moves towards it and
   strikes if close enough. The strike kills with catch_prob (times
   hidden_catch_factor if the prey hides). A kill starts a rest of `handling` steps.

All randomness comes from `seed` (drawn from the World's generator), so runs
stay deterministic.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .kernels import _sigmoid, _torus_delta, tag_similarity

P_DETECTIONS = 0       # prey that noticed a predator
P_ALARMS = 1           # alarm calls
P_WARNED = 2           # agents warned by calls
P_SIM_WARNED = 3       # sum of tag similarity caller-warned
P_ALARM_COST = 4       # energy paid for calls
P_STRIKES = 5          # predator strikes
P_KILLS = 6            # prey killed by predators
P_KILLED_ENERGY = 7    # energy the killed prey had (removed)
P_CALLERS_KILLED = 8   # kills of agents that called this step
P_HIDDEN_KILLED = 9    # kills of hiding agents
N_PLEDGER = 10


@njit(cache=True)
def gather_within(qx, qy, exclude, x, y, alive, start, order, world_size, nb, radius, buf):
    """Living agents within `radius` of point (qx, qy), except `exclude`; returns count."""
    cs = world_size / nb
    reach = int(np.ceil(radius / cs))
    bx = int(qx / cs) % nb
    by = int(qy / cs) % nb
    r2 = radius * radius
    cnt = 0
    full = 2 * reach + 1 >= nb  # radius covers the whole map: visit each bucket once
    lo = 0 if full else -reach
    hi = nb if full else reach + 1
    for ox in range(lo, hi):
        cx = (ox if full else (bx + ox) % nb) * nb
        for oy in range(lo, hi):
            b = cx + (oy if full else (by + oy) % nb)
            for s in range(start[b], start[b + 1]):
                j = order[s]
                if j == exclude or not alive[j]:
                    continue
                dx = _torus_delta(qx, x[j], world_size)
                dy = _torus_delta(qy, y[j], world_size)
                if dx * dx + dy * dy <= r2:
                    buf[cnt] = j
                    cnt += 1
    return cnt


@njit(cache=True)
def _wander(p, px, py, heading, speed, world_size):
    heading[p] += np.random.normal() * 0.5
    px[p] = (px[p] + np.cos(heading[p]) * speed) % world_size
    py[p] = (py[p] + np.sin(heading[p]) * speed) % world_size


@njit(cache=True)
def predation(seed, idx, x, y, alive, energy, genome, alarm0, tag0, alert, caller,
              px, py, pcool, heading, start, order, world_size, nb, interaction_radius,
              energy_scale, alarm_to_random, pred_speed, pred_sight, catch_radius, handling,
              catch_prob, hidden_factor, detect_radius, detect_prob, alert_duration, alarm_cost,
              target_callers, buf, ledger, log_i, log_j, log_n):
    np.random.seed(seed)
    n = idx.size
    npred = px.size
    dr2 = detect_radius * detect_radius

    # ---- 1. detection and alarm calls
    perm = np.random.permutation(n)
    for m in range(n):
        i = idx[perm[m]]
        if not alive[i]:
            continue
        near = False
        for p in range(npred):
            dx = _torus_delta(x[i], px[p], world_size)
            dy = _torus_delta(y[i], py[p], world_size)
            if dx * dx + dy * dy <= dr2:
                near = True
                break
        if not near or np.random.random() >= detect_prob:
            continue
        ledger[P_DETECTIONS] += 1
        if alert[i] < alert_duration:
            alert[i] = alert_duration

        cnt = gather_within(x[i], y[i], i, x, y, alive, start, order, world_size, nb,
                            interaction_radius, buf)
        msim = 0.0
        for k in range(cnt):
            msim += tag_similarity(genome, tag0, i, buf[k])
        if cnt > 0:
            msim /= cnt
        z = (genome[i, alarm0] + genome[i, alarm0 + 1] * msim
             + genome[i, alarm0 + 2] * min(energy[i] / energy_scale, 2.0)
             + genome[i, alarm0 + 3] * cnt / 10.0)
        if np.random.random() >= _sigmoid(z):
            continue

        ledger[P_ALARMS] += 1
        energy[i] -= alarm_cost
        ledger[P_ALARM_COST] += alarm_cost
        caller[i] = 1
        if alarm_to_random:  # control: as many random agents as there were neighbours
            got = 0
            for _ in range(4 * cnt):
                if got == cnt:
                    break
                j = idx[np.random.randint(n)]
                if j != i and alive[j]:
                    buf[got] = j
                    got += 1
            cnt = got
        for k in range(cnt):
            j = buf[k]
            if alert[j] < alert_duration:
                alert[j] = alert_duration
            ledger[P_WARNED] += 1
            ledger[P_SIM_WARNED] += tag_similarity(genome, tag0, i, j)
            ln = log_n[0]
            if ln < log_i.size:
                log_i[ln] = i
                log_j[ln] = j
                log_n[0] = ln + 1

    # ---- 2. predators hunt
    for p in range(npred):
        if pcool[p] > 0:
            pcool[p] -= 1
            _wander(p, px, py, heading, pred_speed, world_size)
            continue
        cnt = gather_within(px[p], py[p], -1, x, y, alive, start, order, world_size, nb,
                            pred_sight, buf)
        best = -1
        best_d2 = 1e18
        best_caller = False
        for k in range(cnt):
            j = buf[k]
            dx = _torus_delta(px[p], x[j], world_size)
            dy = _torus_delta(py[p], y[j], world_size)
            d2 = dx * dx + dy * dy
            c = target_callers and caller[j] == 1
            if (c and not best_caller) or (c == best_caller and d2 < best_d2):
                best, best_d2, best_caller = j, d2, c
        if best < 0:
            _wander(p, px, py, heading, pred_speed, world_size)
            continue
        dx = _torus_delta(px[p], x[best], world_size)
        dy = _torus_delta(py[p], y[best], world_size)
        dist = np.sqrt(dx * dx + dy * dy)
        step = min(pred_speed, dist)
        if dist > 0:
            px[p] = (px[p] + dx / dist * step) % world_size
            py[p] = (py[p] + dy / dist * step) % world_size
            heading[p] = np.arctan2(dy, dx)
        if dist - step > catch_radius:
            continue
        ledger[P_STRIKES] += 1
        hidden = alert[best] > 0 and caller[best] == 0
        pc = catch_prob * (hidden_factor if hidden else 1.0)
        if np.random.random() < pc:
            ledger[P_KILLS] += 1
            ledger[P_KILLED_ENERGY] += energy[best]
            if caller[best] == 1:
                ledger[P_CALLERS_KILLED] += 1
            if hidden:
                ledger[P_HIDDEN_KILLED] += 1
            energy[best] = 0.0
            alive[best] = False
            pcool[p] = handling
