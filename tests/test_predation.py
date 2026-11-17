import numpy as np
import pytest

from kin_dots.core import Params, World
from kin_dots.core import genome as G
from kin_dots.core.kinship import kinship_report

SMALL = Params(world_size=60.0, grid_cells=15, init_agents=100, max_agents=3000)
PRED = SMALL.replace(n_predators=4)


def _scene(n_prey, **over):
    """n_prey agents and one predator, no randomness in detection or catching."""
    p = SMALL.replace(init_agents=n_prey, n_predators=1, detect_prob=1.0, catch_prob=1.0,
                      hidden_catch_factor=0.0, **over)
    w = World(p)
    a = np.arange(n_prey)
    w.energy[a] = 10.0
    w.genome[a, G.ATTACK_W] = [-50, 0, 0, 0]
    w.genome[a, G.SHARE_W] = [-50, 0, 0, 0]
    w.genome[a, G.ALARM_W] = [-50, 0, 0, 0]   # nobody calls unless a test says so
    w.pcool[:] = 0
    return w


def test_ledger_balances_with_predators_and_alarms():
    w = World(PRED.replace(detect_prob=0.8))
    w.genome[w.alive, G.ALARM_W] = [0, 0, 0, 0]  # call half the time
    saw_kill = saw_alarm = False
    for _ in range(300):
        e0 = w.total_energy()
        w.step()
        L = w.last
        de = (L["eaten"] - L["metabolic"] - L["movement"] - L["birth_cost"] - L["death_removed"]
              - L["share_given"] + L["share_received"] - L["attack_cost"] - L["killed_energy"]
              + L["predation_gain"] - L["alarm_cost"] - L["predated_energy"])
        assert w.total_energy() - e0 == pytest.approx(de, abs=1e-6)
        saw_kill |= L["predated"] > 0
        saw_alarm |= L["alarms"] > 0
    assert saw_kill and saw_alarm
    assert w.total_predated > 0


def test_predators_are_deterministic():
    a, b = World(PRED), World(PRED)
    for _ in range(200):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.alive, b.alive)
    np.testing.assert_array_equal(a.px, b.px)


def test_unprotected_prey_next_to_predator_is_caught():
    w = _scene(1, detect_radius=0.0)  # cannot notice the predator, so does not hide
    w.x[0], w.y[0] = 10.0, 10.0
    w.px[0], w.py[0] = 10.5, 10.0
    out = w._predation(np.arange(1))
    assert out["predated"] == 1 and not w.alive[0]
    assert w.pcool[0] == w.p.predator_handling


def test_hidden_prey_survives_when_hiding_is_perfect():
    w = _scene(1)  # detects (prob 1) and hides; hidden_catch_factor = 0
    w.x[0], w.y[0] = 10.0, 10.0
    w.px[0], w.py[0] = 10.5, 10.0
    out = w._predation(np.arange(1))
    assert out["detections"] == 1 and out["strikes"] == 1 and out["predated"] == 0
    assert w.alive[0]


def test_alarm_warns_neighbours_and_exposes_the_caller():
    w = _scene(3)
    w.x[:3] = [10.0, 12.0, 30.0]   # agent 2 is far from everyone
    w.y[:3] = [10.0, 10.0, 30.0]
    w.genome[0, G.ALARM_W] = [50, 0, 0, 0]  # agent 0 always calls
    w.p = w.p.replace(detect_radius=3.0)    # only agent 0 is close enough to notice
    w.px[0], w.py[0] = 9.0, 10.0            # predator right next to the caller
    out = w._predation(np.arange(3))
    assert out["alarms"] == 1 and out["warned"] == 1   # agent 1 is the only neighbour
    assert w.alert[1] > 0 and w.alert[2] == 0
    assert out["predated"] == 1 and out["callers_predated"] == 1 and not w.alive[0]
    assert w.alive[1]


def test_predator_chases_caller_before_closer_prey():
    w = _scene(2, detect_radius=8.0)
    w.x[:2] = [10.0, 14.0]
    w.y[:2] = [10.0, 10.0]
    w.genome[1, G.ALARM_W] = [50, 0, 0, 0]  # the farther agent calls
    w.px[0], w.py[0] = 9.0, 10.0
    w._predation(np.arange(2))
    assert w.px[0] > 9.0  # moved towards the caller (to the right), not just struck agent 0
    assert w.alive[0]


def test_alarm_to_random_warns_far_agents():
    w = _scene(3, alarm_to_random=True)
    w.x[:3] = [10.0, 12.0, 40.0]
    w.y[:3] = [10.0, 10.0, 40.0]
    w.genome[0, G.ALARM_W] = [50, 0, 0, 0]
    w.p = w.p.replace(detect_radius=3.0)
    warned_far = False
    for s in range(30):
        w.alert[:] = 0
        w.px[0], w.py[0] = 9.0, 30.0  # out of reach: nobody dies, the call still happens
        w.p = w.p.replace(detect_radius=25.0)
        w.alive[:3] = True
        w.energy[:3] = 10.0
        w._predation(np.arange(3))
        warned_far |= w.alert[2] > 0 and not (np.hypot(w.x[2] - 9.0, w.y[2] - 30.0) <= 25.0)
    assert warned_far


def test_alarm_events_reach_kinship_report():
    w = World(PRED.replace(detect_prob=1.0))
    w.genome[w.alive, G.ALARM_W] = [5, 0, 0, 0]
    for _ in range(100):
        w.step()
    rep = kinship_report(w)
    assert rep["n_alarm"] > 0 and rep["r_alarm"] == rep["r_alarm"]  # not NaN
