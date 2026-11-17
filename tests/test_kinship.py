import numpy as np
import pytest

from kin_dots.core import Params, World
from kin_dots.core import genome as G
from kin_dots.core.genealogy import EV_RANDOM, EV_SHARE, EventLog, Genealogy
from kin_dots.core.kinship import PriceTracker, clans_and_cheaters, kinship_report

SMALL = Params(world_size=60.0, grid_cells=15, init_agents=100, max_agents=3000)


def _tree():
    #        0           1        (founders)
    #      /   \         |
    #     2     3        4
    #    / \    |
    #   5   6   7
    g = Genealogy(capacity=4)  # small capacity to exercise growing
    g.add(np.array([0, 1]), np.array([-1, -1]), 0)
    g.add(np.array([2, 3, 4]), np.array([0, 0, 1]), 1)
    g.add(np.array([5, 6, 7]), np.array([2, 2, 3]), 2)
    return g


@pytest.mark.parametrize("a,b,d", [
    (5, 5, 0),   # self
    (5, 2, 1),   # child - parent
    (5, 6, 2),   # siblings
    (5, 0, 2),   # grandchild - grandparent
    (5, 3, 3),   # niece - uncle
    (5, 7, 4),   # cousins
    (5, 4, -1),  # different founders
])
def test_genealogical_distance(a, b, d):
    g = _tree()
    assert g.distance(np.array([a]), np.array([b]))[0] == d
    assert g.distance(np.array([b]), np.array([a]))[0] == d
    assert list(g.depth[:8]) == [0, 0, 1, 1, 1, 2, 2, 2]


def test_event_log_ring_keeps_newest_and_drains():
    log = EventLog(capacity=5)
    for start in (0, 3):
        n = 3
        log.append(kind=np.zeros(n, np.int8), actor=np.arange(start, start + n), target=np.zeros(n),
                   d2=np.zeros(n), sim=np.zeros(n), actor_e=np.zeros(n), target_e=np.zeros(n))
    ev = log.drain()
    assert list(ev["actor"]) == [1, 2, 3, 4, 5]
    assert log.drain()["actor"].size == 0


def test_world_genealogy_matches_parent_ids():
    w = World(SMALL)
    for _ in range(200):
        w.step()
    a = np.flatnonzero(w.alive & (w.parent_id >= 0))
    assert (w.genealogy.parent[w.id[a]] == w.parent_id[a]).all()
    assert w.genealogy.n == w.next_id


def test_measuring_does_not_change_the_simulation():
    a, b = World(SMALL), World(SMALL)
    for s in range(150):
        a.step()
        b.step()
        if s % 10 == 0:
            kinship_report(b)
            clans_and_cheaters(b)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.genome, b.genome)


def test_random_pairs_have_zero_relatedness_on_average():
    w = World(SMALL)
    for _ in range(200):
        w.step()
    rep = kinship_report(w)
    assert rep["r_random"] == pytest.approx(0.0, abs=1e-12)  # by definition
    assert rep["n_random"] > 0 and rep["n_met"] > 0


def test_neighbours_are_more_related_than_random_pairs():
    w = World(SMALL.replace(move_speed=0.1, dispersal_sd=1.0))
    for _ in range(600):
        w.step()
    w.events.drain()
    for _ in range(50):
        w.step()
    rep = kinship_report(w)
    assert rep["r_met"] > 0.1
    assert rep["close_kin_met"] > rep["close_kin_random"]


def test_share_to_own_child_is_detected():
    w = World(SMALL.replace(init_agents=1, birth_threshold=5.0, init_energy=10.0))
    w._reproduce()
    a = np.flatnonzero(w.alive)
    w.x[a] = [10.0, 10.5]
    w.y[a] = [10.0, 10.0]
    w.energy[a] = 10.0
    w.genome[a, G.ATTACK_W] = [-50, 0, 0, 0]
    w.genome[a[0], G.SHARE_W] = [50, 0, 0, 0]   # the parent always shares
    w.genome[a[1], G.SHARE_W] = [-50, 0, 0, 0]
    w.events.drain()
    w._interact(a)
    rep = kinship_report(w)
    assert rep["n_share"] == 1
    assert rep["parent_child_share"] == 1.0 and rep["share_to_child"] == 1.0


def test_price_identity_holds():
    w = World(SMALL.replace(mutation_sd=0.1))
    for _ in range(100):
        w.step()
    pt = PriceTracker(w)
    for _ in range(150):
        w.step()
    out = pt.finish(w)
    for i in G.BEHAVIOUR_GENES:
        name = G.GENE_NAMES[i]
        assert out[f"dz_{name}"] == pytest.approx(out[f"sel_{name}"] + out[f"trans_{name}"], abs=1e-9)


def test_two_separate_families_form_two_clans():
    p = SMALL.replace(init_agents=2, birth_threshold=1.0, init_energy=1e4, dispersal_sd=0.5)
    w = World(p)
    w.x[:2] = [10.0, 40.0]
    w.y[:2] = [10.0, 40.0]
    for _ in range(3):  # 2 -> 4 -> 8 -> 16 agents, two families
        w._reproduce()
    w.genome[w.alive, G.SHARE_W] = [0, 0, 0, 0]  # p(share) = 0.5 for everyone: no cheaters
    out = clans_and_cheaters(w)
    assert out["n_clans"] == 2
    assert out["frac_in_clans"] == 1.0
    assert out["frac_cheaters"] == 0.0
