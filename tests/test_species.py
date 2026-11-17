import numpy as np
import pytest

from kin_dots.core import Params, World
from kin_dots.core import genome as G
from kin_dots.core.kinship import kinship_report

SMALL = Params(world_size=60.0, grid_cells=15, init_agents=100, max_agents=3000)
TWO = SMALL.replace(n_species=2, defense_enabled=True)


def _agents(n, species, xs, **over):
    w = World(TWO.replace(init_agents=n, **over))
    a = np.arange(n)
    w.x[a] = xs
    w.y[a] = 10.0
    w.species[a] = species
    w.energy[a] = 10.0
    w.genome[a, G.SHARE_W] = [-50, 0, 0, 0]
    w.genome[a, G.ATTACK_W] = [-50, 0, 0, 0]
    w.genome[a, G.DEFEND_W] = [-50, 0, 0, 0]
    return w


def test_halves_init_and_inheritance():
    w = World(TWO)
    a = np.flatnonzero(w.alive)
    assert ((w.x[a] >= 30.0) == (w.species[a] == 1)).all()
    for _ in range(200):
        w.step()
    kids = np.flatnonzero(w.alive & (w.parent_id >= 0))
    assert kids.size > 0
    # every living agent's species equals its parent's (parents may be dead: use genealogy)
    alive_ids = dict(zip(w.id[w.alive].tolist(), w.species[w.alive].tolist()))
    for k in kids:
        pid = w.parent_id[k]
        if pid in alive_ids:
            assert alive_ids[pid] == w.species[k]


@pytest.mark.parametrize("same_species,within,expect_attack", [
    (True, False, 0),   # same species, fights only across species: no attack
    (True, True, 1),    # control: fights within species allowed
    (False, False, 1),  # different species: attack
])
def test_who_can_be_attacked(same_species, within, expect_attack):
    w = _agents(2, [0, 0 if same_species else 1], [10.0, 11.0], fights_within_species=within)
    w.genome[0, G.ATTACK_W] = [50, 0, 0, 0]
    out = w._interact(np.arange(2))
    assert out["attacks"] == expect_attack


def test_no_gifts_to_the_other_species():
    w = _agents(2, [0, 1], [10.0, 11.0])
    w.genome[0, G.SHARE_W] = [50, 0, 0, 0]
    assert w._interact(np.arange(2))["shares"] == 0
    w.species[1] = 0
    assert w._interact(np.arange(2))["shares"] == 1


def test_defender_takes_the_fight():
    # attacker (species 1) - victim (0) - defender (0); attacker only reaches the victim
    w = _agents(3, [1, 0, 0], [10.0, 13.5, 17.0])
    w.genome[0, G.ATTACK_W] = [50, 0, 0, 0]
    w.genome[2, G.DEFEND_W] = [50, 0, 0, 0]
    out = w._interact(np.array([0, 1, 2]))
    assert out["attacks"] == 1 and out["defenses"] == 1 and out["killed"] == 1
    assert w.alive[1]                       # the victim is untouched
    assert w.alive[0] != w.alive[2]         # attacker or defender died
    assert out["defender_died"] == (0 if w.alive[2] else 1)


def test_ledger_balances_with_species_and_defense():
    w = World(TWO)
    a = w.alive
    w.genome[a, G.ATTACK_W] = [0, 0, 0, 0]
    w.genome[a, G.DEFEND_W] = [0, 0, 0, 0]
    saw_defense = False
    for _ in range(200):
        e0 = w.total_energy()
        w.step()
        L = w.last
        de = (L["eaten"] - L["metabolic"] - L["movement"] - L["birth_cost"] - L["death_removed"]
              - L["share_given"] + L["share_received"] - L["attack_cost"] - L["killed_energy"]
              + L["predation_gain"] - L["alarm_cost"] - L["predated_energy"])
        assert w.total_energy() - e0 == pytest.approx(de, abs=1e-6)
        assert L["cross_attacks"] == L["attacks"]
        saw_defense |= L["defenses"] > 0
    assert saw_defense


def test_kinship_report_with_two_species():
    w = World(TWO)
    for _ in range(150):
        w.step()
    rep = kinship_report(w)
    assert rep["r_random"] == pytest.approx(0.0, abs=1e-12)
    assert "r_defend" in rep


def test_single_species_runs_are_unchanged_by_species_code():
    a = World(SMALL)
    b = World(SMALL.replace(fights_within_species=True, species_init="mixed"))  # ignored with 1 species
    for _ in range(100):
        a.step()
        b.step()
    np.testing.assert_array_equal(a.genome, b.genome)
