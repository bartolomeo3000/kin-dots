import numpy as np
import pytest

from kin_dots.core import Params, World
from kin_dots.core import genome as G

SMALL = Params(world_size=60.0, grid_cells=15, init_agents=100, max_agents=3000)


def test_same_seed_same_run():
    a, b = World(SMALL), World(SMALL)
    for _ in range(200):
        a.step()
        b.step()
    assert a.population == b.population
    np.testing.assert_array_equal(a.alive, b.alive)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.energy, b.energy)
    np.testing.assert_array_equal(a.genome, b.genome)
    np.testing.assert_array_equal(a.food, b.food)


def test_different_seed_different_run():
    a, b = World(SMALL), World(SMALL.replace(seed=1))
    for _ in range(50):
        a.step()
        b.step()
    assert not np.array_equal(a.x, b.x)


@pytest.mark.parametrize("dispersal", [2.0, -1.0])
def test_energy_and_food_ledger_balances(dispersal):
    w = World(SMALL.replace(dispersal_sd=dispersal))
    for _ in range(300):
        e0, f0 = w.total_energy(), w.total_food()
        w.step()
        L = w.last
        expected_de = L["eaten"] - L["metabolic"] - L["movement"] - L["birth_cost"] - L["death_removed"]
        assert w.total_energy() - e0 == pytest.approx(expected_de, abs=1e-6)
        assert w.total_food() - f0 == pytest.approx(L["food_growth"] - L["eaten"], abs=1e-6)


def test_state_stays_valid():
    w = World(SMALL)
    for _ in range(300):
        w.step()
        a = w.alive
        assert (w.x[a] >= 0).all() and (w.x[a] < SMALL.world_size).all()
        assert (w.y[a] >= 0).all() and (w.y[a] < SMALL.world_size).all()
        assert (w.energy[a] > 0).all()
        assert (w.food >= 0).all() and (w.food <= SMALL.food_capacity).all()
        assert ((w.genome[a, G.TAG] >= 0) & (w.genome[a, G.TAG] <= 1)).all()
        assert len(set(w.id[a])) == a.sum()


def test_eating_shares_food_like_sequential_eating():
    w = World(SMALL.replace(init_agents=3, food_growth=0.0, food_inflow=0.0))
    w.x[:3] = w.y[:3] = 1.0  # all three in cell (0, 0)
    w.genome[:3, G.HARVEST] = [0.5, 0.25, 0.1]
    w.food[0, 0] = 8.0
    e0 = w.energy[:3].copy()
    eaten = w._eat(np.arange(3))
    expected = 8.0 * (1 - 0.5 * 0.75 * 0.9)
    assert eaten == pytest.approx(expected)
    assert w.food[0, 0] == pytest.approx(8.0 - expected)
    gains = w.energy[:3] - e0
    np.testing.assert_allclose(gains / gains.sum(), np.array([0.5, 0.25, 0.1]) / 0.85)


def test_child_inherits_and_records_parent():
    w = World(SMALL.replace(init_agents=1, birth_threshold=5.0, init_energy=10.0))
    w._reproduce()
    assert w.population == 2
    child = np.flatnonzero(w.alive)[1]
    assert w.parent_id[child] == w.id[0]
    assert w.energy[0] + w.energy[child] == pytest.approx(10.0 - SMALL.birth_cost)


def test_global_dispersal_ignores_parent_position():
    p = SMALL.replace(init_agents=1, birth_threshold=1.0, init_energy=1e6, dispersal_sd=-1.0)
    w = World(p)
    w.x[0] = w.y[0] = 0.5
    for _ in range(8):
        w._reproduce()
    a = w.alive
    assert w.x[a].max() > 10 and w.y[a].max() > 10


def test_params_yaml_roundtrip(tmp_path):
    p = Params(seed=7, dispersal_sd=-1.0)
    path = tmp_path / "p.yaml"
    p.to_yaml(path)
    assert Params.from_yaml(path) == p


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        Params.from_dict({"no_such_param": 1})
