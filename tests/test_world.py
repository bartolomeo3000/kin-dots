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


def _make_social(w, share_bias=0.0, attack_bias=-2.0):
    """Turn sharing and attacking on for the whole population (to exercise them)."""
    a = w.alive
    w.genome[a, G.SHARE_W] = [share_bias, 0, 0, 0]
    w.genome[a, G.ATTACK_W] = [attack_bias, 0, 0, 0]


@pytest.mark.parametrize("dispersal,social", [(2.0, False), (-1.0, False), (2.0, True)])
def test_energy_and_food_ledger_balances(dispersal, social):
    w = World(SMALL.replace(dispersal_sd=dispersal))
    if social:
        _make_social(w)
    saw_share = saw_kill = False
    for _ in range(300):
        e0, f0 = w.total_energy(), w.total_food()
        w.step()
        L = w.last
        expected_de = (L["eaten"] - L["metabolic"] - L["movement"] - L["birth_cost"] - L["death_removed"]
                       - L["share_given"] + L["share_received"]
                       - L["attack_cost"] - L["killed_energy"] + L["predation_gain"])
        assert w.total_energy() - e0 == pytest.approx(expected_de, abs=1e-6)
        assert w.total_food() - f0 == pytest.approx(L["food_growth"] - L["eaten"], abs=1e-6)
        saw_share |= L["shares"] > 0
        saw_kill |= L["killed"] > 0
    if social:
        assert saw_share and saw_kill


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


# ---------------------------------------------------------------- M2: interactions

def _two_agents(**overrides):
    """Two agents 1 unit apart, sharing/attack off, no randomness in who meets whom."""
    w = World(SMALL.replace(init_agents=2, **overrides))
    w.x[:2] = [10.0, 11.0]
    w.y[:2] = [10.0, 10.0]
    w.energy[:2] = [10.0, 10.0]
    w.genome[:2, G.SHARE_W] = [-50, 0, 0, 0]
    w.genome[:2, G.ATTACK_W] = [-50, 0, 0, 0]
    return w


def test_neighbour_search_matches_brute_force():
    from kin_dots.core import kernels as K
    w = World(SMALL.replace(init_agents=500))
    idx = np.flatnonzero(w.alive)
    p = w.p
    start, order = K.build_hash(w.x, w.y, idx, p.world_size, w.nb)
    for i in idx[:50]:
        n = K.gather_neighbours(i, w.x, w.y, w.alive, start, order, p.world_size, w.nb,
                                p.interaction_radius, w._buf)
        got = set(w._buf[:n].tolist())
        dx = np.abs(w.x[idx] - w.x[i]); dx = np.minimum(dx, p.world_size - dx)
        dy = np.abs(w.y[idx] - w.y[i]); dy = np.minimum(dy, p.world_size - dy)
        near = idx[(dx * dx + dy * dy <= p.interaction_radius ** 2) & (idx != i)]
        assert got == set(near.tolist())


def test_share_moves_cost_and_benefit():
    w = _two_agents()
    w.genome[0, G.SHARE_W] = [50, 0, 0, 0]  # agent 0 always shares
    out = w._interact(np.arange(2))
    assert out["shares"] == 1
    assert w.energy[0] == pytest.approx(10.0 - w.p.share_cost)
    assert w.energy[1] == pytest.approx(10.0 + w.p.share_benefit)


def test_share_needs_enough_energy():
    w = _two_agents()
    w.genome[0, G.SHARE_W] = [50, 0, 0, 0]
    w.energy[0] = w.p.share_cost * 0.5
    assert w._interact(np.arange(2))["shares"] == 0


@pytest.mark.parametrize("blind,expected", [(False, 1), (True, 0)])
def test_tag_blind_share_ignores_similarity(blind, expected):
    w = _two_agents(tag_blind_share=blind)
    w.genome[:2, G.TAG] = [0.5, 0.5, 0.5]                # identical tags: similarity 1
    w.genome[0, G.SHARE_W] = [-25, 50, 0, 0]            # share only with similar
    assert w._interact(np.arange(2))["shares"] == expected


def test_fight_kills_exactly_one_and_feeds_winner():
    w = _two_agents()
    w.genome[0, G.ATTACK_W] = [50, 0, 0, 0]
    out = w._interact(np.arange(2))
    assert out["attacks"] == 1 and out["killed"] == 1
    assert w.alive[:2].sum() == 1
    winner = int(np.flatnonzero(w.alive[:2])[0])
    base = 10.0 - (w.p.attack_cost if winner == 0 else 0.0)
    loser_e = out["killed_energy"]
    assert w.energy[winner] == pytest.approx(base + w.p.predation_efficiency * loser_e)


def test_dead_agents_do_not_act():
    w = _two_agents()
    w.genome[:2, G.ATTACK_W] = [50, 0, 0, 0]  # both want to attack; only one fight can happen
    out = w._interact(np.arange(2))
    assert out["attacks"] == 1 and w.alive[:2].sum() == 1


def test_weights_stay_clipped():
    w = World(SMALL.replace(init_agents=1, birth_threshold=1.0, init_energy=1e6, weight_clip=2.0,
                            mutation_sd=5.0))
    w.genome[0, G.SHARE_W] = [2, 2, 2, 2]
    for _ in range(5):
        w._reproduce()
    kids = w.alive & (w.parent_id >= 0)  # founders are not clipped, only offspring
    assert kids.sum() > 0
    assert np.abs(w.genome[kids][:, G.WEIGHT_GENES]).max() <= 2.0


def test_seasons_modulate_regrowth_and_keep_ledger():
    p = SMALL.replace(season_period=100, season_amplitude=0.9, food_init_frac=0.3)
    w = World(p)
    factors = []
    for _ in range(100):
        f0 = w.total_food()
        factors.append(w.season_factor())
        w.step()
        L = w.last
        assert w.total_food() - f0 == pytest.approx(L["food_growth"] - L["eaten"], abs=1e-6)
    assert min(factors) == pytest.approx(0.1, abs=0.01) and max(factors) == pytest.approx(1.9, abs=0.01)
    assert World(SMALL).season_factor() == 1.0


def test_attacks_can_be_disabled():
    w = _two_agents(attacks_enabled=False)
    w.genome[0, G.ATTACK_W] = [50, 0, 0, 0]
    out = w._interact(np.arange(2))
    assert out["attacks"] == 0 and w.alive[:2].all()


def test_share_to_random_can_reach_far_agents():
    w = World(SMALL.replace(init_agents=3, share_to_random=True))
    w.x[:3] = [10.0, 11.0, 50.0]
    w.y[:3] = [10.0, 10.0, 50.0]
    w.energy[:3] = 10.0
    w.genome[:3, G.ATTACK_W] = [-50, 0, 0, 0]
    w.genome[:3, G.SHARE_W] = [-50, 0, 0, 0]
    w.genome[0, G.SHARE_W] = [50, 0, 0, 0]
    got_far = False
    for _ in range(40):
        w.energy[:3] = 10.0
        w._interact(np.arange(3))
        got_far |= w.energy[2] > 10.0
    assert got_far  # agent 2 is never a neighbour of agent 0


@pytest.mark.parametrize("to_random", [False, True])
def test_gardening_costs_energy_and_raises_fertility(to_random):
    w = World(SMALL.replace(garden_enabled=True, founder_garden=1.0, garden_to_random=to_random))
    for _ in range(50):
        e0 = w.total_energy()
        w.step()
        L = w.last
        de = (L["eaten"] - L["metabolic"] - L["movement"] - L["birth_cost"] - L["death_removed"]
              - L["share_given"] + L["share_received"] - L["attack_cost"] - L["killed_energy"]
              + L["predation_gain"] - L["alarm_cost"] - L["predated_energy"] - L["garden_cost"])
        assert w.total_energy() - e0 == pytest.approx(de, abs=1e-6)
    assert L["garden_cost"] > 0 and w.fertility.sum() > 0


def test_freeze_harvest_keeps_the_gene_fixed():
    w = World(SMALL.replace(freeze_harvest=True, mutation_sd=0.5, founder_harvest=0.3))
    for _ in range(200):
        w.step()
    assert np.allclose(w.genome[w.alive, G.HARVEST], 0.3)


def test_sacrifice_gives_everything_and_dies():
    w = _two_agents(share_all=True)
    w.genome[0, G.SHARE_W] = [50, 0, 0, 0]
    w.energy[0] = 7.0
    out = w._interact(np.arange(2))
    assert out["sacrifices"] == 1 and not w.alive[0] and w.energy[0] == 0.0
    assert w.energy[1] == pytest.approx(10.0 + 7.0 * w.p.share_benefit / w.p.share_cost)
    assert out["share_given"] == pytest.approx(7.0)


def test_ledger_balances_with_sacrifice():
    w = World(SMALL.replace(share_all=True))
    w.genome[w.alive, G.SHARE_W] = [-2, 0, -3, 0]
    for _ in range(200):
        e0 = w.total_energy()
        w.step()
        L = w.last
        de = (L["eaten"] - L["metabolic"] - L["movement"] - L["birth_cost"] - L["death_removed"]
              - L["share_given"] + L["share_received"] - L["attack_cost"] - L["killed_energy"]
              + L["predation_gain"] - L["alarm_cost"] - L["predated_energy"] - L["garden_cost"])
        assert w.total_energy() - e0 == pytest.approx(de, abs=1e-6)
    assert w.total_sacrificed > 0


def test_parse_overrides():
    import pytest
    from kin_dots.core.params import parse_overrides

    assert parse_overrides(["season_amplitude=0.7", "max_age=200", "attacks_enabled=False"]) == {
        "season_amplitude": 0.7, "max_age": 200, "attacks_enabled": False}
    assert parse_overrides(["move_speed=1"]) == {"move_speed": 1}  # int is fine for a float
    for bad in (["no_such_param=1"], ["max_age=abc"], ["attacks_enabled=1"], ["max_age=2.5"], ["move_speed"]):
        with pytest.raises(ValueError):
            parse_overrides(bad)
