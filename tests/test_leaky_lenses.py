'''
Unit tests for melt.leaky_lens_routing (partially permeable ice lenses).
'''

import numpy as np
import pytest

from melt import leaky_lens_routing
from config_validation import ConfigValidationError, validate_config


def column(dense_nodes, nnd=10, dz=0.1, stcap=0.01):
    '''A column of nnd nodes with the given dense node indices; dense nodes and
    the bottom node have no storage capacity (as in bucket).'''
    dense = np.zeros(nnd, dtype=bool)
    dense[list(dense_nodes)] = True
    dz = np.full(nnd, dz)
    cap = np.where(dense, 0.0, stcap)
    cap[-1] = 0.0
    return dense, dz, cap


def blocked_above(node, amount, nnd=10):
    LWCblocked = np.zeros(nnd)
    LWCblocked[node] = amount
    return LWCblocked


def test_water_is_conserved():
    rng = np.random.default_rng(1)
    nnd = 50
    dense = rng.random(nnd) < 0.3
    dz = rng.uniform(0.01, 0.2, nnd)
    stcap = np.where(dense, 0.0, rng.uniform(0.0, 0.02, nnd))
    LWCblocked = np.where(rng.random(nnd) < 0.2, rng.uniform(0.0, 0.05, nnd), 0.0)
    for decay_length in [0.0, 0.05, 0.3, 10.0]:
        storage, blocked = leaky_lens_routing(LWCblocked, stcap, dense, dz, decay_length)
        assert storage.sum() + blocked.sum() == pytest.approx(LWCblocked.sum(), rel=1e-12)
        assert np.all(storage <= stcap + 1e-15)
        assert np.all(storage >= 0) and np.all(blocked >= 0)


def test_zero_decay_length_blocks_everything():
    # as ThickImp = 0: water blocked above a lens stays where it is
    dense, dz, stcap = column([4, 5])
    LWCblocked = blocked_above(3, 0.02)
    storage, blocked = leaky_lens_routing(LWCblocked, stcap, dense, dz, 0.0)
    assert np.all(storage == 0)
    np.testing.assert_array_equal(blocked, LWCblocked)


def test_transmitted_fraction_fills_storage_below():
    # one dense node of 0.1 m with decay length 0.1 m passes exp(-1) of the water
    dense, dz, stcap = column([4], stcap=1.0)
    LWCblocked = blocked_above(3, 0.02)
    storage, blocked = leaky_lens_routing(LWCblocked, stcap, dense, dz, 0.1)
    assert storage[5] == pytest.approx(0.02 * np.exp(-1.0))
    assert blocked[3] == pytest.approx(0.02 * (1 - np.exp(-1.0)))
    assert storage[:5].sum() == 0


def test_large_decay_length_passes_all_lenses():
    dense, dz, stcap = column([2, 5, 6], stcap=0.004)
    LWCblocked = blocked_above(1, 0.02)
    storage, blocked = leaky_lens_routing(LWCblocked, stcap, dense, dz, 1e9)
    # everything below the first lens fills up; only what does not fit is left at the bottom
    np.testing.assert_allclose(storage[[3, 4, 7, 8]], 0.004)
    assert blocked[8] == pytest.approx(0.02 - 4 * 0.004)
    assert blocked[:8].sum() == pytest.approx(0.0, abs=1e-9)


def test_lens_thickness_adds_up():
    # two adjacent 0.05 m dense nodes transmit the same as one 0.1 m node
    nnd = 10
    dense_two = np.zeros(nnd, dtype=bool)
    dense_two[[4, 5]] = True
    dz_two = np.full(nnd, 0.1)
    dz_two[[4, 5]] = 0.05
    dense_one, dz_one, _ = column([4])
    stcap_two = np.where(dense_two, 0.0, 1.0)
    stcap_one = np.where(dense_one, 0.0, 1.0)
    stcap_two[-1] = stcap_one[-1] = 0.0
    LWCblocked = blocked_above(3, 0.02)
    s_two, b_two = leaky_lens_routing(LWCblocked, stcap_two, dense_two, dz_two, 0.07)
    s_one, b_one = leaky_lens_routing(LWCblocked, stcap_one, dense_one, dz_one, 0.07)
    assert s_two.sum() == pytest.approx(s_one.sum())
    assert b_two[3] == pytest.approx(b_one[3])


def test_water_inside_a_lens_only_passes_the_rest_of_the_lens():
    dense, dz, stcap = column([4, 5], stcap=1.0)
    LWCblocked = blocked_above(5, 0.02) # excess water in the lower lens node
    storage, blocked = leaky_lens_routing(LWCblocked, stcap, dense, dz, 0.1)
    assert storage[6] == pytest.approx(0.02 * np.exp(-1.0))


def test_nothing_blocked_returns_zeros():
    dense, dz, stcap = column([4])
    storage, blocked = leaky_lens_routing(np.zeros(10), stcap, dense, dz, 0.1)
    assert np.all(storage == 0) and np.all(blocked == 0)


@pytest.mark.parametrize('options, expected', [
    ({}, None),
    ({'LeakyLenses': True}, None),
    ({'LeakyLenses': True, 'liquid': 'darcy'}, 'only used by the bucket scheme'),
    ({'LeakyLenses': True, 'DownToIce': True}, "ignored when 'DownToIce'"),
    ({'LeakyLenses': False, 'liquid': 'darcy'}, None),
])
def test_validator_warns_when_leaky_lenses_has_no_effect(capsys, options, expected):
    from config_validation import validate_config
    from cfm_testing import example_config
    validate_config({**example_config(), **options})
    out = capsys.readouterr().out
    if expected is None:
        assert 'LeakyLenses' not in out
    else:
        assert expected in out


### Fixed pass fraction (LensPassFraction) ###

MODES = ['first', 'every']


def random_column(seed=3, nnd=50):
    rng = np.random.default_rng(seed)
    dense = rng.random(nnd) < 0.3
    dz = rng.uniform(0.01, 0.2, nnd)
    stcap = np.where(dense, 0.0, rng.uniform(0.0, 0.02, nnd))
    stcap[-1] = 0.0
    LWCblocked = np.where(rng.random(nnd) < 0.2, rng.uniform(0.0, 0.05, nnd), 0.0)
    return LWCblocked, stcap, dense, dz


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fraction, decay_length', [(0.0, 0.0), (1.0, 1e12)])
def test_pass_fraction_limits_match_decay_length(mode, fraction, decay_length):
    LWCblocked, stcap, dense, dz = random_column()
    ref = leaky_lens_routing(LWCblocked, stcap, dense, dz, decay_length)
    got = leaky_lens_routing(LWCblocked, stcap, dense, dz, 0.1, fraction, mode)
    np.testing.assert_allclose(got[0], ref[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got[1], ref[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fraction', [0.0, 0.2, 0.5, 1.0])
def test_pass_fraction_conserves_water(mode, fraction):
    LWCblocked, stcap, dense, dz = random_column()
    storage, blocked = leaky_lens_routing(LWCblocked, stcap, dense, dz, 0.1, fraction, mode)
    assert storage.sum() + blocked.sum() == pytest.approx(LWCblocked.sum(), rel=1e-12)
    assert np.all(storage <= stcap + 1e-15)
    assert np.all(storage >= 0) and np.all(blocked >= 0)


def test_every_run_passes_the_same_share_whatever_its_thickness():
    f = 0.3
    for run in ([3, 4, 5], [3]):
        dense, dz, cap = column(run, stcap=0.0)
        cap[6:9] = 0.01
        storage, blocked = leaky_lens_routing(blocked_above(2, 0.004), cap, dense, dz, 0.1, f, 'every')
        assert storage[6:].sum() == pytest.approx(f * 0.004, rel=1e-12)
        assert blocked[2] == pytest.approx((1 - f) * 0.004, rel=1e-12)


def test_every_two_runs_in_series_pass_f_squared():
    f = 0.5
    dense, dz, cap = column([3, 4, 7], stcap=0.0)
    cap[8] = 0.01   # only the node below the second run can store water
    storage, blocked = leaky_lens_routing(blocked_above(2, 0.004), cap, dense, dz, 0.1, f, 'every')
    assert storage[8:].sum() == pytest.approx(0.004 * f * f, rel=1e-12)
    assert blocked[6] == pytest.approx(0.004 * f * (1 - f), rel=1e-12)
    assert blocked[2] == pytest.approx(0.004 * (1 - f), rel=1e-12)


def test_first_share_f_reaches_below_the_second_run():
    f = 0.4
    dense, dz, cap = column([3, 4, 7], stcap=0.0)
    cap[8] = 0.01
    storage, blocked = leaky_lens_routing(blocked_above(2, 0.004), cap, dense, dz, 0.1, f, 'first')
    assert storage[8:].sum() == pytest.approx(0.004 * f, rel=1e-12)
    assert blocked[2] == pytest.approx(0.004 * (1 - f), rel=1e-12)
    assert blocked[6] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize('options, message', [
    ({'LeakyLenses': True, 'LensPassFraction': 1.5}, 'LensPassFraction'),
    ({'LeakyLenses': True, 'LensPassFraction': -0.1}, 'LensPassFraction'),
    ({'LeakyLenses': True, 'LensPassFraction': True}, 'LensPassFraction'),
    ({'LeakyLenses': True, 'LensPassFraction': 0.5, 'LensPassMode': 'all'}, 'LensPassMode'),
])
def test_validator_refuses_bad_pass_keys(options, message):
    from cfm_testing import example_config
    with pytest.raises(ConfigValidationError, match=message):
        validate_config({**example_config(), **options})


def test_validator_accepts_good_pass_keys(capsys):
    from cfm_testing import example_config
    validate_config({**example_config(), 'LeakyLenses': True, 'LensPassFraction': 0.5, 'LensPassMode': 'every'})
    assert 'LensPass' not in capsys.readouterr().out
