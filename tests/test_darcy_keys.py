'''
The config keys of darcyscheme: each key reaches the scheme, and a config without
the keys gives the result of the example configs.
'''

import json
from pathlib import Path

import numpy as np
import pytest

import melt
from config_validation import KNOWN_KEYS

DAY = 86400.0
DEFAULTS = {
    'darcy_RhoImp': 873.0,
    'darcy_ThickImp': 0.5,
    'darcy_lateral_runoff': True,
    'darcy_slope': 0.02,
    'darcy_rho_cap': 910.0,
    'darcy_dtmin': 60,
    'darcy_dtmax': 3600,
    'darcy_theta_s_factor': 0.917,
    'darcy_theta_irreducible': 0.02,
}


def _layered_column(synthetic_column, lens_density=880.0, lens_top=0.5, lens_thickness=1.0,
                    melt_ie=0.05, rain=0.0, **config):
    '''
    Temperate firn of 400 kg m-3 with one lens, and a thick lens at the base.
    darcyscheme always seals the lowest lens, so the base lens leaves the
    thresholds free to act on the upper lens. Returns the column and the lens mask.
    '''
    dz, n = 0.125, 40
    depth = np.arange(n) * dz
    in_lens = (depth >= lens_top) & (depth < lens_top + lens_thickness)
    rho = np.where(in_lens, lens_density, 400.0)
    rho[depth >= 4.0] = 900.0
    column = synthetic_column(n=n, rho_profile=rho, T_profile=273.15, dz=dz, r2=(5e-4) ** 2,
                              melt=melt_ie, rain=rain, dt=DAY, config=config)
    return column, in_lens


def _run(column):
    '''Run the scheme once. Return the refrozen water and the runoff [m w.e.].'''
    outputs = melt.darcyscheme(column, 0)
    return outputs[10], outputs[11]


def _sub_step_lengths(monkeypatch):
    '''Record dt of each runoffDarcy call. The scheme makes one call in each sub-step.'''
    lengths = []
    original = melt.runoffDarcy

    def wrapper(dt, slope, bigk, inds):
        lengths.append(dt)
        return original(dt, slope, bigk, inds)

    monkeypatch.setattr(melt, 'runoffDarcy', wrapper)
    return lengths


@pytest.mark.unit
@pytest.mark.parametrize('config_name', ['example.json', 'example_df.json', 'example_csv.json'])
def test_example_configs_set_the_defaults(config_name):
    config = json.loads((Path('.') / config_name).read_text())
    assert {key: config[key] for key in DEFAULTS} == DEFAULTS


@pytest.mark.unit
def test_validator_knows_the_keys():
    assert set(DEFAULTS) <= KNOWN_KEYS


@pytest.mark.scheme
def test_config_without_the_keys_gives_the_same_result(synthetic_column):
    with_keys, _ = _layered_column(synthetic_column)
    without_keys, _ = _layered_column(synthetic_column)
    for key in DEFAULTS:
        del without_keys.c[key]
    assert _run(with_keys) == _run(without_keys)
    for name in ('LWC', 'rho', 'dz', 'Tz'):
        np.testing.assert_array_equal(getattr(with_keys, name), getattr(without_keys, name))


@pytest.mark.scheme
def test_bucket_keys_do_not_reach_darcy(synthetic_column):
    reference, _ = _layered_column(synthetic_column)
    changed, _ = _layered_column(synthetic_column, RhoImp=500.0, ThickImp=0.0, Slope=0.5)
    assert _run(reference) == _run(changed)
    np.testing.assert_array_equal(reference.LWC, changed.LWC)


@pytest.mark.scheme
def test_rho_imp_sets_which_lens_is_impermeable(synthetic_column):
    # The lens of 880 kg m-3 is above the default threshold, so it holds no water.
    sealed, in_lens = _layered_column(synthetic_column)
    _run(sealed)
    assert np.sum(sealed.LWC[in_lens]) == 0.0

    # A higher threshold removes the lens from the impermeable set.
    leaky, in_lens = _layered_column(synthetic_column, darcy_RhoImp=890.0)
    _run(leaky)
    assert np.sum(leaky.LWC[in_lens]) > 0.0


@pytest.mark.scheme
@pytest.mark.parametrize('thick_imp, lens_seals', [(0.49, True), (0.5, False)])
def test_thick_imp_seals_a_lens_only_if_the_lens_is_thicker(synthetic_column, thick_imp, lens_seals):
    # The lens is 0.5 m thick: 4 nodes of 0.125 m, an exact sum.
    column, in_lens = _layered_column(synthetic_column, lens_thickness=0.5, darcy_ThickImp=thick_imp)
    assert np.sum(column.dz[in_lens]) == 0.5
    _run(column)
    assert (np.sum(column.LWC[in_lens]) == 0.0) == lens_seals


@pytest.mark.scheme
def test_lateral_runoff_switch_and_slope(synthetic_column):
    def runoff(**config):
        column, _ = _layered_column(synthetic_column, **config)
        return _run(column)[1]

    default_runoff = runoff()
    assert default_runoff > 0.0
    assert runoff(darcy_lateral_runoff=False) == 0.0
    assert runoff(darcy_slope=0.0) == 0.0
    assert runoff(darcy_slope=0.2) > default_runoff


@pytest.mark.scheme
def test_rho_cap_sets_the_density_of_the_flow_calculation(synthetic_column):
    # A thin lens of 905 kg m-3 leaks. A cap below the lens density makes the lens
    # more permeable, and the water in a capped node runs off at the end of the call.
    def result(**config):
        column, in_lens = _layered_column(synthetic_column, lens_density=905.0, lens_thickness=0.25,
                                          darcy_lateral_runoff=False, **config)
        _, runoff = _run(column)
        return np.sum(column.LWC[in_lens]), runoff

    lens_water, runoff = result()
    assert lens_water > 0.0
    assert runoff == 0.0

    lens_water, runoff = result(darcy_rho_cap=880.0)
    assert lens_water == 0.0
    assert runoff > 0.0


@pytest.mark.scheme
def test_rho_imp_above_rho_cap_stops_the_run(synthetic_column, capsys):
    column, _ = _layered_column(synthetic_column, darcy_RhoImp=905.0, darcy_rho_cap=900.0)
    with pytest.raises(SystemExit):
        melt.darcyscheme(column, 0)
    message = capsys.readouterr().out
    assert 'darcy_RhoImp' in message and 'darcy_rho_cap' in message


@pytest.mark.scheme
def test_sub_step_bounds_apply_after_the_first_sub_step(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, darcy_dtmin=1800, darcy_dtmax=1800)
    _run(column)
    assert lengths[0] == 60.0
    assert all(length == 1800 for length in lengths[1:-1])
    assert sum(lengths) == DAY


@pytest.mark.scheme
def test_dtmax_limits_the_sub_step(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, darcy_dtmax=120)
    _run(column)
    assert max(lengths) == 120


@pytest.mark.scheme
def test_dtmin_limits_the_sub_step(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, darcy_dtmin=600)
    _run(column)
    assert min(lengths[1:-1]) >= 600


@pytest.mark.scheme
def test_theta_s_factor_limits_the_water_in_a_node(synthetic_column):
    # More rain than the firn above the lens can hold, so these nodes fill.
    # Rain removes no firn, and no water leaves the nodes by lateral runoff.
    def fullest_node(factor):
        column, _ = _layered_column(synthetic_column, melt_ie=0.0, rain=0.5, darcy_lateral_runoff=False,
                                    darcy_theta_s_factor=factor)
        _run(column)
        above_lens = column.rho == 400.0
        return np.max(column.LWC[above_lens] / column.dz[above_lens]), (917.0 - 400.0) / 917.0

    theta, porosity = fullest_node(0.5)
    assert theta == pytest.approx(0.5 * porosity, rel=0.01)
    theta, porosity = fullest_node(0.917)
    assert theta == pytest.approx(0.917 * porosity, rel=0.01)


@pytest.mark.scheme
def test_theta_irreducible_holds_water_in_place(synthetic_column):
    # Water fills the top nodes to theta = 0.05. There is no melt.
    def water_profile(theta_irreducible):
        lwc = lambda z: np.where(z < 0.5, 0.05 * 0.125, 0.0)
        column = synthetic_column(n=40, rho_profile=400.0, T_profile=273.15, dz=0.125, r2=(5e-4) ** 2,
                                  LWC=lwc, melt=0.0, dt=DAY,
                                  config={'darcy_theta_irreducible': theta_irreducible})
        start = column.LWC.copy()
        melt.darcyscheme(column, 0)
        return start, column.LWC

    start, end = water_profile(0.1)  # all the water is below the irreducible content
    np.testing.assert_array_equal(end, start)
    start, end = water_profile(0.02)
    assert np.sum(end[start == 0.0]) > 0.0
