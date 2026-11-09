'''
The config keys of darcyscheme: each key reaches the scheme, and the defaults in the
scheme and in the example configs are the same. ColeouLesaffre and IrrVal set the
irreducible water, in darcyscheme and in bucket.
'''

import json
from pathlib import Path

import numpy as np
import pytest

import melt
from config_validation import KNOWN_KEYS

DAY = 86400.0
DEFAULTS = {
    'RhoImp_darcy': 873.0,
    'ThickImp_darcy': 0.5,
    'lat_runoff_darcy': True,
    'slope_proxy_darcy': 0.02,
    'rholim_dcy': 910.0,
    'dtmin_darcy': 60,
    'dtmax_darcy': 3600,
    'dtsub_darcy': 60.0,
    'theta_s_factor_darcy': 0.917,
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


@pytest.mark.unit
def test_scheme_defaults_are_the_values_for_a_config_without_the_keys():
    # The example configs set the keys, so a run from them does not use these defaults.
    assert {key: melt._DARCY_DEFAULTS[key] for key in DEFAULTS} == DEFAULTS


@pytest.mark.scheme
@pytest.mark.parametrize('lens_thickness, bucket_keys', [
    (0.25, {'RhoImp': 890.0, 'ThickImp': 0.0}),  # a thin lens that leaks: these values change it
    (1.0, {'Slope': 0.5}),  # a thick lens that seals: this value changes the lateral runoff
])
def test_bucket_keys_do_not_reach_darcy(synthetic_column, lens_thickness, bucket_keys):
    reference, _ = _layered_column(synthetic_column, lens_thickness=lens_thickness)
    changed, _ = _layered_column(synthetic_column, lens_thickness=lens_thickness, **bucket_keys)
    assert _run(reference) == _run(changed)
    np.testing.assert_array_equal(reference.LWC, changed.LWC)


@pytest.mark.scheme
def test_rho_imp_sets_which_lens_is_impermeable(synthetic_column):
    # The lens of 880 kg m-3 is above the default threshold, so it holds no water.
    sealed, in_lens = _layered_column(synthetic_column)
    _run(sealed)
    assert np.sum(sealed.LWC[in_lens]) == 0.0

    # A higher threshold removes the lens from the impermeable set.
    leaky, in_lens = _layered_column(synthetic_column, RhoImp_darcy=890.0)
    _run(leaky)
    assert np.sum(leaky.LWC[in_lens]) > 0.0


@pytest.mark.scheme
@pytest.mark.parametrize('thick_imp, lens_seals', [(0.5, True), (0.51, False)])
def test_thick_imp_seals_a_lens_at_least_this_thick(synthetic_column, thick_imp, lens_seals):
    # The lens is 0.5 m thick: 4 nodes of 0.125 m, an exact sum.
    column, in_lens = _layered_column(synthetic_column, lens_thickness=0.5, ThickImp_darcy=thick_imp)
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
    assert runoff(lat_runoff_darcy=False) == 0.0
    assert runoff(slope_proxy_darcy=0.0) == 0.0
    assert runoff(slope_proxy_darcy=0.2) > default_runoff


@pytest.mark.scheme
def test_rho_cap_sets_the_density_of_the_flow_calculation(synthetic_column):
    # A thin lens of 905 kg m-3 leaks. A cap below the lens density makes the lens
    # more permeable, and the water in a capped node runs off at the end of the call.
    def result(**config):
        column, in_lens = _layered_column(synthetic_column, lens_density=905.0, lens_thickness=0.25,
                                          lat_runoff_darcy=False, **config)
        _, runoff = _run(column)
        return np.sum(column.LWC[in_lens]), runoff

    lens_water, runoff = result()
    assert lens_water > 0.0
    assert runoff == 0.0

    lens_water, runoff = result(rholim_dcy=880.0)
    assert lens_water == 0.0
    assert runoff > 0.0


@pytest.mark.scheme
def test_rho_imp_above_rho_cap_stops_the_run(synthetic_column, capsys):
    column, _ = _layered_column(synthetic_column, RhoImp_darcy=905.0, rholim_dcy=900.0)
    with pytest.raises(SystemExit):
        melt.darcyscheme(column, 0)
    message = capsys.readouterr().out
    assert 'RhoImp must be below rholim_dcy' in message


@pytest.mark.scheme
def test_sub_step_bounds_apply_after_the_first_sub_step(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, dtmin_darcy=1800, dtmax_darcy=1800)
    _run(column)
    assert lengths[0] == 60.0
    assert all(length == 1800 for length in lengths[1:-1])
    assert sum(lengths) == DAY


@pytest.mark.scheme
def test_dtmax_limits_the_sub_step(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, dtmax_darcy=120)
    _run(column)
    assert max(lengths) == 120


@pytest.mark.scheme
def test_dtmin_limits_the_sub_step(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, dtmin_darcy=600)
    _run(column)
    assert min(lengths[1:-1]) >= 600


@pytest.mark.scheme
def test_dtmax_takes_priority_if_the_bounds_conflict(synthetic_column, monkeypatch):
    lengths = _sub_step_lengths(monkeypatch)
    column, _ = _layered_column(synthetic_column, dtmin_darcy=1800, dtmax_darcy=600)
    _run(column)
    assert all(length == 600 for length in lengths[1:-1])


@pytest.mark.scheme
def test_theta_s_factor_limits_the_water_in_a_node(synthetic_column):
    # More rain than the firn above the lens can hold, so these nodes fill.
    # Rain removes no firn, and no water leaves the nodes by lateral runoff.
    def fullest_node(factor):
        column, _ = _layered_column(synthetic_column, melt_ie=0.0, rain=0.5, lat_runoff_darcy=False,
                                    theta_s_factor_darcy=factor)
        _run(column)
        above_lens = column.rho == 400.0
        return np.max(column.LWC[above_lens] / column.dz[above_lens]), (917.0 - 400.0) / 917.0

    theta, porosity = fullest_node(0.5)
    assert theta == pytest.approx(0.5 * porosity, rel=0.01)
    theta, porosity = fullest_node(0.917)
    assert theta == pytest.approx(0.917 * porosity, rel=0.01)


@pytest.mark.scheme
def test_theta_s_factor_does_not_change_the_refreeze_limit(synthetic_column):
    # One sub-step on very cold firn: the volume, not the cold content, limits refreeze.
    # So water refreezes up to 0.917 of the porosity, which gives the density of ice.
    # A factor of 1.2 lets a node hold more water, but the density does not go above ice.
    column = synthetic_column(n=40, rho_profile=800.0, T_profile=200.0, dz=0.125, r2=(5e-4) ** 2,
                              rain=0.1, dt=60.0, config={'theta_s_factor_darcy': 1.2})
    melt.darcyscheme(column, 0)
    assert np.max(column.rho) == pytest.approx(917.0, abs=0.01)


@pytest.mark.scheme
def test_irrval_holds_water_in_place(synthetic_column):
    # IrrVal is a fraction of the pore space, about 0.52 of the volume at 400 kg m-3.
    # Water fills the top nodes to a volumetric content of 0.05. There is no melt.
    def water_profile(irr_val):
        lwc = lambda z: np.where(z < 0.5, 0.05 * 0.125, 0.0)
        column = synthetic_column(n=40, rho_profile=400.0, T_profile=273.15, dz=0.125, r2=(5e-4) ** 2,
                                  LWC=lwc, melt=0.0, dt=DAY,
                                  config={'ColeouLesaffre': False, 'IrrVal': irr_val})
        start = column.LWC.copy()
        melt.darcyscheme(column, 0)
        return start, column.LWC

    start, end = water_profile(0.2)  # all the water is below the irreducible content
    np.testing.assert_array_equal(end, start)
    start, end = water_profile(0.02)
    assert np.sum(end[start == 0.0]) > 0.0
    start, end = water_profile(0.0)  # no irreducible water: the setting without residual water
    assert np.sum(end[start == 0.0]) > 0.0
