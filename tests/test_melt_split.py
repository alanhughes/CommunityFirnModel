'''
The option melt_split_darcy of the darcy scheme: the melt of a step falls in a burst at a fixed peak
rate (implicit gravity flow step only). The schedule tests check _melt_schedule on its own. The scheme
tests call darcyscheme on a synthetic column.
'''

import numpy as np
import pytest

import melt
from cfm_testing import SyntheticColumn
from config_validation import ConfigValidationError, validate_config
from constants import RHO_W_KGM

HOUR = 3600.
DAY = 86400.
PEAK = 2.8e-7   # 1 mm/h


def _melt_for(duration_h, rate=PEAK):
    '''Melt mass [kg] that the rate gives in duration_h hours.'''
    return duration_h * HOUR * rate * RHO_W_KGM


### The schedule ###

@pytest.mark.unit
@pytest.mark.parametrize('duration_h, dtsub, peak_hour', [
    (4.2, 1800., 12.), (1.0, 1800., 12.), (0.2, 1800., 12.), (7.9, 3600., 3.), (3.0, 600., 17.),
    (10.0, 1800., 0.), (10.0, 1800., 24.), (23.9, 3600., 12.)])
def test_schedule_adds_up_to_the_melt(duration_h, dtsub, peak_hour):
    melt_mass = _melt_for(duration_h)
    schedule, window = melt._melt_schedule(melt_mass, DAY, dtsub, PEAK, peak_hour)
    assert len(schedule) == int(np.ceil(DAY / dtsub))
    assert np.sum(schedule) == pytest.approx(melt_mass, rel=1e-12)
    assert np.all(schedule >= 0.)


@pytest.mark.unit
def test_schedule_rounds_the_burst_up_to_whole_substeps():
    # 4.2 h at 1800 s: 8.4 sub-steps, so 9. The rate is below the peak.
    melt_mass = _melt_for(4.2)
    schedule, window = melt._melt_schedule(melt_mass, DAY, 1800., PEAK, 12.)
    assert np.count_nonzero(schedule) == 9
    assert window[1] - window[0] == pytest.approx(9 * 1800.)
    rate = schedule[schedule > 0] / 1800. / RHO_W_KGM
    assert np.all(rate <= PEAK)
    assert rate[0] == pytest.approx(PEAK * 4.2 / 4.5, rel=1e-12)


@pytest.mark.unit
def test_schedule_example_of_the_docstring():
    # 2.1 mm at 1 mm/h with 30 min sub-steps: 5 sub-steps of 0.84 mm/h.
    melt_mass = 2.1e-3 * RHO_W_KGM
    schedule, _ = melt._melt_schedule(melt_mass, DAY, 1800., 1e-3 / HOUR, 12.)
    assert np.count_nonzero(schedule) == 5
    assert schedule[schedule > 0] / 1800. / RHO_W_KGM * HOUR == pytest.approx(np.full(5, 0.84e-3))


@pytest.mark.unit
@pytest.mark.parametrize('peak_hour', [12., 8.25, 17.])
def test_schedule_is_centred_on_the_hour(peak_hour):
    schedule, window = melt._melt_schedule(_melt_for(3.0), DAY, 1800., PEAK, peak_hour)
    centre = 0.5 * (window[0] + window[1])
    assert abs(centre - peak_hour * HOUR) <= 900.   # at most half a sub-step away
    assert window[0] % 1800. == 0.


@pytest.mark.unit
def test_schedule_is_moved_to_fit_at_the_start_of_the_step():
    melt_mass = _melt_for(4.0)   # 8 sub-steps
    schedule, window = melt._melt_schedule(melt_mass, DAY, 1800., PEAK, 1.0)
    assert window == (0., 8 * 1800.)
    assert np.count_nonzero(schedule[:8]) == 8 and np.count_nonzero(schedule[8:]) == 0
    assert np.sum(schedule) == pytest.approx(melt_mass, rel=1e-12)


@pytest.mark.unit
def test_schedule_is_moved_to_fit_at_the_end_of_the_step():
    melt_mass = _melt_for(4.0)
    schedule, window = melt._melt_schedule(melt_mass, DAY, 1800., PEAK, 23.5)
    assert window == (DAY - 8 * 1800., DAY)
    assert np.count_nonzero(schedule[-8:]) == 8 and np.count_nonzero(schedule[:-8]) == 0
    assert np.sum(schedule) == pytest.approx(melt_mass, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('duration_h', [24., 30., 100.])
def test_schedule_is_uniform_when_the_peak_rate_cannot_give_the_melt(duration_h):
    melt_mass = _melt_for(duration_h)
    schedule, window = melt._melt_schedule(melt_mass, DAY, 1800., PEAK, 3.)
    np.testing.assert_array_equal(schedule, (melt_mass / DAY) * np.full(48, 1800.))
    assert window == (0., DAY)


@pytest.mark.unit
def test_schedule_of_zero_melt():
    schedule, window = melt._melt_schedule(0., DAY, 1800., PEAK, 12.)
    np.testing.assert_array_equal(schedule, np.zeros(48))
    assert window is None


@pytest.mark.unit
def test_schedule_with_a_shorter_last_substep():
    # 5000 s of sub-steps of 3600 s in a step of 20000 s: 5 full sub-steps and one of 2000 s.
    timetot, dtsub = 20000., 3600.
    melt_mass = _melt_for(1.0)
    schedule, window = melt._melt_schedule(melt_mass, timetot, dtsub, PEAK, 5.5)
    assert len(schedule) == 6
    assert np.sum(schedule) == pytest.approx(melt_mass, rel=1e-12)
    # a burst that ends the step: the last, short sub-step takes melt in proportion to its length
    schedule, window = melt._melt_schedule(_melt_for(2.0), timetot, dtsub, PEAK, 5.5)   # 2 sub-steps, moved to the end
    assert window == (3600. * 4, timetot)
    assert schedule[5] / schedule[4] == pytest.approx(2000. / 3600.)
    assert np.sum(schedule) == pytest.approx(_melt_for(2.0), rel=1e-12)
    # the step is shorter than the sub-step
    schedule, window = melt._melt_schedule(_melt_for(0.1), 1200., 1200., PEAK, 12.)
    assert len(schedule) == 1 and window == (0., 1200.)


### The scheme ###

IMPLICIT = {'flow_step_darcy': 'implicit_gravity', 'dtsub_implicit_darcy': 1800., 'RhoImp_darcy': 830,
            'ThickImp_darcy': 0.03, 'slope_proxy_darcy': 0.05}


def burst_column(config=None, melt_amount=0.004, rain=0.0, T=273.15 - 5.0):
    n = 80
    dz = np.r_[np.full(40, 0.005), np.full(40, 0.02)]
    rho = np.full(n, 450.0)
    rho[-3:] = 917.0
    return SyntheticColumn(n=n, rho_profile=rho, T_profile=np.full(n, T), dz=dz, r2=1.0e-8, melt=melt_amount,
                           rain=rain, dt=DAY, steps=1, config={**IMPLICIT, **(config or {})})


BURST = {'melt_split_darcy': 'burst', 'melt_peak_rate_darcy': PEAK, 'melt_peak_hour_darcy': 17.}


@pytest.mark.scheme
def test_burst_water_budget_and_window():
    column = burst_column(BURST)
    melt.darcyscheme(column, 0)
    checks = column.melt_diagnostics['checks']
    assert checks['budget']['ok'] and abs(checks['budget']['residual']) < 1e-12
    assert column.melt_diagnostics['water_in'] > 0.
    start, end = column.melt_burst_window
    assert 0. <= start < 17 * HOUR < end <= DAY
    assert abs(0.5 * (start + end) - 17 * HOUR) <= 900.


@pytest.mark.scheme
def test_uniform_leaves_no_window():
    column = burst_column()
    melt.darcyscheme(column, 0)
    assert column.melt_burst_window is None


@pytest.mark.scheme
def test_snapshot_before_the_burst_shows_no_surface_lowering():
    # The burst is at 17 h. A snapshot at 0.25 of the day (6 h) is before it, so the surface has not moved.
    uniform = burst_column({'snapshot_fraction_darcy': 0.25})
    burst = burst_column({**BURST, 'snapshot_fraction_darcy': 0.25})
    dh_melt = {}
    for name, column in (('uniform', uniform), ('burst', burst)):
        column.darcy_snapshot = True
        z_start = column.z.copy()
        dh_melt[name] = melt.darcyscheme(column, 0)[-1]
        column.z_start = z_start
    np.testing.assert_array_equal(burst.z_snapshot, burst.z_start)   # no melt yet: the grid is as it was
    assert not np.array_equal(uniform.z_snapshot, uniform.z_start)   # uniform melt has lowered the surface
    # at the end of the step, the burst has lowered the surface about as much as the uniform melt
    # (not exactly: refreezing between the melt sub-steps changes the density of the nodes that melt)
    assert dh_melt['burst'] < 0.
    assert dh_melt['burst'] == pytest.approx(dh_melt['uniform'], rel=0.01)


@pytest.mark.scheme
def test_burst_substeps_are_not_idle():
    # Dry, cold firn, no rain. The sub-steps before the burst have no input and no water: they are idle.
    # Each sub-step of the burst has melt input, so none of them is idle.
    column = burst_column(BURST)
    melt.darcyscheme(column, 0)
    start, end = column.melt_burst_window
    n_before, n_burst = int(start / 1800.), int((end - start) / 1800.)
    diagnostics = column.melt_diagnostics
    assert diagnostics['substeps'] == 48
    assert n_before <= diagnostics['idle_substeps'] <= 48 - n_burst


@pytest.mark.scheme
def test_the_default_keys_give_the_arrays_of_no_key():
    absent = burst_column()
    explicit = burst_column({'melt_split_darcy': 'uniform', 'melt_peak_rate_darcy': 2.8e-7, 'melt_peak_hour_darcy': 12.})
    out_absent = melt.darcyscheme(absent, 0)
    out_explicit = melt.darcyscheme(explicit, 0)
    for a, b in zip(out_absent, out_explicit):
        np.testing.assert_array_equal(a, b)
    assert absent.melt_diagnostics['idle_substeps'] == explicit.melt_diagnostics['idle_substeps']


@pytest.mark.scheme
def test_a_burst_that_fills_the_step_is_the_uniform_run():
    # 0.03 m ie of melt needs far more than a day at the peak rate: the burst is the uniform rule.
    uniform = burst_column(melt_amount=0.03)
    burst = burst_column(BURST, melt_amount=0.03)
    out_uniform = melt.darcyscheme(uniform, 0)
    out_burst = melt.darcyscheme(burst, 0)
    for a, b in zip(out_uniform, out_burst):
        np.testing.assert_array_equal(a, b)
    assert burst.melt_burst_window == (0., DAY)


@pytest.mark.scheme
def test_the_scheme_stops_for_a_wrong_burst_setting():
    with pytest.raises(ValueError, match='implicit_gravity'):
        melt.darcyscheme(burst_column({**BURST, 'flow_step_darcy': 'explicit'}), 0)
    with pytest.raises(ValueError, match='melt_peak_rate_darcy'):
        melt.darcyscheme(burst_column({**BURST, 'melt_peak_rate_darcy': 0.}), 0)
    with pytest.raises(ValueError, match='melt_peak_hour_darcy'):
        melt.darcyscheme(burst_column({**BURST, 'melt_peak_hour_darcy': 25.}), 0)
    with pytest.raises(ValueError, match='melt_split_darcy'):
        melt.darcyscheme(burst_column({'melt_split_darcy': 'peak'}), 0)


### The validator ###

@pytest.mark.unit
@pytest.mark.parametrize('config, message', [
    ({'MELT': True, 'liquid': 'darcy', 'melt_split_darcy': 'burst'}, "needs 'flow_step_darcy'"),
    ({'MELT': True, 'liquid': 'darcy', 'melt_split_darcy': 'burst', 'flow_step_darcy': 'explicit'},
     "needs 'flow_step_darcy'"),
    ({'melt_split_darcy': 'peak'}, "invalid value 'peak' for 'melt_split_darcy'"),
    ({'melt_peak_rate_darcy': 0.}, 'melt_peak_rate_darcy'),
    ({'melt_peak_rate_darcy': -1e-7}, 'melt_peak_rate_darcy'),
    ({'melt_peak_hour_darcy': -0.5}, 'melt_peak_hour_darcy'),
    ({'melt_peak_hour_darcy': 24.5}, 'melt_peak_hour_darcy'),
])
def test_validator_stops_the_run(config, message):
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(dict(config))


@pytest.mark.unit
@pytest.mark.parametrize('config', [{'MELT': True, 'liquid': 'bucket'}, {'MELT': False, 'liquid': 'darcy'}])
def test_validator_ignores_the_burst_when_darcy_does_not_run(config):
    # The burst key has no effect, so the run does not stop for it.
    config = {**config, 'melt_split_darcy': 'burst', 'flow_step_darcy': 'explicit'}
    with pytest.raises(ConfigValidationError) as error:
        validate_config(config)
    assert 'burst' not in str(error.value)


@pytest.mark.unit
def test_validator_accepts_the_burst_with_the_implicit_step():
    # The error list holds only the missing required keys, so no message names the burst keys.
    config = {'melt_split_darcy': 'burst', 'flow_step_darcy': 'implicit_gravity',
              'melt_peak_rate_darcy': 2.8e-7, 'melt_peak_hour_darcy': 17}
    with pytest.raises(ConfigValidationError) as error:
        validate_config(config)
    assert 'melt_' not in str(error.value) and 'burst' not in str(error.value)
