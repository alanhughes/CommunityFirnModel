'''
The option conduction_interval_darcy of the darcy scheme: heat conduction once in each implicit
sub-step ("substep") or once for the whole model step ("step", the default). The scheme tests call
darcyscheme on a synthetic column. The run tests count the heat calls of a short run.
'''

import numpy as np
import pytest

import firn_density_nospin
import melt
from cfm_testing import SyntheticColumn
from config_validation import ConfigValidationError, validate_config
from constants import CP_I, LF_I, RHO_W_KGM, T_MELT

DAY = 86400.
SUBSTEP = 1800.
IMPLICIT = {'flow_step_darcy': 'implicit_gravity', 'dtsub_implicit_darcy': SUBSTEP, 'RhoImp_darcy': 830,
            'ThickImp_darcy': 0.03, 'slope_proxy_darcy': 0.05, 'LWCheat': 'enthalpy'}
SUBSTEP_KEY = {'conduction_interval_darcy': 'substep'}


def conduction_column(config=None, melt_amount=0.004, surface=253.15, firn=258.15, LWC=0.):
    '''A column of 80 nodes (5 mm over 20 mm) with the attributes that the heat functions read.'''
    n = 80
    dz = np.r_[np.full(40, 0.005), np.full(40, 0.02)]
    rho = np.full(n, 450.0)
    rho[-3:] = 917.0
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=np.full(n, firn), dz=dz, LWC=LWC, r2=1.0e-8,
                             melt=melt_amount, dt=DAY, steps=1, config={**IMPLICIT, **(config or {})})
    column.Ts = np.array([surface])            # surface temperature of the step [K]
    column.melt_solver = 'transient_solve_enthalpy'
    column.total_count = 0
    column.T10m = None
    column.heat_time_in_scheme = 0.
    return column


def column_heat(column):
    '''Heat of the column [J m-2], from ice at the melting point: sensible (specific heat of ice) plus latent.'''
    T = column.Tz
    sensible = column.mass * (152.5 * (T - T_MELT) + 7.122 / 2 * (T ** 2 - T_MELT ** 2))
    return np.sum(sensible) + np.sum(column.LWC * RHO_W_KGM * LF_I)


def wet_top_column(config=None, surface=253.15, firn=258.15, water=1e-4, nodes=10):
    '''A column with no melt and water at the melting point in the top nodes.'''
    LWC = np.zeros(80)
    LWC[:nodes] = water
    column = conduction_column(config, melt_amount=0., surface=surface, firn=firn, LWC=LWC)
    column.Tz[:nodes] = T_MELT
    return column


def count_heat_calls(monkeypatch):
    '''Wrap heatDiff and refreezeDiff in melt and in firn_density_nospin. Return the list of (name, dt).'''
    calls = []
    for module in (melt, firn_density_nospin):
        for name in ('heatDiff', 'refreezeDiff'):
            def wrapper(model, iii, *args, original=getattr(module, name), name=name, **kwargs):
                calls.append((name, kwargs.get('dt')))
                return original(model, iii, *args, **kwargs)
            monkeypatch.setattr(module, name, wrapper)
    return calls


### 1. The heat calls ###

@pytest.mark.scheme
def test_one_heat_call_in_each_substep(monkeypatch):
    calls = count_heat_calls(monkeypatch)
    column = conduction_column(SUBSTEP_KEY)
    melt.darcyscheme(column, 0)
    assert len(calls) == 48
    assert all(dt == SUBSTEP for _, dt in calls)
    assert {name for name, _ in calls} == {'heatDiff', 'refreezeDiff'}   # water in the first sub-steps, none at the end
    assert column.heat_time_in_scheme == pytest.approx(DAY)


@pytest.mark.scheme
def test_the_step_setting_makes_no_heat_call_in_the_scheme(monkeypatch):
    calls = count_heat_calls(monkeypatch)
    column = conduction_column()
    melt.darcyscheme(column, 0)
    assert calls == []
    assert column.heat_time_in_scheme == 0.


@pytest.mark.conservation
def test_run_conducts_once_in_each_substep_and_not_after_the_scheme(short_config, monkeypatch):
    calls_per_step = {}
    state = {'iii': None}

    def mark_scheme(original):
        def wrapper(model, iii):
            state['iii'] = iii
            return original(model, iii)
        return wrapper
    monkeypatch.setattr(firn_density_nospin, 'darcyscheme', mark_scheme(firn_density_nospin.darcyscheme))

    for module in (melt, firn_density_nospin):
        for name in ('heatDiff', 'refreezeDiff'):
            def wrapper(model, iii, *args, original=getattr(module, name), **kwargs):
                calls_per_step.setdefault(iii, []).append((state['iii'] == iii, kwargs.get('dt')))
                return original(model, iii, *args, **kwargs)
            monkeypatch.setattr(module, name, wrapper)

    run = short_config(years=2, liquid='darcy', bucket_until_year=1981, DFfile='CFM_example_66.5_-46.25.pkl',
                       DFresample='1d', num_reps=1, TWriteInt=1, outputs=['density', 'depth', 'temperature', 'LWC'],
                       flow_step_darcy='implicit_gravity', dtsub_implicit_darcy=SUBSTEP,
                       conduction_interval_darcy='substep')
    model = run.run()
    with_scheme = {iii: calls for iii, calls in calls_per_step.items() if calls[0][0]}
    assert with_scheme                                        # the run has days with water
    for iii, calls in with_scheme.items():
        assert len(calls) == int(np.ceil(model.dt[iii] / SUBSTEP))   # the step is a little longer than a day: the last sub-step is short
        assert all(in_scheme for in_scheme, _ in calls)             # no call after the scheme
        assert all(dt == SUBSTEP for _, dt in calls[:-1])
        assert sum(dt for _, dt in calls) == pytest.approx(model.dt[iii], abs=1e-6)   # the calls cover the step
    for iii, calls in calls_per_step.items():
        if iii not in with_scheme:
            assert calls == [(False, None)]                   # a day with no water: the step as before


### 2. The water budget ###

@pytest.mark.scheme
def test_water_budget_and_night_freeze():
    column = conduction_column(SUBSTEP_KEY)
    melt.darcyscheme(column, 0)
    diagnostics = column.melt_diagnostics
    assert diagnostics['checks']['budget']['ok'] and abs(diagnostics['checks']['budget']['residual']) < 1e-12
    assert diagnostics['refrozen_conduction'] > 0.
    assert diagnostics['refrozen'] == pytest.approx(diagnostics['refrozen_flow'] + diagnostics['refrozen_conduction'],
                                                    rel=1e-12)
    assert column.refrozen.sum() == pytest.approx(diagnostics['refrozen'], rel=1e-9)   # the layers have the same total


@pytest.mark.scheme
def test_the_step_setting_has_no_conduction_freezing():
    column = conduction_column()
    melt.darcyscheme(column, 0)
    diagnostics = column.melt_diagnostics
    assert diagnostics['refrozen_conduction'] == 0.
    assert diagnostics['surface_heat'] == 0.
    assert diagnostics['refrozen_flow'] == diagnostics['refrozen']


### 3. The top node and the ghost node ###

@pytest.mark.scheme
def test_wet_top_node_does_not_stay_cold_and_wet(capsys):
    # The top node holds more water than its cold content can freeze, so it ends wet and at 0 C.
    column = wet_top_column(SUBSTEP_KEY, water=3e-4, nodes=1)
    column.LWC[1:5] = 0.
    column.Ts = np.array([263.15])
    melt.darcyscheme(column, 0)
    assert 'water content in a cold layer' not in capsys.readouterr().out
    assert column.melt_diagnostics['checks']['cold']['ok']
    assert column.LWC[0] == 0. or column.Tz[0] >= T_MELT


@pytest.mark.unit
@pytest.mark.parametrize('solver', ['transient_solve_enthalpy', 'transient_solve_ncz'])
def test_wet_top_node_freezes_only_with_the_heat_that_leaves(solver):
    # Node 0 is wet at 0 C, the column below is at 0 C, and the surface is 1 K colder. Node 0 holds more water
    # than the heat through the surface can freeze in one sub-step. With a fixed node 0, the solver would
    # freeze all the water for free. With the ghost node, the latent heat of the water that freezes is the
    # heat that leaves through the surface, and node 0 stays wet.
    column = wet_top_column(SUBSTEP_KEY, surface=T_MELT - 1., firn=T_MELT, water=1.5e-3, nodes=1)
    column.melt_solver = solver
    water_before = column.LWC[0] * RHO_W_KGM
    froze, surface_heat = melt._conduction_substep(column, 0, SUBSTEP)
    latent = froze[0] * RHO_W_KGM * LF_I
    assert 0. < froze[0] * RHO_W_KGM < water_before
    assert column.LWC[0] > 0.
    assert latent == pytest.approx(-surface_heat, rel=1e-4)


@pytest.mark.unit
def test_the_ghost_node_leaves_the_column_arrays_as_they_were():
    column = wet_top_column(SUBSTEP_KEY)
    shapes = {name: getattr(column, name).shape for name in ('z', 'dz', 'rho', 'mass', 'LWC', 'Tz')}
    z, dz = column.z.copy(), column.dz.copy()
    melt._conduction_substep(column, 0, SUBSTEP)
    assert {name: getattr(column, name).shape for name in shapes} == shapes
    assert np.array_equal(column.z, z) and np.array_equal(column.dz, dz)
    assert np.all(np.isfinite(column.Tz)) and np.all(column.Tz <= T_MELT)


### 4. The energy ###

def solver_heat(column):
    '''Heat of the column [J m-2] in the metric of the refreezing solvers: constant specific heat of ice, plus latent.'''
    return np.sum(column.mass * CP_I * (column.Tz - T_MELT)) + np.sum(column.LWC * RHO_W_KGM * LF_I)


@pytest.mark.unit
@pytest.mark.parametrize('solver', ['transient_solve_enthalpy', 'transient_solve_ncz'])
@pytest.mark.parametrize('surface, firn, water, nodes', [(253.15, 258.15, 1.5e-3, 1), (253.15, 258.15, 1e-4, 10),
                                                         (272.15, 271.15, 1.5e-3, 1)])
def test_one_conduction_call_closes_the_energy_with_a_wet_top_node(solver, surface, firn, water, nodes):
    # Node 0 is wet and the surface is cold: the case in which a fixed node 0 freezes its water for free.
    # The heat of the column changes by the heat through the face between the ghost node and node 0, in the
    # metric of the solver. The measured error is below 3e-7 of the heat.
    column = wet_top_column(SUBSTEP_KEY, surface=surface, firn=firn, water=water, nodes=nodes)
    column.melt_solver = solver
    heat_before = solver_heat(column)
    froze, surface_heat = melt._conduction_substep(column, 0, SUBSTEP)
    assert froze[0] > 0.                                    # node 0 freezes water, with the latent heat counted
    error = solver_heat(column) - heat_before - surface_heat
    print(f'{solver}, surface {surface} K: heat through the surface {surface_heat:.0f}, error {error:.3g} J/m2')
    assert abs(error) < 1e-6 * abs(surface_heat)


@pytest.mark.unit
def test_one_conduction_call_closes_the_energy_with_no_water():
    # heatDiff only. Its heat capacity is the specific heat of ice at the temperature before the solve.
    column = conduction_column(SUBSTEP_KEY, melt_amount=0., surface=253.15, firn=258.15)
    T_before, mass = column.Tz.copy(), column.mass.copy()
    froze, surface_heat = melt._conduction_substep(column, 0, SUBSTEP)
    change = np.sum(mass * (152.5 + 7.122 * T_before) * (column.Tz - T_before))
    print(f'heat through the surface {surface_heat:.0f}, error {change - surface_heat:.3g} J/m2')
    assert np.all(froze == 0.)
    assert change == pytest.approx(surface_heat, rel=1e-9)


@pytest.mark.scheme
@pytest.mark.parametrize('surface, firn', [(253.15, 258.15), (258.15, 263.15), (268.15, 271.15)])
def test_column_heat_change_is_the_surface_heat(surface, firn):
    # No melt and no runoff, a wet node 0 at a cold surface, 48 sub-steps: the heat of the column (specific
    # heat of ice at the node temperature, plus latent) changes by the heat through the face between the ghost
    # node and node 0. The surface heat comes from the solved state, not from the energy change. The only
    # difference is the freezing block of the scheme, which uses the specific heat of ice at the node
    # temperature, and the solver, which uses a constant 2097 J kg-1 K-1. The measured error is 0.04 % to
    # 0.18 % of the surface heat (0.58 % if the water is one wet node of 0.3 kg m-2). The tolerance is 0.5 %.
    column = wet_top_column(SUBSTEP_KEY, surface=surface, firn=firn)
    heat_before = column_heat(column)
    melt.darcyscheme(column, 0)
    diagnostics = column.melt_diagnostics
    assert diagnostics['runoff'] == 0.
    assert diagnostics['refrozen_conduction'] > 0.
    error = column_heat(column) - heat_before - diagnostics['surface_heat']
    print(f'surface {surface} K: surface heat {diagnostics["surface_heat"]:.0f}, error {error:.0f} J/m2 = {error / abs(diagnostics["surface_heat"]):.4f}')
    assert abs(error) < 0.005 * abs(diagnostics['surface_heat'])


### 4b. The first step with water ###

@pytest.mark.conservation
def test_first_step_with_melt_runs_with_substep(short_config):
    # With "substep" the sub-steps cover the step, and time_evolve() skips its conduction block. The
    # diagnostics of the first step must still run.
    run = short_config(years=1, start_year=1981, num_reps=1, liquid='darcy', DFfile='CFM_example_66.5_-46.25.pkl',
                       DFresample='1d', TWriteInt=1, outputs=['density', 'depth', 'temperature', 'LWC'],
                       flow_step_darcy='implicit_gravity', dtsub_implicit_darcy=SUBSTEP,
                       conduction_interval_darcy='substep')
    model = run.build()
    model.snowmeltSec[0] = model.snowmeltSec.max()
    model.snowmelt[0] = model.snowmelt.max()
    assert model.snowmeltSec[0] > 0.
    model.time_evolve()
    assert np.all(np.isfinite(model.Tz))


@pytest.mark.unit
def test_time_evolve_stops_if_the_scheme_covered_more_than_the_step(short_config, monkeypatch):
    run = short_config(years=1, start_year=1981, num_reps=1, liquid='darcy', DFfile='CFM_example_66.5_-46.25.pkl',
                       DFresample='1d', TWriteInt=1, outputs=['density', 'depth', 'temperature', 'LWC'],
                       flow_step_darcy='implicit_gravity', dtsub_implicit_darcy=SUBSTEP,
                       conduction_interval_darcy='substep')
    model = run.build()
    model.snowmeltSec[0] = model.snowmeltSec.max()
    model.snowmelt[0] = model.snowmelt.max()

    def too_long(original):
        def wrapper(self, iii):
            result = original(self, iii)
            self.heat_time_in_scheme = 2. * self.dt[iii]
            return result
        return wrapper
    monkeypatch.setattr(firn_density_nospin, 'darcyscheme', too_long(firn_density_nospin.darcyscheme))
    with pytest.raises(ValueError, match='more than the step'):
        model.time_evolve()


### 5. The default ###

@pytest.mark.conservation
def test_step_setting_is_the_default(short_config):
    outputs = {}
    for name, extra in (('absent', {}), ('step', {'conduction_interval_darcy': 'step'})):
        run = short_config(years=2, liquid='darcy', bucket_until_year=1981, DFfile='CFM_example_66.5_-46.25.pkl',
                           DFresample='1d', num_reps=1, TWriteInt=1, outputs=['density', 'depth', 'temperature', 'LWC'],
                           flow_step_darcy='implicit_gravity', dtsub_implicit_darcy=SUBSTEP, **extra)
        model = run.run()
        outputs[name] = (model.rho, model.Tz, model.LWC, model.z, model.mass)
    for a, b in zip(outputs['absent'], outputs['step']):
        assert np.array_equal(a, b)


### 6. The validator ###

@pytest.mark.unit
@pytest.mark.parametrize('config, message', [
    ({'conduction_interval_darcy': 'substep', 'heatDiff': True}, "needs 'flow_step_darcy'"),
    ({'conduction_interval_darcy': 'substep', 'heatDiff': True, 'flow_step_darcy': 'explicit'}, "needs 'flow_step_darcy'"),
    ({'conduction_interval_darcy': 'substep', 'heatDiff': True, 'flow_step_darcy': 'implicit_gravity', 'manualT': True},
     "'manualT'"),
    ({'conduction_interval_darcy': 'substep', 'heatDiff': False, 'flow_step_darcy': 'implicit_gravity'},
     "needs 'heatDiff'"),
    ({'conduction_interval_darcy': 'substep', 'heatDiff': True, 'flow_step_darcy': 'implicit_gravity', 'SEB': True},
     "'SEB'"),
    ({'conduction_interval_darcy': 'substep', 'heatDiff': True, 'flow_step_darcy': 'implicit_gravity',
      'meltwater_solver': 'legacy'}, "'legacy'"),
    ({'conduction_interval_darcy': 'hourly'}, "invalid value 'hourly' for 'conduction_interval_darcy'"),
])
def test_validator_stops_the_run(config, message):
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(dict(config))


@pytest.mark.unit
def test_validator_accepts_substep_with_the_implicit_step():
    # The error list holds only the missing required keys, so no message names the key.
    config = {'conduction_interval_darcy': 'substep', 'flow_step_darcy': 'implicit_gravity', 'heatDiff': True}
    with pytest.raises(ConfigValidationError) as error:
        validate_config(config)
    assert 'conduction_interval_darcy' not in str(error.value)


@pytest.mark.scheme
def test_the_scheme_stops_for_a_wrong_setting():
    with pytest.raises(ValueError, match='conduction_interval_darcy'):
        melt.darcyscheme(conduction_column({'conduction_interval_darcy': 'hourly'}), 0)
    with pytest.raises(ValueError, match='implicit_gravity'):
        melt.darcyscheme(conduction_column({**SUBSTEP_KEY, 'flow_step_darcy': 'explicit'}), 0)
