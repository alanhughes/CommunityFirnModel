'''
Tests of the implicit gravity flow step of darcyscheme (flow_step_darcy "implicit_gravity",
darcy_implicit.py).

The unit tests check the root solve and the sweep on their own. The scheme tests call darcyscheme on
synthetic columns. The numbers of the thin-node tests and the cold-firn test are the answers of the
kinematic wave (tests/oracles).
'''

import numpy as np
import pytest
from scipy.optimize import brentq

import melt
from cfm_testing import SyntheticColumn
from constants import LF_I, RHO_I, RHO_I_MGM
from darcy_funcs import (hydrconducsat_Calonne, krel_vG, vG_Yama_params)
from darcy_implicit import (implicit_gravity_sweep, implicit_lateral_runoff, nodes_with_lateral_runoff,
                            water_after_outflow, _krel_vG_scalar)
from test_darcy_front_speed import (DAY, DENSITY, IRR_VAL, RADIUS, THIN, RESOLVED,
                                    TOLERANCE, _front_depth, _plateau, oracle)

IMPLICIT = {'flow_step_darcy': 'implicit_gravity'}


### The root solve ###

def residual(x, water, irreducible, saturated, rate, law, parameter):
    '''The equation of water_after_outflow.'''
    saturation = (x - irreducible) / (saturated - irreducible)
    if law == 'power_law':
        return x + rate * saturation ** parameter - water
    return x + rate * _krel_vG_scalar(saturation, parameter) - water


LAWS = [('vanGenuchten', 0.4), ('vanGenuchten', 0.8), ('power_law', 2.), ('power_law', 3.)]


@pytest.mark.unit
@pytest.mark.parametrize('law, parameter', LAWS)
@pytest.mark.parametrize('water, rate', [(0.002, 0.0005), (0.004, 0.003), (0.0035, 0.0001), (0.0031, 0.5)])
def test_root_satisfies_its_equation(law, parameter, water, rate):
    irreducible, saturated = 0.001, 0.004
    x, solved = water_after_outflow(water, irreducible, saturated, rate, law, parameter)
    assert solved == 1
    assert irreducible <= x <= water
    assert residual(x, water, irreducible, saturated, rate, law, parameter) == pytest.approx(0, abs=1e-12)


@pytest.mark.unit
def test_closed_form_agrees_with_brentq_at_exponent_two():
    irreducible, saturated, water, rate = 0.001, 0.004, 0.0033, 0.002
    closed, _ = water_after_outflow(water, irreducible, saturated, rate, 'power_law', 2.)
    root = brentq(residual, irreducible, water, args=(water, irreducible, saturated, rate, 'power_law', 2.),
                  xtol=1e-16, rtol=1e-10)
    assert closed == pytest.approx(root, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('law, parameter', LAWS)
def test_no_outflow_at_or_below_the_irreducible_content(law, parameter):
    for water in (0.0005, 0.001):
        assert water_after_outflow(water, 0.001, 0.004, 0.01, law, parameter) == (water, 0)


@pytest.mark.unit
@pytest.mark.parametrize('law, parameter', LAWS)
def test_saturated_rate_when_the_node_stays_full(law, parameter):
    x, solved = water_after_outflow(0.0102, 0.001, 0.004, 0.005, law, parameter)
    assert x == pytest.approx(0.0102 - 0.005)
    assert solved == 0


@pytest.mark.unit
@pytest.mark.parametrize('law, parameter', LAWS)
def test_result_is_never_below_the_irreducible_content(law, parameter):
    rng = np.random.default_rng(1)
    for _ in range(200):
        irreducible, saturated = 0.001, 0.004
        water = rng.uniform(0.0, 0.012)
        rate = 10 ** rng.uniform(-6, 1)
        x, _ = water_after_outflow(water, irreducible, saturated, rate, law, parameter)
        assert x >= min(water, irreducible) - 1e-15
        assert x <= water + 1e-15


### The sweep ###

def random_column(rng, n=30):
    saturated = rng.uniform(0.003, 0.006, n)
    irreducible = 0.2 * saturated
    capacity = saturated.copy()
    lwc = irreducible + rng.uniform(0, 1, n) * (saturated - irreducible) * (rng.uniform(size=n) < 0.5)
    rate = 10 ** rng.uniform(-5, -2, n)
    blocked = rng.uniform(size=n) < 0.1
    blocked[-1] = True
    capacity[blocked] = lwc[blocked]
    return lwc, capacity, irreducible, saturated, rate, blocked


@pytest.mark.unit
@pytest.mark.parametrize('law, parameter', [('vanGenuchten', 0.5), ('power_law', 2.), ('power_law', 3.)])
def test_sweep_conserves_mass_and_respects_capacity(law, parameter):
    rng = np.random.default_rng(2)
    for _ in range(40):
        lwc, capacity, irreducible, saturated, rate, blocked = random_column(rng)
        surface_input = rng.uniform(0, 0.02)
        held = np.zeros(len(lwc))
        new, excess, _ = implicit_gravity_sweep(lwc, capacity, irreducible, saturated, rate, law, parameter,
                                                surface_input, blocked, held)
        assert new.sum() + excess == pytest.approx(lwc.sum() + surface_input, rel=1e-12)
        assert np.all(new <= np.maximum(capacity, lwc) + 1e-15)
        assert np.all(new >= -1e-15)
        assert excess >= 0.


@pytest.mark.unit
def test_water_backs_up_above_a_blocked_node_to_the_surface_excess():
    n = 10
    saturated = np.full(n, 0.004)
    irreducible = np.full(n, 0.0008)
    blocked = np.zeros(n, dtype=bool)
    blocked[-1] = True
    lwc = irreducible.copy()   # every node holds its irreducible water
    rate = np.full(n, 1e9)   # a large rate: the nodes pass all water above the irreducible content
    # the input is below the capacity of the column: it passes down to the blocked bottom node
    new, excess, _ = implicit_gravity_sweep(lwc, saturated.copy(), irreducible, saturated, rate, 'power_law', 2.,
                                            0.001, blocked, np.zeros(n))
    assert excess == 0.
    assert new[-1] == pytest.approx(0.0008 + 0.001, abs=1e-6)   # a blocked node takes the water and does not pass it on
    assert np.allclose(new[:-1], 0.0008, atol=1e-6)
    # more than the bottom node can take: the water backs up node by node
    new, excess, _ = implicit_gravity_sweep(lwc, saturated.copy(), irreducible, saturated, rate, 'power_law', 2.,
                                            0.0075, blocked, np.zeros(n))
    assert np.all(new <= saturated + 1e-15)
    assert new[-1] == pytest.approx(0.004)
    assert new[-2] == pytest.approx(0.004)
    assert 0.0008 < new[-3] < 0.004
    assert np.allclose(new[:-3], 0.0008, atol=1e-6)
    assert new.sum() + excess == pytest.approx(lwc.sum() + 0.0075)
    assert excess == 0.
    # a full column: the rest is the surface excess
    new, excess, _ = implicit_gravity_sweep(lwc, saturated.copy(), irreducible, saturated, rate, 'power_law', 2.,
                                            0.1, blocked, np.zeros(n))
    assert np.allclose(new, saturated)
    assert excess == pytest.approx(lwc.sum() + 0.1 - saturated.sum())


@pytest.mark.unit
def test_a_blocked_node_has_no_outflow():
    saturated = np.full(5, 0.004)
    irreducible = np.full(5, 0.0008)
    blocked = np.array([False, False, True, False, True])
    lwc = np.full(5, 0.002)
    new, excess, _ = implicit_gravity_sweep(lwc, saturated, irreducible, saturated, np.full(5, 0.01), 'power_law', 2.,
                                            0.0, blocked, np.zeros(5))
    # nothing leaves a blocked node. The nodes above it pass their water to it, and it keeps all of it
    assert new[2] > lwc[2]
    assert new[4] >= lwc[4]
    assert new[3] < lwc[3]   # a node below a blocked node gets no water from it, and it drains
    assert new.sum() + excess == pytest.approx(lwc.sum())


@pytest.mark.unit
def test_held_water_does_not_flow():
    '''A cold node with an inflow below its hold passes nothing.'''
    n = 4
    saturated = np.full(n, 0.004)
    irreducible = np.full(n, 0.0008)
    blocked = np.zeros(n, dtype=bool)
    blocked[-1] = True
    held = np.full(n, 0.002)
    new, excess, _ = implicit_gravity_sweep(np.zeros(n), saturated, irreducible, saturated, np.full(n, 1.0),
                                            'power_law', 2., 0.0015, blocked, held)
    assert new[0] == pytest.approx(0.0015)
    assert np.all(new[1:] == 0.)
    assert excess == 0.


@pytest.mark.unit
def test_lateral_runoff_helpers():
    saturation = np.array([0.1, 0.96, 0.97, 0.2, 0.3, 0.0])
    nodes = nodes_with_lateral_runoff([4], saturation, 6)
    assert list(nodes) == [3]
    nodes = nodes_with_lateral_runoff([3], np.array([0.1, 0.96, 0.97, 0.2, 0.3, 0.0]), 6)
    assert list(nodes) == [0, 1, 2]
    assert list(nodes_with_lateral_runoff([0], saturation, 6)) == []
    runoff = implicit_lateral_runoff(np.array([0.003, 0.003]), np.full(2, 0.001), np.full(2, 0.004), np.full(2, 0.001),
                                     'power_law', 2., [1])
    assert runoff[0] == 0. and 0. < runoff[1] < 0.002


### The scheme: the front ###

def run_implicit_front(flux, dz, duration, dtsub, calls=24, **config):
    '''
    Rain on dry firn at the melting temperature, as in test_darcy_front_speed. dz is a number or a
    function of the number of nodes. Returns the plateau and the speed of the front, from the
    second half of the run, and the column.
    '''
    dt = duration / calls
    front = oracle(flux)
    config = {'ColeouLesaffre': False, 'IrrVal': IRR_VAL, 'dtsub_implicit_darcy': dtsub, **IMPLICIT, **config}
    if np.ndim(dz) == 0:
        n = int(1.5 * front.speed * duration / dz) + 30
        dz = np.full(n, dz)
    n = len(dz)
    column = SyntheticColumn(n=n, rho_profile=DENSITY, T_profile=273.15, dz=dz, r2=RADIUS ** 2,
                             rain=flux * dt / 0.917, dt=dt, steps=calls, config=config)
    depth = np.cumsum(column.dz) - column.dz / 2
    times, depths = [], []
    for call in range(calls):
        melt.darcyscheme(column, 0)
        if call >= calls // 2:
            theta = column.LWC / column.dz
            times.append((call + 1) * dt)
            depths.append(_front_depth(theta, depth, _plateau(theta, front.water_behind)))
    theta = column.LWC / column.dz
    return _plateau(theta, front.water_behind), np.polyfit(times, depths, 1)[0], column, depths[-1]


@pytest.mark.scheme
@pytest.mark.parametrize('flux, dz, days', THIN + RESOLVED[:2])
def test_front_matches_oracle_at_thin_nodes(flux, dz, days):
    '''The cases that are strict xfail for the explicit step in test_darcy_front_speed.'''
    plateau, speed, column, _ = run_implicit_front(flux, dz, days * DAY, 1800.)
    front = oracle(flux)
    assert plateau == pytest.approx(front.water_behind, rel=TOLERANCE)
    assert speed == pytest.approx(front.speed, rel=TOLERANCE)
    assert column.melt_diagnostics['runoff'] == 0.0
    assert column.LWC.sum() == pytest.approx(flux * days * DAY, rel=1e-10)


@pytest.mark.scheme
def test_front_matches_oracle_on_a_mixed_column():
    '''Nodes of 4 mm in the top 0.3 m and of 20 mm below.'''
    flux, days = 5e-7, 0.5
    front = oracle(flux)
    n_thin, n_thick = 75, 100
    dz = np.r_[np.full(n_thin, 0.004), np.full(n_thick, 0.02)]
    plateau, speed, column, _ = run_implicit_front(flux, dz, days * DAY, 1800.)
    assert speed == pytest.approx(front.speed, rel=TOLERANCE)
    theta = column.LWC / column.dz
    assert np.mean(theta[10:50]) == pytest.approx(front.water_behind, rel=TOLERANCE)
    assert column.LWC.sum() == pytest.approx(flux * days * DAY, rel=1e-10)


@pytest.mark.scheme
@pytest.mark.parametrize('dz, days', [(0.004, 0.5), (0.02, 1.0)])
def test_substep_does_not_matter_up_to_3600_s(dz, days):
    flux = 5e-7
    results = {dtsub: run_implicit_front(flux, dz, days * DAY, dtsub) for dtsub in (60., 1800., 3600.)}
    plateau_ref, _, _, front_ref = results[60.]
    for dtsub in (1800., 3600.):
        plateau, _, _, front_depth = results[dtsub]
        assert abs(front_depth - front_ref) <= dz
        assert plateau == pytest.approx(plateau_ref, rel=0.01)


### The scheme: cold firn ###

def cold_firn_answer(dz, flux, cold):
    '''The kinematic wave with refreezing, for 400 kg/m3 firn at -cold K: front speed and plateau.'''
    from melt import _available_pore_space, _irreducible_lwc
    t0 = 273.15 - cold
    heat_capacity = 152.5 + 7.122 * t0
    frozen = heat_capacity * 400. * cold / LF_I   # kg/m3 of water that freezes in the front
    density = 400. + frozen
    r, thickness, grain = np.array([density]), np.array([dz]), np.array([1e-3])
    pore_space = _available_pore_space(r, thickness, rho_i=RHO_I, cap_density=RHO_I)
    irreducible = (_irreducible_lwc(r, pore_space, 830, rho_i=RHO_I, coleou_lesaffre=True, irr_val=0.) / thickness)[0]
    saturated = (RHO_I - density) / RHO_I * RHO_I_MGM - 1e-6
    m = vG_Yama_params(grain, r)[2]
    k_sat = hydrconducsat_Calonne(grain, r, 0.001792)[0]
    conductivity = lambda theta: k_sat * krel_vG(m, np.array([(theta - irreducible) / (saturated - irreducible)]))[0]
    plateau = brentq(lambda theta: conductivity(theta) - flux, irreducible + 1e-12, saturated - 1e-9)
    return flux / (plateau + frozen / 1000.), plateau


@pytest.mark.scheme
@pytest.mark.parametrize('dtsub', [1800., 86400.])
def test_cold_firn_front_matches_the_kinematic_wave_with_refreezing(dtsub):
    '''
    Without melt the sub-step can be a whole day. Then the cold content of each node, held back
    in the sweep, sets the front. Without the hold the front is 11 % too deep.
    '''
    dz, depth, flux, cold, days = 0.004, 1.2, 2e-7, 13.0, 2
    speed, plateau = cold_firn_answer(dz, flux, cold)
    n = int(round(depth / dz))
    rho = np.full(n, 400.0)
    rho[-1] = 917.0
    config = {'RhoImp_darcy': 830, 'ThickImp_darcy': 0.1, 'slope_proxy_darcy': 0.1, 'dtmin_darcy': 60,
              'dtsub_implicit_darcy': dtsub, **IMPLICIT}
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=273.15 - cold, dz=dz, r2=1e-6, rain=flux * DAY / 0.917,
                             dt=DAY, steps=days, config=config)
    for day in range(days):
        melt.darcyscheme(column, day)
    theta = column.LWC / column.dz
    centre = column.z + column.dz / 2
    j = np.where(theta >= 0.5 * plateau)[0][-1]
    front = centre[j] + (theta[j] - 0.5 * plateau) / (theta[j] - theta[j + 1]) * (centre[j + 1] - centre[j])
    assert front == pytest.approx(speed * DAY * days, rel=0.03)


### The scheme: the budget and the limits ###

def edge_column(case, dtsub):
    n = 80
    dz = np.r_[np.full(40, 0.005), np.full(40, 0.02)]
    rho = np.full(n, 450.0)
    rho[20:26] = 900.0
    rho[50:53] = 917.0
    rho[-3:] = 917.0
    T = np.full(n, 273.15 - 5.0)
    T[:5] = 273.15
    lwc, melt_, rain = 0.0, 0.03, 0.002
    config = {'RhoImp_darcy': 830, 'ThickImp_darcy': 0.03, 'slope_proxy_darcy': 0.05, 'melt_strict': True,
              'dtsub_implicit_darcy': dtsub, **IMPLICIT}
    if case == 'surface_lens':
        rho[0:8] = 917.0
    if case == 'lens_at_1':
        rho[1:9] = 917.0
        melt_, rain = 0.0, 0.01
    if case == 'zero_input':
        melt_, rain, T[:] = 0.0, 0.0, 273.15
        lwc = 0.3 * dz * (RHO_I - rho) / RHO_I
    if case == 'dry_no_input':
        melt_, rain = 0.0, 0.0
    if case == 'saturated':
        T[:], rain = 273.15, 0.01
        lwc = dz * np.maximum((RHO_I - np.minimum(rho, 910.)) / RHO_I * RHO_I_MGM - 1e-6, 0)
    if case == 'zuo':
        config['runoff_method_darcy'] = 'ZuoOerlemans'
    if case == 'pl2':
        config.update({'krel_darcy': 'power_law', 'Ksat_darcy': 'power_law'})
    if case == 'pl3':
        config.update({'krel_darcy': 'power_law', 'krel_exponent_darcy': 3.})
    if case == 'no_lat':
        config['lat_runoff_darcy'] = False
    return SyntheticColumn(n=n, rho_profile=rho, T_profile=T, dz=dz, LWC=lwc, r2=(0.7e-3) ** 2, melt=melt_, rain=rain,
                           dt=DAY, steps=3, config=config)


EDGE_CASES = ['base', 'surface_lens', 'lens_at_1', 'zero_input', 'dry_no_input', 'saturated', 'zuo', 'pl2', 'pl3', 'no_lat']


@pytest.mark.scheme
@pytest.mark.parametrize('case', EDGE_CASES)
def test_edge_cases_keep_the_budget(case):
    '''With melt_strict a failed check raises. The water stays non-negative and below the capacity.'''
    column = edge_column(case, 1800.)
    for day in range(3):
        melt.darcyscheme(column, day)
        diagnostics = column.melt_diagnostics
        assert diagnostics['checks']['budget']['ok']
        assert abs(diagnostics['checks']['budget']['residual']) < 1e-12
        capacity = column.dz * np.maximum((RHO_I - np.minimum(column.rho, 910.)) / RHO_I * RHO_I_MGM - 1e-6, 1e-9)
        assert column.LWC.min() >= 0.
        assert np.all(column.LWC <= capacity + 1e-12)
        assert diagnostics['checks']['cold']['n_wet_cold_layers'] == 0


@pytest.mark.scheme
def test_lens_at_the_surface_sends_the_input_to_runoff():
    column = edge_column('surface_lens', 1800.)
    melt.darcyscheme(column, 0)
    assert column.melt_diagnostics['runoff'] > 0.01


### The scheme: lenses ###

LENS_DEPTH = 0.2


def lens_column(dz, lens, thick_imp, flux=2e-7, depth=1.2, days=3, **config):
    n = int(round(depth / dz))
    top = np.arange(n) * dz
    rho = np.full(n, 400.0)
    rho[-1] = 917.0
    rho[(top >= LENS_DEPTH - 1e-9) & (top < LENS_DEPTH + lens - 1e-9)] = 917.0
    config = {'RhoImp_darcy': 830, 'ThickImp_darcy': thick_imp, 'slope_proxy_darcy': 0.1,
              'dtsub_implicit_darcy': 1800., **IMPLICIT, **config}
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=273.15, dz=dz, r2=1e-6, rain=flux * DAY / 0.917, dt=DAY,
                             steps=days, config=config)
    below = top >= LENS_DEPTH + lens - 1e-9
    return column, below


@pytest.mark.scheme
def test_impermeable_lens_stops_the_water_and_runs_the_input_off():
    flux, days = 2e-7, 3
    column, below = lens_column(0.01, 0.2, thick_imp=0.1, days=days)   # a lens of 0.2 m is above ThickImp
    input_per_day = flux * DAY
    runoff = []
    for day in range(days):
        melt.darcyscheme(column, day)
        runoff.append(column.melt_diagnostics['runoff'])
    assert column.LWC[below].sum() == 0.
    assert runoff[-1] == pytest.approx(input_per_day, rel=0.05)


@pytest.mark.scheme
def test_a_lens_does_not_depend_on_its_nodes():
    '''A lens of 20 mm in one node and in five nodes, below ThickImp: it passes the same water.'''
    passed = []
    for dz in (0.02, 0.004):
        column, below = lens_column(dz, 0.02, thick_imp=0.1, days=2)
        for day in range(2):
            melt.darcyscheme(column, day)
        passed.append(column.LWC[below].sum())
    assert passed[0] == pytest.approx(passed[1], rel=0.02)


### Melt with a long sub-step ###

@pytest.mark.scheme
def test_melt_with_a_lens_at_3600_s_is_close_to_60_s():
    '''A 30 mm lens at 0.1 m, at the thickness limit, with melt. Daily runoff and refreezing within 4 %.'''
    results = {}
    for dtsub in (60., 3600.):
        n = 80
        dz = np.r_[np.full(40, 0.005), np.full(40, 0.02)]
        rho = np.full(n, 450.0)
        rho[20:26] = 900.0
        rho[50:53] = 917.0
        rho[-3:] = 917.0
        T = np.full(n, 273.15 - 5.0)
        T[:5] = 273.15
        config = {'RhoImp_darcy': 830, 'ThickImp_darcy': 0.03, 'slope_proxy_darcy': 0.05, 'lat_runoff_darcy': True,
                  'dtsub_implicit_darcy': dtsub, **IMPLICIT}
        column = SyntheticColumn(n=n, rho_profile=rho, T_profile=T, dz=dz, r2=(0.7e-3) ** 2, melt=0.03, rain=0.002,
                                 dt=DAY, steps=3, config=config)
        runoff, refreeze = [], []
        for day in range(3):
            melt.darcyscheme(column, day)
            runoff.append(column.melt_diagnostics['runoff'])
            refreeze.append(column.melt_diagnostics['refrozen'])
        results[dtsub] = (np.array(runoff), np.array(refreeze))
    scale = results[60.][0].max()
    assert np.all(np.abs(results[3600.][0] - results[60.][0]) <= 0.04 * scale)
    assert np.all(np.abs(results[3600.][1] - results[60.][1]) <= 0.04 * scale)


### The keys ###

@pytest.mark.scheme
def test_explicit_step_is_the_default():
    '''No key, and the key at "explicit": the same arrays, bit for bit. The implicit step gives other arrays.'''
    results = {}
    for name, extra in (('none', None), ('explicit', 'explicit'), ('implicit', 'implicit_gravity')):
        column = edge_column('base', 1800.)
        del column.c['flow_step_darcy']
        del column.c['dtsub_implicit_darcy']
        if extra:
            column.c['flow_step_darcy'] = extra
        for day in range(2):
            melt.darcyscheme(column, day)
        results[name] = column
    assert np.array_equal(results['none'].LWC, results['explicit'].LWC)
    assert np.array_equal(results['none'].Tz, results['explicit'].Tz)
    assert np.array_equal(results['none'].rho, results['explicit'].rho)
    assert not np.array_equal(results['none'].LWC, results['implicit'].LWC)


@pytest.mark.scheme
def test_a_wrong_value_of_the_key_stops_the_run():
    column = edge_column('base', 1800.)
    column.c['flow_step_darcy'] = 'implicit'
    with pytest.raises(ValueError, match='flow_step_darcy'):
        melt.darcyscheme(column, 0)


@pytest.mark.scheme
@pytest.mark.parametrize('dtsub', [0., -60.])
def test_a_substep_of_zero_or_below_stops_the_run(dtsub):
    column = edge_column('base', dtsub)
    with pytest.raises(ValueError, match='dtsub_implicit_darcy'):
        melt.darcyscheme(column, 0)


@pytest.mark.scheme
def test_implicit_step_uses_its_own_substep_key():
    '''dtsub_darcy, dtmin_darcy and dtmax_darcy do not change the implicit step. dtsub_implicit_darcy sets the sub-steps.'''
    counts = {}
    for name, extra in (('base', {}), ('explicit_keys', {'dtsub_darcy': 7., 'dtmin_darcy': 5, 'dtmax_darcy': 50}),
                        ('fixed', {'dtsub_implicit_darcy': 3600.})):
        column = edge_column('base', 1800.)
        column.c.update(extra)
        melt.darcyscheme(column, 0)
        counts[name] = (column.melt_diagnostics['substeps'], column.melt_diagnostics['root_solves'], column.LWC.copy())
    assert counts['base'][0] == 48
    assert counts['explicit_keys'][0] == 48
    assert np.array_equal(counts['base'][2], counts['explicit_keys'][2])
    assert counts['fixed'][0] == 24
    assert counts['base'][1] > 0


@pytest.mark.scheme
def test_substep_counters():
    column = edge_column('base', 1800.)
    melt.darcyscheme(column, 0)
    assert column.melt_diagnostics['substeps'] == 48
    assert column.melt_diagnostics['root_solves'] > 0
    explicit = edge_column('base', 1800.)
    explicit.c['flow_step_darcy'] = 'explicit'
    melt.darcyscheme(explicit, 0)
    assert explicit.melt_diagnostics['substeps'] > 0
    assert 'root_solves' not in explicit.melt_diagnostics


@pytest.mark.scheme
def test_long_substep_message_is_printed_once(capsys):
    column = edge_column('base', 86400.)
    melt.darcyscheme(column, 0)
    melt.darcyscheme(column, 1)
    text = capsys.readouterr().out
    assert text.count('dtsub_implicit_darcy is 86400 s.') == 1
    assert ('With a sub-step above 3600 s, the runoff and the refreezing can be wrong on days when melt reaches an '
            'ice lens. See the docstring of darcyscheme.') in text


@pytest.mark.scheme
def test_no_message_for_a_substep_up_to_3600_s(capsys):
    for dtsub in (1800., 3600.):
        melt.darcyscheme(edge_column('base', dtsub), 0)
    assert 'dtsub_implicit_darcy is' not in capsys.readouterr().out


@pytest.mark.scheme
def test_message_uses_the_substep_in_effect(capsys):
    '''A key of 7200 s on a model step of 3600 s gives a sub-step of 3600 s: no message.'''
    column = edge_column('base', 7200.)
    column.dt[:] = 3600.
    melt.darcyscheme(column, 0)
    assert 'dtsub_implicit_darcy is' not in capsys.readouterr().out
