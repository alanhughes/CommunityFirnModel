'''
The thin partial node 0 of the darcy scheme with the implicit flow step.

The melt removal of each sub-step leaves one node partly melted (the partial node, the new node 0).
The key min_partial_node_dz_darcy merges a partial node that is thinner than the key into the node
below. The conduction in each sub-step cannot freeze more water than the pores of a node hold: the
excess turns back into water and goes down, or runs off. The tests call darcyscheme and
_conduction_substep on a synthetic column.
'''

import numpy as np
import pytest

import melt
from cfm_testing import SyntheticColumn
from constants import CP_I, LF_I, RHO_I, RHO_W_KGM, T_MELT

DAY = 86400.
FLOW = {'flow_step_darcy': 'implicit_gravity', 'dtsub_implicit_darcy': DAY, 'RhoImp_darcy': 830,
        'ThickImp_darcy': 0.03, 'slope_proxy_darcy': 0.05, 'LWCheat': 'enthalpy'}


def partial_node_column(config=None, node0_mass=2.0, melt_mass=1.99, LWC0=0., LWC1=0., T0=258.15, T1=263.15):
    '''
    A column of 40 nodes. Node 0 has the mass node0_mass [kg m-2] at 450 kg m-3. One sub-step of one day
    melts melt_mass [kg m-2]. Nodes 0 and 1 have different temperatures and some liquid water.
    '''
    n = 40
    dz = np.full(n, 0.05)
    dz[0] = node0_mass / 450.
    rho = np.full(n, 450.0)
    rho[-3:] = 917.0
    T = np.full(n, 263.15)
    T[0], T[1] = T0, T1
    LWC = np.zeros(n)
    LWC[0], LWC[1] = LWC0, LWC1
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=T, dz=dz, LWC=LWC, r2=1.0e-8,
                             melt=melt_mass / RHO_I, dt=DAY, steps=1, config={**FLOW, **(config or {})})
    column.r2[0], column.r2[1] = 2.0e-8, 6.0e-8
    column.age[0], column.age[1] = 100., 500.
    return column


def sensible_heat(column):
    '''Sensible heat of the ice [J m-2] above ice at the melting point (the integral of 152.5 + 7.122 T).'''
    T = column.Tz
    return np.sum(column.mass * (152.5 * (T - T_MELT) + 7.122 / 2 * (T ** 2 - T_MELT ** 2)))


def removal_only(monkeypatch):
    '''Switch off the flow step, so that the melt removal is the only change (the melt water goes nowhere).'''
    monkeypatch.setattr(melt, '_implicit_flow_step', lambda *args, **kwargs: (0., 0., 0))


### Part A: the merge ###

@pytest.mark.scheme
def test_a_thin_partial_node_merges_and_the_column_keeps_its_mass_water_and_heat(monkeypatch):
    removal_only(monkeypatch)
    state = {}
    for name, key in (('plain', {}), ('merged', {'min_partial_node_dz_darcy': 1e-3})):
        column = partial_node_column(key)
        before = {'heat': sensible_heat(column), 'water': column.LWC.sum(), 'mass': column.mass.sum(),
                  'dz': column.dz.sum(), 'bottom': (column.mass[-1], column.dz[-1], sensible_heat_of_node(column, -1))}
        melt.darcyscheme(column, 0)
        state[name] = (column, before)
    plain, before = state['plain']
    merged, _ = state['merged']

    assert plain.dz[0] < 1e-3                                 # without the merge node 0 is thin
    assert merged.dz[0] >= 1e-3                               # with it, node 0 is at least 1 mm
    assert merged.melt_diagnostics['partial_node_merges'] == 1
    assert plain.melt_diagnostics['partial_node_merges'] == 0

    # The merge moves nothing out of the column. It appends one more copy of the bottom node.
    extra_mass, extra_dz, extra_heat = before['bottom']
    assert merged.mass.sum() - plain.mass.sum() == pytest.approx(extra_mass, rel=1e-12)
    assert merged.dz.sum() - plain.dz.sum() == pytest.approx(extra_dz, rel=1e-12)
    assert merged.LWC.sum() == pytest.approx(plain.LWC.sum(), rel=1e-12, abs=1e-18)
    assert sensible_heat(merged) - sensible_heat(plain) == pytest.approx(extra_heat, rel=1e-12)
    assert merged.mass[0] == pytest.approx(plain.mass[0] + plain.mass[1], rel=1e-12)
    assert merged.rho[0] == pytest.approx(merged.mass[0] / merged.dz[0], rel=1e-12)
    assert len(merged.dz) == len(plain.dz) == 40
    assert np.array_equal(merged.mass[1:-2], plain.mass[2:-1]) or np.allclose(merged.mass[1:-2], plain.mass[2:-1], rtol=1e-14)


def sensible_heat_of_node(column, i):
    T = column.Tz[i]
    return column.mass[i] * (152.5 * (T - T_MELT) + 7.122 / 2 * (T ** 2 - T_MELT ** 2))


@pytest.mark.scheme
def test_merged_node_properties(monkeypatch):
    removal_only(monkeypatch)
    column = partial_node_column({'min_partial_node_dz_darcy': 1e-3})
    m0, m1 = 0.01, 450. * 0.05
    melt.darcyscheme(column, 0)
    w0, w1 = m0 / (m0 + m1), m1 / (m0 + m1)
    assert column.r2[0] == pytest.approx(w0 * 2.0e-8 + w1 * 6.0e-8, rel=1e-6)
    assert column.age[0] == pytest.approx(w0 * 100. + w1 * 500., rel=1e-6)
    assert 258.15 < column.Tz[0] < 263.15


@pytest.mark.scheme
def test_merged_node_keeps_the_water_of_the_two_nodes(monkeypatch):
    # Wet nodes at the melting point: the freezing block does not act. The partial node keeps its
    # share of the water of node 0 (the partial node is 0.01 kg of the 2 kg of node 0).
    removal_only(monkeypatch)
    totals = []
    for key in ({}, {'min_partial_node_dz_darcy': 1e-3}):
        column = partial_node_column(key, LWC0=1e-5, LWC1=2e-5, T0=T_MELT, T1=T_MELT)
        melt.darcyscheme(column, 0)
        totals.append(column)
    plain, merged = totals
    assert merged.LWC[0] == pytest.approx(1e-5 * 0.01 / 2.0 + 2e-5, rel=1e-12)
    assert merged.LWC.sum() == pytest.approx(plain.LWC.sum(), rel=1e-12)
    assert merged.Tz[0] == T_MELT


@pytest.mark.scheme
def test_a_partial_node_above_the_limit_does_not_merge(monkeypatch):
    removal_only(monkeypatch)
    column = partial_node_column({'min_partial_node_dz_darcy': 1e-3}, node0_mass=2.0, melt_mass=1.0)   # partial node: 1 kg at 450 kg m-3, 2.2 mm
    melt.darcyscheme(column, 0)
    assert column.melt_diagnostics['partial_node_merges'] == 0
    assert column.dz[0] == pytest.approx(1.0 / 450., rel=1e-12)


@pytest.mark.scheme
def test_no_merge_into_the_bottom_node(monkeypatch):
    # The melt ends in the second node from the bottom. The partial node is thin, but the node below it
    # is the bottom node, so it does not merge.
    removal_only(monkeypatch)
    n = 6
    rho = np.full(n, 450.0)
    rho[-1] = 917.0
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=263.15, dz=0.05, LWC=0., r2=1.0e-8,
                             melt=(4 * 22.5 + 22.5 - 0.01) / RHO_I, dt=DAY, steps=1,
                             config={**FLOW, 'min_partial_node_dz_darcy': 1e-3})
    melt.darcyscheme(column, 0)
    assert column.melt_diagnostics['partial_node_merges'] == 0
    assert len(column.dz) == n
    assert column.dz[0] == pytest.approx(0.01 / 450., rel=1e-9)


@pytest.mark.scheme
@pytest.mark.parametrize('key', [None, 0, 0.])
def test_key_at_zero_is_the_same_as_no_key(key, monkeypatch, capsys):
    results = []
    for config in ({}, {} if key is None else {'min_partial_node_dz_darcy': key}):
        column = partial_node_column(config, LWC0=1e-5, LWC1=2e-5)
        melt.darcyscheme(column, 0)
        results.append((column.rho, column.dz, column.Tz, column.LWC, column.mass, column.age, column.r2))
    for a, b in zip(*results):
        assert np.array_equal(a, b)


@pytest.mark.unit
@pytest.mark.parametrize('value', [-1e-3, 'thin', True])
def test_the_scheme_stops_for_a_wrong_key(value):
    with pytest.raises(ValueError, match='min_partial_node_dz_darcy'):
        melt.darcyscheme(partial_node_column({'min_partial_node_dz_darcy': value}), 0)


@pytest.mark.scheme
def test_warning_for_substep_with_the_key_at_zero(capsys):
    config = {'conduction_interval_darcy': 'substep', 'dtsub_implicit_darcy': 1800.}
    for key, expected in (({}, True), ({'min_partial_node_dz_darcy': 1e-3}, False)):
        column = partial_node_column({**config, **key}, melt_mass=0.)
        column.Ts = np.array([260.])
        column.melt_solver = 'transient_solve_enthalpy'
        column.total_count = 0
        column.T10m = None
        column.heat_time_in_scheme = 0.
        melt.darcyscheme(column, 0)
        out = capsys.readouterr().out
        assert ('min_partial_node_dz_darcy is 0' in out) == expected
        if expected:
            melt.darcyscheme(column, 0)
            assert 'min_partial_node_dz_darcy is 0' not in capsys.readouterr().out   # printed once in each run


### Part C: the pore-space limit of the conduction solve ###

SUBSTEP = 1800.
SUBSTEP_FLOW = {**FLOW, 'dtsub_implicit_darcy': SUBSTEP, 'conduction_interval_darcy': 'substep'}


def heat_column(config=None, dense_rho=900., water=3e-3, melt_amount=0., surface=245.15):
    '''
    A column of 60 nodes (5 cm each) at 450 kg m-3, with a dense node 5 at dense_rho and the water
    `water` [m we] at the melting point. The rest of the column is cold. Node 5 freezes more water
    than its pores hold.
    '''
    n = 60
    rho = np.full(n, 450.0)
    rho[5] = dense_rho
    rho[-3:] = 917.0
    T = np.full(n, surface)
    T[5] = T_MELT
    LWC = np.zeros(n)
    LWC[5] = water
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=T, dz=0.005, LWC=LWC, r2=1.0e-8, melt=melt_amount,
                             dt=DAY, steps=1, config={**SUBSTEP_FLOW, **(config or {})})
    column.Ts = np.array([surface])
    column.melt_solver = 'transient_solve_enthalpy'
    column.total_count = 0
    column.T10m = None
    column.heat_time_in_scheme = 0.
    return column


def solver_heat(column):
    '''Heat of the column [J m-2] in the metric of the refreezing solvers: constant specific heat of ice, plus latent.'''
    return np.sum(column.mass * CP_I * (column.Tz - T_MELT)) + np.sum(column.LWC * RHO_W_KGM * LF_I)


def total_water_mass(column):
    return np.sum(column.mass) + np.sum(column.LWC) * RHO_W_KGM   # ice and liquid [kg m-2]


@pytest.mark.unit
@pytest.mark.parametrize('solver', ['transient_solve_enthalpy', 'transient_solve_ncz'])
def test_the_solve_alone_freezes_more_than_the_pores_hold(solver):
    # The fault that the limit corrects: with no limit, the solve takes node 5 above the density of ice.
    column = heat_column()
    column.melt_solver = solver
    melt._conduct_with_ghost_node(column, 0, SUBSTEP, 245.15)
    assert column.rho[5] > RHO_I + 1.


@pytest.mark.unit
@pytest.mark.parametrize('solver', ['transient_solve_enthalpy', 'transient_solve_ncz'])
def test_the_limit_puts_the_excess_in_the_node_below(solver):
    column = heat_column()
    column.melt_solver = solver
    water_before = np.sum(column.LWC) * RHO_W_KGM
    LWC_node_6_before = column.LWC[6]
    mass_before, heat_before = total_water_mass(column), solver_heat(column)
    froze, surface_heat = melt._conduction_substep(column, 0, SUBSTEP)
    excess, runoff = column.conduction_excess_last
    assert np.all(column.rho <= RHO_I + 1e-6)
    assert column.rho[5] == pytest.approx(RHO_I, abs=1e-6)       # node 5 is full of ice
    assert excess > 1e-5                                         # the solve froze more than the pores hold
    assert runoff == 0.
    room_6 = column.dz[6] * (RHO_I - 450.) / RHO_I * 0.917 - 1e-6 * column.dz[6]   # about 2.3 mm of water; the excess is 2.9 mm
    assert column.LWC[6] - LWC_node_6_before == pytest.approx(room_6, rel=1e-6)   # the node below takes what it can hold
    assert np.sum(column.LWC[6:]) - LWC_node_6_before == pytest.approx(excess, rel=1e-9)   # and the next node takes the rest
    # Budgets: mass of ice plus water is the same, and the net ice that the nodes froze is the water that left
    assert total_water_mass(column) == pytest.approx(mass_before, rel=1e-12)
    assert froze.sum() * RHO_W_KGM == pytest.approx(water_before - np.sum(column.LWC) * RHO_W_KGM, rel=1e-9)
    # The heat: the correction changes no heat of the column (the latent heat of the excess is taken from
    # node 5). The error of the solve itself (2 J m-2 of 3.4e5, from the solver) is the same with and without it.
    error = solver_heat(column) - heat_before - surface_heat
    alone = heat_column()
    alone.melt_solver = solver
    heat_alone = solver_heat(alone)
    surface_heat_alone = melt._conduct_with_ghost_node(alone, 0, SUBSTEP, 245.15)
    error_alone = solver_heat(alone) - heat_alone - surface_heat_alone
    assert error == pytest.approx(error_alone, abs=1e-6)
    assert np.all(column.Tz <= T_MELT)


@pytest.mark.unit
def test_the_excess_is_runoff_if_an_impermeable_node_comes_first():
    column = heat_column()
    mass_before = total_water_mass(column)
    froze, _ = melt._conduction_substep(column, 0, SUBSTEP, imp=np.array([6]))
    excess, runoff = column.conduction_excess_last
    assert excess > 1e-5 and runoff == pytest.approx(excess, rel=1e-12)
    assert column.LWC[6] == 0.
    assert np.all(column.rho <= RHO_I + 1e-6)
    # The water budget closes with the runoff: ice and water in the column plus the runoff
    assert total_water_mass(column) + runoff * RHO_W_KGM == pytest.approx(mass_before, rel=1e-12)
    assert froze.sum() == pytest.approx(0. - np.sum(column.LWC) - runoff + 3e-3, rel=1e-9)   # frozen net = water lost, less the runoff


@pytest.mark.unit
def test_the_excess_is_runoff_if_the_dense_node_is_impermeable():
    column = heat_column()
    melt._conduction_substep(column, 0, SUBSTEP, imp=np.array([5]))
    excess, runoff = column.conduction_excess_last
    assert excess > 1e-5 and runoff == pytest.approx(excess, rel=1e-12)
    assert column.LWC[6] == 0.


@pytest.mark.unit
def test_the_excess_fills_the_room_of_a_node_and_goes_on():
    column = heat_column()
    theta_s = np.full(60, 1e-3 / 0.005)                  # room of 1 mm of water in each node
    theta_s[6] = 0.1e-3 / 0.005                           # node 6 has room for 0.1 mm only
    melt._conduction_substep(column, 0, SUBSTEP, theta_s=theta_s)
    excess, runoff = column.conduction_excess_last
    assert excess > 1.5e-4 and runoff == 0.
    assert column.LWC[6] == pytest.approx(0.1e-3, rel=1e-9)
    assert column.LWC[7] == pytest.approx(min(1e-3, excess - 0.1e-3), rel=1e-9)


def two_dense_nodes_column():
    # Nodes 5 and 6 are both dense and wet, so the same solve takes both above their pores.
    column = heat_column()
    column.rho[6] = 900.
    column.mass[6] = 900. * column.dz[6]
    column.Tz[6] = T_MELT
    column.LWC[6] = 3e-4   # all of it freezes in the solve: 0.3 mm is above the 0.085 mm of pore ice
    return column


@pytest.mark.unit
def test_no_water_goes_into_a_node_that_the_same_solve_froze_to_ice():
    column = two_dense_nodes_column()
    alone = two_dense_nodes_column()
    melt._conduct_with_ghost_node(alone, 0, SUBSTEP, 245.15)
    LWC_node_6_after_solve = alone.LWC[6]
    mass_before = total_water_mass(column)
    theta_s = np.maximum(0., (RHO_I - column.rho) / RHO_I) * 0.917 - 1e-6   # from the start of the sub-step, as the scheme gives it
    melt._conduction_substep(column, 0, SUBSTEP, theta_s=theta_s)
    excess, runoff = column.conduction_excess_last
    assert runoff == 0. and excess > 1e-5
    assert column.rho[5] == pytest.approx(RHO_I, abs=1e-6) and column.rho[6] == pytest.approx(RHO_I, abs=1e-6)
    assert column.LWC[6] == pytest.approx(LWC_node_6_after_solve, abs=1e-15)   # node 6 is ice: it takes no water
    assert column.LWC[7] > 0.                                     # the water goes to the next node with room
    assert total_water_mass(column) + runoff * RHO_W_KGM == pytest.approx(mass_before, rel=1e-12)


@pytest.mark.unit
def test_a_node_that_the_solve_takes_past_RhoImp_is_impermeable():
    # Node 6 is at 800 kg m-3 with 0.45 mm of water at the melting point. The solve takes it to 890 kg m-3:
    # above RhoImp (873), below the density of ice. It is not in imp.
    def column_with_node_6():
        column = heat_column()
        column.rho[6] = 800.
        column.mass[6] = 800. * column.dz[6]
        column.Tz[6] = T_MELT
        column.LWC[6] = 4.5e-4
        return column
    column = column_with_node_6()
    mass_before = total_water_mass(column)
    melt._conduction_substep(column, 0, SUBSTEP, RhoImp=873.)
    excess, runoff = column.conduction_excess_last
    assert 873. <= column.rho[6] < RHO_I
    assert excess > 1e-5 and runoff == pytest.approx(excess, rel=1e-12)   # the lens stops the water
    assert np.sum(column.LWC[6:]) == 0.
    assert total_water_mass(column) + runoff * RHO_W_KGM == pytest.approx(mass_before, rel=1e-12)
    # With no RhoImp the same excess is stored below node 5
    column = column_with_node_6()
    melt._conduction_substep(column, 0, SUBSTEP)
    assert column.conduction_excess_last[1] == 0.
    assert np.sum(column.LWC[6:]) > 0.


@pytest.mark.unit
def test_theta_s_above_the_true_pores_does_not_give_a_node_more_room():
    column = heat_column()
    column.rho[6] = 800.                                          # true pores 0.128 of the volume
    column.mass[6] = 800. * column.dz[6]
    mass_before = total_water_mass(column)
    theta_s = np.full(60, 0.9)                                    # as from rhodcy: far above the true pores
    melt._conduction_substep(column, 0, SUBSTEP, theta_s=theta_s)
    excess, runoff = column.conduction_excess_last
    assert excess > 1e-5
    assert column.LWC[6] == pytest.approx(column.dz[6] * ((RHO_I - 800.) / RHO_I * 0.917 - 1e-6), rel=1e-9)
    assert column.LWC[7] > 0.                                     # the rest goes on
    assert total_water_mass(column) + runoff * RHO_W_KGM == pytest.approx(mass_before, rel=1e-12)


@pytest.mark.unit
def test_nothing_changes_without_excess():
    column = heat_column(dense_rho=600., water=1e-5)
    LWC = column.LWC.copy()
    melt._conduction_substep(column, 0, SUBSTEP)
    assert column.conduction_excess_last == (0., 0.)
    assert column.LWC[6] == 0.
    assert np.all(column.LWC <= LWC)


@pytest.mark.scheme
def test_the_scheme_counts_the_excess_and_closes_the_water_budget():
    # One day of 48 sub-steps. Node 5 is in no lens (RhoImp_darcy 830, ThickImp_darcy 0.03: the node is
    # thinner than the lens limit), so the excess goes down, and the check passes.
    column = heat_column({'melt_strict': True})
    melt.darcyscheme(column, 0)
    d = column.melt_diagnostics
    assert d['checks']['budget']['ok'] and abs(d['checks']['budget']['residual']) < 1e-12
    assert d['checks']['density_above_ice']['ok']
    assert d['excess_from_conduction'] >= d['excess_from_conduction_runoff'] >= 0.
    assert np.all(column.rho <= RHO_I + 1e-6)


@pytest.mark.scheme
def test_the_scheme_adds_the_runoff_of_the_limit_to_the_runoff_total():
    # Node 5 and node 6 are ice, so that the excess runs off. RhoImp_darcy is below their density.
    column = heat_column({'melt_strict': True, 'RhoImp_darcy': 850, 'ThickImp_darcy': 0.})
    column.rho[6:8] = 880.
    column.mass[:] = column.rho * column.dz
    column.rho[5] = 840.
    column.mass[5] = column.rho[5] * column.dz[5]
    before = total_water_mass(column)
    melt.darcyscheme(column, 0)
    d = column.melt_diagnostics
    assert d['checks']['budget']['ok'] and abs(d['checks']['budget']['residual']) < 1e-12
    assert d['runoff'] >= d['excess_from_conduction_runoff'] > 0.
    assert total_water_mass(column) + d['runoff'] * RHO_W_KGM == pytest.approx(before, rel=1e-12)


@pytest.mark.scheme
def test_the_step_setting_makes_no_excess():
    column = heat_column({'conduction_interval_darcy': 'step'})
    melt.darcyscheme(column, 0)
    d = column.melt_diagnostics
    assert d['excess_from_conduction'] == 0. and d['excess_from_conduction_runoff'] == 0.
    assert 'density_above_ice' not in d['checks']


### The thin cold node 0 with a steady small input ###

def thin_node_column(config, node0_mass=0.07, melt_mass=0.115, surface=262.5):
    '''Node 0 holds 0.07 kg m-2 (0.16 mm), and the melt of the day is 0.125 mm of water, in 288 sub-steps of 300 s.'''
    n = 60
    dz = np.full(n, 0.05)
    dz[0] = node0_mass / 450.
    rho = np.full(n, 450.0)
    rho[-3:] = 917.0
    column = SyntheticColumn(n=n, rho_profile=rho, T_profile=np.full(n, surface), dz=dz, LWC=0., r2=1.0e-8,
                             melt=melt_mass / RHO_I, dt=DAY, steps=1,
                             config={**FLOW, 'dtsub_implicit_darcy': 300., 'conduction_interval_darcy': 'substep',
                                     'RhoImp_darcy': 900, 'ThickImp_darcy': 0., 'melt_strict': True, **config})
    column.Ts = np.array([surface])
    column.melt_solver = 'transient_solve_enthalpy'
    column.total_count = 0
    column.T10m = None
    column.heat_time_in_scheme = 0.
    return column


def run_with_lens_record(column, monkeypatch):
    '''Run darcyscheme and return, for each sub-step, whether node 0 is the top of a lens.'''
    record = []
    original = melt._implicit_flow_step

    def wrapper(model, liq_input, imptop, *args, **kwargs):
        record.append(0 in set(np.atleast_1d(imptop)))
        return original(model, liq_input, imptop, *args, **kwargs)
    monkeypatch.setattr(melt, '_implicit_flow_step', wrapper)
    melt.darcyscheme(column, 0)
    return record


@pytest.mark.scheme
def test_a_thin_cold_node_0_does_not_become_a_lens(monkeypatch, capsys):
    column = thin_node_column({'min_partial_node_dz_darcy': 1e-3})
    record = run_with_lens_record(column, monkeypatch)       # melt_strict is true: a density above ice stops the run
    d = column.melt_diagnostics
    assert len(record) == 288
    assert not any(record)                                   # node 0 is never impermeable
    assert d['partial_node_merges'] == 1
    assert d['runoff'] == 0.
    assert d['checks']['density_above_ice']['ok']
    assert column.dz[0] > 1e-3 and column.rho[0] < 600.


@pytest.mark.scheme
def test_without_the_merge_a_thin_cold_node_0_becomes_a_lens(monkeypatch, capsys):
    # The fault that the key corrects. The test shows that the one above sees it.
    column = thin_node_column({'min_partial_node_dz_darcy': 0.})
    record = run_with_lens_record(column, monkeypatch)
    assert any(record)
    assert column.melt_diagnostics['runoff'] > 0.
