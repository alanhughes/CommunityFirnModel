'''
The thin partial node 0 of the darcy scheme with the implicit flow step.

The melt removal of each sub-step leaves one node partly melted (the partial node, the new node 0).
The key min_partial_node_dz_darcy merges a partial node that is thinner than the key into the node
below. The tests call darcyscheme on a synthetic column.
'''

import numpy as np
import pytest

import melt
from cfm_testing import SyntheticColumn
from constants import RHO_I, RHO_W_KGM, T_MELT

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
