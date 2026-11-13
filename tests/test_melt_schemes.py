'''
Single calls of the meltwater schemes on a synthetic column.
'''

import numpy as np
import pytest

import melt
from constants import RHO_I


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_dh_melt_is_melted_thickness(synthetic_column, scheme):
    # With a uniform density, the melted thickness is melt mass / density.
    # darcyscheme melts in sub-steps, so this also checks its sum. The column
    # is at the melting point: refreeze between sub-steps would make the
    # surface denser and the later sub-steps would melt a thinner layer.
    density, melt_ice_equivalent = 400.0, 0.02
    column = synthetic_column(n=60, rho_profile=density, T_profile=273.15, melt=melt_ice_equivalent)
    scheme_outputs = getattr(melt, scheme)(column, 0)
    assert len(scheme_outputs) == 13
    dh_melt = scheme_outputs[-1]
    assert dh_melt == pytest.approx(-melt_ice_equivalent * RHO_I / density, rel=1e-9)


class _FluxSolverCalls:
    '''Wraps the two flux solvers that darcyscheme calls, and records each call.'''

    def __init__(self, monkeypatch):
        import darcy_funcs
        self.calls = []
        for name in ('flux_bisection', 'flux_newtonraphson'):
            monkeypatch.setattr(melt, name, self._wrap(name, getattr(darcy_funcs, name)))

    def _wrap(self, name, solver):
        import darcy_funcs

        def wrapper(guess, lwc_available, lwc_accommodated, th_i, th_s, lwc, dz, alpha, n, m, eps_cvg):
            saturation = darcy_funcs.thetae_update(guess, th_i, th_s, lwc, dz)
            head = darcy_funcs.phead_vG(alpha, n, m, saturation)
            flux = solver(guess, lwc_available, lwc_accommodated, th_i, th_s, lwc, dz, alpha, n, m, eps_cvg)
            self.calls.append(dict(
                solver=name, f_eq=head[0] - head[1] - dz.sum() / 2, flux=flux,
                n_nodes=len(lwc_available), upper_bound=min(lwc_available[0], lwc_accommodated[1])))
            return flux
        return wrapper


@pytest.fixture
def darcy_wetting_front(synthetic_column, monkeypatch):
    '''
    One day of melt into dry, temperate firn, with the flux solver calls recorded.
    The front moves into dry firn, so many interfaces start with f_eq < -1.
    '''
    calls = _FluxSolverCalls(monkeypatch)
    column = synthetic_column(n=60, rho_profile=lambda z: 350.0 + 60.0 * z, T_profile=273.15,
                              dz=0.05, r2=(5e-4) ** 2, melt=0.02, dt=86400.0)
    melt.darcyscheme(column, 0)
    assert len(calls.calls) > 100
    return column, calls.calls


@pytest.mark.scheme
def test_darcy_flux_solvers_get_the_two_interface_nodes(darcy_wetting_front):
    # The upper bound of the flux at interface j1 is min(LWCav[j1], glwcacm[j1+1]).
    # The solvers read it from elements 0 and 1, so they must get the two nodes
    # at the interface, not the full column.
    _, calls = darcy_wetting_front
    assert {call['n_nodes'] for call in calls} == {2}


@pytest.mark.scheme
def test_darcy_fluxes_are_within_the_available_water_and_pore_space(darcy_wetting_front):
    column, calls = darcy_wetting_front
    for call in calls:
        assert 0.0 <= call['flux'] <= call['upper_bound'] * (1 + 1e-12), call
    saturated_water = (917.0 - column.rho) / 917.0 * 0.917 * column.dz
    assert np.all(column.LWC <= saturated_water)


@pytest.mark.scheme
def test_darcy_uses_bisection_when_the_first_guess_is_far_from_equilibrium(darcy_wetting_front):
    # darcyscheme starts with bisection when |f_eq| > 1 m at the first guess,
    # and with Newton-Raphson otherwise. The front gives f_eq < -1 at many interfaces.
    _, calls = darcy_wetting_front
    assert any(call['f_eq'] < -1.0 for call in calls)
    for call in calls:
        expected = 'flux_bisection' if abs(call['f_eq']) > 1.0 else 'flux_newtonraphson'
        assert call['solver'] == expected, call


@pytest.mark.scheme
def test_darcy_input_on_an_impermeable_column_runs_off(synthetic_column):
    # No node can store water, so all the input runs off.
    column = synthetic_column(n=40, rho_profile=900.0, T_profile=273.15, dz=0.125, r2=(5e-4) ** 2,
                              rain=0.03, dt=86400.0, config={'melt_strict': True})
    runoff = melt.darcyscheme(column, 0)[11]
    assert runoff == pytest.approx(0.03 * RHO_I / 1000.0, rel=1e-12)
    assert np.sum(column.LWC) == 0.0


@pytest.mark.scheme
@pytest.mark.parametrize('config', [{}, {'lat_runoff_darcy': False}, {'slope_proxy_darcy': 0.0}])
def test_darcy_input_above_the_capacity_of_the_column_runs_off(synthetic_column, config):
    # Firn on ice, with more rain than the pore space can hold and with little or no
    # lateral drainage. The water budget check stops the run if water is lost.
    column = synthetic_column(n=40, rho_profile=lambda z: np.where(z < 2.0, 400.0, 900.0),
                              T_profile=273.15, dz=0.125, r2=(5e-4) ** 2, rain=2.0, dt=86400.0,
                              config={'melt_strict': True, **config})
    runoff = melt.darcyscheme(column, 0)[11]
    assert runoff > 0.0
    pore_space = (917.0 - column.rho) / 917.0 * 0.917 * column.dz
    assert np.all(column.LWC <= pore_space)


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_melt_of_several_nodes_does_not_copy_the_bottom_water(synthetic_column, scheme):
    # Melt removes several nodes at the top, and the scheme adds the same number of
    # nodes at the bottom. The new nodes are dry: the water of the wet bottom node
    # is in the column once, so water in = water left + refrozen + runoff.
    lwc = np.zeros(60)
    lwc[-1] = 1.4e-4
    column = synthetic_column(n=60, rho_profile=400.0, T_profile=273.15, dz=0.05, LWC=lwc,
                              r2=(5e-4) ** 2, melt=0.03, dt=86400.0,
                              config=dict(Ponding=False, DirectRunoff=0.0, RunoffZuoOerlemans=False, Slope=0.0))
    getattr(melt, scheme)(column, 0)
    budget = column.melt_diagnostics
    assert budget['lwc_end'] + budget['refrozen'] + budget['runoff'] == pytest.approx(budget['water_in'], abs=1e-12)
    # The bottom water is below the irreducible content, so it does not move, and the
    # node added last at the bottom stays dry.
    assert column.LWC[-1] == 0.0
