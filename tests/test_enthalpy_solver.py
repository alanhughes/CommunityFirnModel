'''
Tests for transient_solve_enthalpy when a layer holds liquid water below 0 C.

The prefsnowpack and resingledomain schemes can return such layers: water in
the preferential-flow domain, round-off residue of the prewetting, and the
temperature average in combineCFM. The solver must refreeze that water with
its latent heat, as if the layer was in equilibrium before the step.
'''

import numpy as np
import pytest

import solver
from constants import CP_I, CP_W, LF_I

N = 20
DZ = 0.1
RHO = 450.0
DT = 15 * 86400.0
COLD_WET = 5  # index of the layer with liquid water below 0 C


def _solve(T_C, th_liquid):
    z_edges = np.arange(N + 1) * DZ
    Z_P = z_edges[:-1] + DZ / 2
    Gamma_P = np.full(N, 0.5)
    th_solid = RHO - th_liquid
    return solver.transient_solve_enthalpy(z_edges, Z_P, DT, Gamma_P, T_C, th_liquid,
                                           th_solid, 0, bc_u=T_C[0])


def _cold_column(liquid):
    T_C = np.linspace(-5.0, -12.0, N)
    th_liquid = np.zeros(N)
    th_liquid[COLD_WET] = liquid
    return T_C, th_liquid


def _refrozen_first(T_C, th_liquid):
    '''Refreeze all liquid at constant enthalpy (valid when the layer stays below 0 C).'''
    H = (RHO - th_liquid) * CP_I * T_C + th_liquid * (CP_W * T_C + LF_I)
    T_eq = H / (RHO * CP_I)
    assert np.all(T_eq <= 0.0)
    return T_eq, np.zeros(N)


@pytest.mark.unit
@pytest.mark.parametrize('liquid', [1e-19, 2.0, 20.0])  # kg m-3
def test_cold_liquid_refreezes_with_its_latent_heat(liquid):
    out = _solve(*_cold_column(liquid))
    ref = _solve(*_refrozen_first(*_cold_column(liquid)))
    np.testing.assert_allclose(out['TzC_return'], ref['TzC_return'], rtol=0, atol=1e-8)
    np.testing.assert_allclose(out['th_liquid'], ref['th_liquid'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(out['th_solid'], ref['th_solid'], rtol=0, atol=1e-8)


@pytest.mark.unit
@pytest.mark.parametrize('liquid', [1e-19, 2.0])  # kg m-3
def test_no_clamp_reported_for_cold_liquid(liquid):
    out = _solve(*_cold_column(liquid))
    assert out['claw_mushy'] == 0.0
