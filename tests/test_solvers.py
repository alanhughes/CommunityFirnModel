'''
Tests for the solvers in solver.py.

- solver(): the tridiagonal solve is compared with scipy.linalg.solve_banded.
- transient_solve_TR: heat conduction with no phase change. It is compared with
  the solution for a step change at the surface of a semi-infinite slab,
      T(z, t) = T_0 + (T_s - T_0) erfc(z / (2 sqrt(kappa t))),
  which is valid while the heat has not reached the bottom of the domain.
- The four meltwater solvers (transient_solve_enthalpy, _ncz, _decp, _ahc):
  energy conservation with latent heat, T <= 0 C, mass conservation, agreement
  with transient_solve_TR in a dry column, and the position of a refreezing
  front. The tolerance for each solver follows docs/extras/meltwater_solver.rst.
  ahc is documented as not conservative and not accurate. Its energy and front
  tests are strict xfail tests.

Grid convention in solver.py: node 0 has a Dirichlet temperature. The last node
is a zero-gradient boundary row, that is, it copies the node above it. Only
nodes 1 to n-2 are control volumes with an energy balance. The energy balance in
this file uses those nodes and the conduction flux across the face between
nodes 0 and 1. With a uniform grid and a uniform conductivity k, that flux is
k (T_0 - T_1) / dz.
'''

import numpy as np
import pytest
from scipy.linalg import solve_banded
from scipy.special import erfc

import solver
from constants import CP_I, LF_I

DAY = 86400.0  # s
CONDUCTIVITY = 0.5  # W m-1 K-1
RHO = 450.0  # kg m-3, ice and liquid water mass per total volume


def _grid(n, dz):
    z_edges = np.arange(n + 1) * dz
    return z_edges, z_edges[:-1] + dz / 2


def _energy_imbalance(z_edges, dt, T_old, solid_old, liquid_old, result):
    '''Enthalpy change of nodes 1 to n-2, minus the heat in across the 0|1 face [J m-2].'''
    dz = np.diff(z_edges)
    T_new = result['TzC_return']
    H_old = solver.enthalpy_of(T_old, solid_old, liquid_old)
    H_new = solver.enthalpy_of(T_new, result['th_solid'], result['th_liquid'])
    heat_in = CONDUCTIVITY / dz[0] * (T_new[0] - T_new[1]) * dt
    return np.sum(((H_new - H_old) * dz)[1:-1]) - heat_in


def _refreeze_column(n=60, dz=0.05):
    '''
    A cold column with a wet zone at 0 C and a cold surface. The bottom node has
    a temperature gradient above it, so it changes during a step.
    '''
    z_edges, Z_P = _grid(n, dz)
    T = np.linspace(-2.0, -8.0, n)
    liquid = np.zeros(n)
    wet = slice(5, 25)
    T[wet] = 0.0
    liquid[wet] = 20.0
    T[0] = -10.0
    return z_edges, Z_P, T, RHO - liquid, liquid


def _wet_column(n=200, dz=0.05):
    '''A deep column at 0 C with 50 kg m-3 of liquid water and a dry surface node at -10 C.'''
    z_edges, Z_P = _grid(n, dz)
    T = np.zeros(n)
    liquid = np.full(n, 50.0)
    T[0] = -10.0
    liquid[0] = 0.0
    return z_edges, Z_P, T, RHO - liquid, liquid


SOLVERS = {
    'enthalpy': (solver.transient_solve_enthalpy, {}),
    'ncz': (solver.transient_solve_ncz, {}),
    'decp': (solver.transient_solve_decp, {}),
    'ahc': (solver.transient_solve_ahc, {}),
}

# Largest cumulative energy imbalance, as a fraction of the initial latent heat.
# enthalpy and decp conserve energy to round-off. ncz keeps a small amount of
# liquid in the window eps below 0 C, so its error is proportional to eps.
ENERGY_TOLERANCE = {'enthalpy': 1e-10, 'ncz': 1e-4, 'decp': 1e-10, 'ahc': 1e-4}


def _run(name, column, steps, dt=DAY, **overrides):
    '''Run a solver for a number of steps. Yield (T_old, solid_old, liquid_old, result) for each step.'''
    function, kwargs = SOLVERS[name]
    kwargs = {**kwargs, **overrides}
    z_edges, Z_P, T, solid, liquid = column
    conductivity = np.full(T.size, CONDUCTIVITY)
    for step in range(steps):
        result = function(z_edges, Z_P, dt, conductivity, T, liquid, solid, step, **kwargs)
        yield T, solid, liquid, result
        T, solid, liquid = result['TzC_return'], result['th_solid'], result['th_liquid']


def _cumulative_imbalance(name, column, steps, **overrides):
    z_edges = column[0]
    return sum(_energy_imbalance(z_edges, DAY, T, solid, liquid, result)
               for T, solid, liquid, result in _run(name, column, steps, **overrides))


def _latent_heat(column):
    z_edges, _, _, _, liquid = column
    return np.sum(liquid * np.diff(z_edges)) * LF_I


def _front_depths(name, steps):
    '''Depth of the top edge of the first node that still holds liquid, after each step.'''
    column = _wet_column()
    z_edges = column[0]
    return np.array([z_edges[np.argmax(result['th_liquid'] > 0.0)]
                     for *_, result in _run(name, column, steps)])


# ---------------------------------------------------------------------------
# solver(): tridiagonal solve
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('use_dgtsv', [True, False])
@pytest.mark.parametrize('n', [3, 10, 100, 2000])
def test_tridiagonal_solver_matches_banded_solve(n, use_dgtsv):
    rng = np.random.default_rng(n)
    a_U = rng.uniform(0.1, 1.0, n)
    a_D = rng.uniform(0.1, 1.0, n)
    a_P = a_U + a_D + rng.uniform(0.1, 1.0, n)  # diagonally dominant
    b = rng.normal(size=n)
    # Row i: a_U[i] phi[i-1] - a_P[i] phi[i] + a_D[i] phi[i+1] = -b[i]
    banded = np.zeros((3, n))
    banded[0, 1:] = a_D[:-1]
    banded[1] = -a_P
    banded[2, :-1] = a_U[1:]
    expected = solve_banded((1, 1), banded, -b)
    result = solver.solver(a_U, a_D, a_P, b, use_dgtsv=use_dgtsv)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# transient_solve_TR: heat conduction, no phase change (Kelvin)
# ---------------------------------------------------------------------------

def _solve_TR(z_edges, Z_P, dt, T_K, T_surface_K, c_vol):
    n = T_K.size
    return solver.transient_solve_TR(z_edges, Z_P, dt, np.full(n, CONDUCTIVITY), T_K, n,
                                     T_surface_K, np.full(n, c_vol))


@pytest.mark.unit
def test_TR_uniform_temperature_stays_uniform():
    z_edges, Z_P = _grid(50, 0.1)
    T = np.full(50, 250.0)
    result = _solve_TR(z_edges, Z_P, DAY, T, 250.0, RHO * CP_I)
    np.testing.assert_allclose(result, 250.0, rtol=0, atol=1e-10)


@pytest.mark.unit
def test_TR_step_change_matches_erfc_solution():
    dz, dt, duration = 0.01, 60.0, DAY
    z_edges, Z_P = _grid(300, dz)
    c_vol = 917.0 * CP_I
    kappa = CONDUCTIVITY / c_vol
    T_initial, T_surface = 263.15, 253.15
    T = np.full(Z_P.size, T_initial)
    T[0] = T_surface
    for _ in range(int(duration / dt)):
        T = _solve_TR(z_edges, Z_P, dt, T, T_surface, c_vol)
    # The Dirichlet node is the surface of the slab.
    depth = Z_P - Z_P[0]
    assert 6 * np.sqrt(kappa * duration) < depth[-1]  # heat has not reached the bottom
    expected = T_initial + (T_surface - T_initial) * erfc(depth / (2 * np.sqrt(kappa * duration)))
    error = np.max(np.abs(T - expected)) / abs(T_surface - T_initial)
    assert error < 1e-3


@pytest.mark.unit
def test_TR_conserves_energy_with_zero_flux_bottom():
    dz, dt = 0.05, DAY
    z_edges, Z_P = _grid(40, dz)
    c_vol = RHO * CP_I
    rng = np.random.default_rng(1)
    T_old = 250.0 + rng.uniform(-5.0, 5.0, Z_P.size)
    T_new = _solve_TR(z_edges, Z_P, dt, T_old.copy(), 245.0, c_vol)
    heat_in = CONDUCTIVITY / dz * (T_new[0] - T_new[1]) * dt
    change = np.sum((c_vol * (T_new - T_old) * dz)[1:-1])
    assert change == pytest.approx(heat_in, rel=1e-10)


@pytest.mark.unit
@pytest.mark.xfail(strict=True, raises=NameError,
                   reason='The airdict branch uses f_u and f_d, which _fv_geometry does not return')
def test_TR_gas_path_without_gravity_thermal_or_advection_is_plain_diffusion():
    # With open porosity 1, no eddy diffusion, and gravity, thermal fractionation
    # and advection off, the gas path solves the same equation as the heat path
    # with c_vol = 1.
    n, dz, dt = 30, 0.5, DAY
    z_edges, Z_P = _grid(n, dz)
    diffusivity = np.full(n, 1e-5)
    phi = np.linspace(1.0, 0.0, n)
    airdict = {'por_op': np.ones(n), 'd_eddy': np.full(n, 1e-30), 'gravity': 'off',
               'thermal': 'off', 'advection_type': 'zero', 'rho': np.full(n, RHO),
               'z_co': z_edges[-1] + 1.0}
    gas, _ = solver.transient_solve_TR(z_edges, Z_P, dt, diffusivity, phi.copy(), n, 1.0,
                                       np.ones(n), airdict)
    heat = solver.transient_solve_TR(z_edges, Z_P, dt, diffusivity, phi.copy(), n, 1.0,
                                     np.ones(n))
    np.testing.assert_allclose(gas, heat, rtol=1e-8)


# ---------------------------------------------------------------------------
# Meltwater solvers (deg C, fusion at 0)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('name', ['enthalpy', 'ncz', 'decp'])
def test_meltwater_solver_conserves_energy(name):
    column = _refreeze_column()
    # decp sub-steps and has no single flux for the full step. With one sub-step,
    # the flux from the final temperatures is the flux it used.
    overrides = {'iters': 1} if name == 'decp' else {}
    imbalance = _cumulative_imbalance(name, column, steps=30, **overrides)
    assert abs(imbalance) < ENERGY_TOLERANCE[name] * _latent_heat(column)


@pytest.mark.unit
def test_ncz_energy_error_falls_in_proportion_to_eps():
    column = _refreeze_column()
    coarse = _cumulative_imbalance('ncz', column, steps=30, eps=1e-4)
    fine = _cumulative_imbalance('ncz', column, steps=30, eps=1e-6)
    assert 50 < coarse / fine < 200


@pytest.mark.unit
@pytest.mark.parametrize('name', ['enthalpy', 'ncz', 'decp', 'ahc'])
def test_meltwater_solver_keeps_temperature_at_most_melting_point(name):
    for *_, result in _run(name, _refreeze_column(), steps=30):
        assert np.all(result['TzC_return'] <= 0.0)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['enthalpy', 'ncz', 'decp', 'ahc'])
def test_meltwater_solver_conserves_mass_and_only_refreezes(name):
    for _, solid, liquid, result in _run(name, _refreeze_column(), steps=30):
        np.testing.assert_allclose(result['th_solid'] + result['th_liquid'], solid + liquid,
                                   rtol=1e-12)
        assert np.all(result['th_liquid'] >= 0.0)
        assert np.all(result['th_liquid'] <= liquid + 1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['enthalpy', 'ncz', 'decp', 'ahc'])
def test_dry_column_matches_transient_solve_TR(name):
    z_edges, Z_P = _grid(40, 0.05)
    T_C = np.linspace(-15.0, -5.0, 40)
    dry = np.zeros(40)
    solid = np.full(40, RHO)
    overrides = {'iters': 1} if name == 'decp' else {}
    *_, result = next(_run(name, (z_edges, Z_P, T_C, solid, dry), steps=1, **overrides))
    T_K = _solve_TR(z_edges, Z_P, DAY, T_C + 273.15, T_C[0] + 273.15, RHO * CP_I)
    np.testing.assert_allclose(result['TzC_return'], T_K - 273.15, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(result['th_liquid'], 0.0)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['ncz', 'decp'])
def test_refreezing_front_agrees_with_enthalpy(name):
    # 20 daily steps move the front about 1 m. decp at its default iters=10 lags
    # by up to one node.
    reference = _front_depths('enthalpy', steps=20)
    assert reference[-1] > 0.5
    np.testing.assert_allclose(_front_depths(name, steps=20), reference, rtol=0,
                               atol=0.05 + 1e-9)


@pytest.mark.unit
def test_ahc_runs_and_returns_finite_values():
    for *_, result in _run('ahc', _refreeze_column(), steps=30):
        for key in ('TzC_return', 'th_solid', 'th_liquid'):
            assert np.all(np.isfinite(result[key]))


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason='ahc does not conserve energy (docs/extras/meltwater_solver.rst)')
def test_ahc_conserves_energy():
    column = _refreeze_column()
    imbalance = _cumulative_imbalance('ahc', column, steps=30)
    assert abs(imbalance) < ENERGY_TOLERANCE['ahc'] * _latent_heat(column)


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason='ahc fails the analytic benchmark (docs/extras/meltwater_solver.rst)')
def test_ahc_refreezing_front_agrees_with_enthalpy():
    np.testing.assert_allclose(_front_depths('ahc', steps=20), _front_depths('enthalpy', steps=20),
                               rtol=0, atol=0.05 + 1e-9)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['enthalpy', 'ncz', 'decp', 'ahc'])
def test_energy_resid_excludes_the_bottom_boundary_node(name):
    # The bottom node of _refreeze_column changes in the step. It is a boundary
    # row, not a control volume, so energy_resid must not include it.
    column = _refreeze_column()
    overrides = {'iters': 1} if name == 'decp' else {}
    T, solid, liquid, result = next(_run(name, column, steps=1, **overrides))
    expected = _energy_imbalance(column[0], DAY, T, solid, liquid, result)
    assert result['energy_resid'] == pytest.approx(expected, rel=1e-9, abs=1e-6)


@pytest.mark.unit
def test_decp_energy_resid_uses_the_sub_step_fluxes():
    column = _refreeze_column()
    *_, result = next(_run('decp', column, steps=1))
    assert abs(result['energy_resid']) < 1e-10 * _latent_heat(column)
