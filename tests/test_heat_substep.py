'''
The argument dt of heatDiff and refreezeDiff: the default is the full step, and a shorter dt gives
a result that is close to the full step and keeps the mass.
'''

from types import SimpleNamespace

import numpy as np
import pytest

from diffusion import heatDiff, refreezeDiff

pytestmark = pytest.mark.unit

DAY = 86400.
SOLVERS = ['transient_solve_' + name for name in ('enthalpy', 'ahc', 'decp', 'ncz')]


def column(water=False):
    n = 200
    dz = np.full(n, 0.1)
    z = np.cumsum(dz) - dz / 2
    rho = 350 + 500 * (1 - np.exp(-z / 15))
    Tz = 253.15 + 5 * np.exp(-z / 2)
    LWC = np.zeros(n)
    if water:
        Tz[:5] = 273.15
        LWC[:5] = 0.002
    return SimpleNamespace(z=z, dz=dz, rho=rho, mass=rho * dz, Tz=Tz, LWC=LWC,
                           c={'conductivity': 'Calonne2019', 'MELT': water, 'LWCheat': 'enthalpy'},
                           dt=np.array([DAY]), modeltime=np.array([0.]), T10m=None, total_count=0)


def heat_steps(model, durations):
    for dt in durations:
        model.Tz, model.T10m = heatDiff(model, 0, dt=dt)
    return model


def refreeze_steps(model, durations, solver):
    for dt in durations:
        model.Tz, model.T10m, model.rho, model.mass, model.LWC, _ = refreezeDiff(
            model, 0, solver_name=solver, dt=dt)
    return model


def total_mass(model):
    return np.sum(model.mass + model.LWC * 1000)


def test_default_dt_is_the_full_step():
    a = heatDiff(column(), 0)
    b = heatDiff(column(), 0, dt=DAY)
    assert np.array_equal(a[0], b[0])

    ma, mb = column(True), column(True)
    ra = refreezeDiff(ma, 0, solver_name='transient_solve_enthalpy')
    rb = refreezeDiff(mb, 0, solver_name='transient_solve_enthalpy', dt=mb.dt[0])
    for x, y in zip(ra[:5], rb[:5]):
        assert np.array_equal(x, y)
    assert ra[5] == rb[5]


def test_dry_column_two_half_steps_equal_one_step():
    full = heat_steps(column(), [DAY])
    half = heat_steps(column(), [DAY / 2, DAY / 2])
    difference = np.abs(full.Tz - half.Tz).max()
    print('dry column, max |full - 2 half| [K]:', difference)
    assert difference < 0.01


@pytest.mark.parametrize('solver', SOLVERS)
def test_wet_column_substeps_keep_mass_and_bounds(solver):
    model = column(True)
    mass_before = total_mass(model)
    refreeze_steps(model, [1800.] * 48, solver)
    assert abs(total_mass(model) - mass_before) < 1e-9
    assert np.all(model.Tz <= 273.15)
    assert np.all(model.LWC >= 0)


def test_wet_column_substeps_are_close_to_one_step():
    solver = 'transient_solve_enthalpy'
    full = refreeze_steps(column(True), [DAY], solver)
    subs = refreeze_steps(column(True), [1800.] * 48, solver)
    full_lwc, sub_lwc = full.LWC.sum() * 1000, subs.LWC.sum() * 1000
    print('wet column LWC [kg/m2], one step:', full_lwc, ' 48 sub-steps:', sub_lwc, ' difference:', sub_lwc - full_lwc)
    assert abs(sub_lwc - full_lwc) < 0.2
