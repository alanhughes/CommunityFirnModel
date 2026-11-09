'''
The front of darcyscheme against the oracle with the darcy flux law. A constant rain enters
dry uniform firn at the melting temperature. Nothing freezes, and no node melts.

Mass balance fixes the centre of mass of the water once the water content behind the front is
known. So the plateau behind the front is the real test of the flux law, and the speed follows.

The limits of the numerics are in the tests:

- The sub-step control keeps qstep/qlim up to 0.75, so qstep/dtsub is below the law. The plateau
  is then a few percent above the oracle, at any node thickness. A short fixed sub-step removes
  this difference, so the law itself is correct.
- At thin nodes, the sub-step stops at dtmin_darcy. Then qlim sets the flux, and the plateau rises
  until qlim carries the input. The error grows as the nodes get thinner.
'''

import sys
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import brentq

import melt
from darcy_funcs import hydrconducsat_Calonne, krel_vG, phead_vG, vG_Yama_params
from oracles.kinematic_wave import FirnConstants, solve_front

DAY = 86400.0
DENSITY = 400.0
RADIUS = 1e-3
THETA_S_FACTOR = 0.917
THETA_IRREDUCIBLE = 0.02
CONSTANTS = FirnConstants(rho_ice=917.0, rho_water=1000.0, specific_heat=2097.0, latent_heat=334000.0)
POROSITY = 1 - DENSITY / 917.0
THETA_S = THETA_S_FACTOR * POROSITY - 1e-6
# darcyscheme reads IrrVal as a fraction of the pore space, POROSITY * 917 / 1000 of the volume.
IRR_VAL = THETA_IRREDUCIBLE / (POROSITY * 0.917)
CALLS = 24  # calls of darcyscheme in a run

TOLERANCE = 0.04  # default sub-steps. It holds the largest difference in RESOLVED, 3.6 %
TOLERANCE_SHORT_STEP = 0.005  # short fixed sub-steps, for the plateau


def conductivity(theta):
    '''hydrconducsat_Calonne times krel_vG, as darcyscheme computes them [m s-1].'''
    theta_e = np.clip((theta - THETA_IRREDUCIBLE) / (THETA_S - THETA_IRREDUCIBLE), 1e-9, 1 - 1e-9)
    return float(hydrconducsat_Calonne(RADIUS, DENSITY) * krel_vG(vG_Yama_params(RADIUS, DENSITY)[2], theta_e))


def darcy_law(theta, porosity):
    '''The law in the form that the oracle needs. No front in these tests has cold firn ahead.'''
    return conductivity(theta)


def oracle(flux):
    return solve_front(darcy_law, flux, POROSITY, CONSTANTS, saturated_fraction=THETA_S_FACTOR)


@dataclass
class FrontRun:
    plateau: float  # water content behind the front
    speed: float  # [m s-1], from the second half of the run
    water: float  # water in the column at the end [m w.e.]
    bottom_water: float  # water in the lowest nodes at the end [m w.e.]
    runoff: float  # [m w.e.]
    substeps: int
    substeps_spread: int  # sub-steps that spread the input below the top node


def _front_depth(theta, depth, plateau):
    '''The depth where the water content crosses half of the plateau, between node centres.'''
    half = 0.5 * plateau
    below = np.where(theta > half)[0][-1]
    return depth[below] + (theta[below] - half) / (theta[below] - theta[below + 1]) * (depth[below + 1] - depth[below])


def _plateau(theta, plateau_guess):
    '''
    The mean water content of the middle half of the wet part of the column. Not the median:
    near the limit of the limiter, neighbouring nodes alternate between two values.
    '''
    last_wet = np.where(theta > 0.5 * plateau_guess)[0][-1]
    return float(np.mean(theta[last_wet // 4:3 * last_wet // 4 + 1]))


def run_front(monkeypatch, synthetic_column, flux, dz, duration, calls=CALLS, **config):
    '''Call darcyscheme in a loop. flux is in m w.e. s-1.'''
    dt = duration / calls
    config = {'ColeouLesaffre': False, 'IrrVal': IRR_VAL, **config}
    front = oracle(flux)
    n = int(1.5 * front.speed * duration / dz) + 30
    column = synthetic_column(n=n, rho_profile=DENSITY, T_profile=273.15, dz=dz, r2=RADIUS ** 2,
                              rain=flux * dt / 0.917, dt=dt, config=config)

    # darcyscheme calls vG_Yama_params once in each sub-step, after it places the input.
    # So its caller holds i_input_bottom, the lowest node that receives the input in this sub-step.
    input_nodes = []

    def vG_Yama_params_and_record(*args):
        input_nodes.append(sys._getframe(1).f_locals['i_input_bottom'])
        return vG_Yama_params(*args)

    monkeypatch.setattr(melt, 'vG_Yama_params', vG_Yama_params_and_record)

    depth = np.cumsum(column.dz) - column.dz / 2
    times, depths = [], []
    runoff = 0.0
    for call in range(calls):
        melt.darcyscheme(column, 0)
        runoff += column.melt_diagnostics['runoff']
        theta = column.LWC / column.dz
        if call >= calls // 2:
            times.append((call + 1) * dt)
            depths.append(_front_depth(theta, depth, _plateau(theta, front.water_behind)))

    return FrontRun(plateau=_plateau(theta, front.water_behind), speed=np.polyfit(times, depths, 1)[0],
                    water=float(column.LWC.sum()), bottom_water=float(column.LWC[-5:].sum()), runoff=runoff,
                    substeps=len(input_nodes), substeps_spread=sum(node > 0 for node in input_nodes))


def assert_matches_oracle(run, flux, tolerance=TOLERANCE):
    front = oracle(flux)
    assert run.plateau == pytest.approx(front.water_behind, rel=tolerance)
    assert run.speed == pytest.approx(front.speed, rel=tolerance)


# flux [m/s], node thickness [m], duration [days]. In each case the front crosses 45 nodes or more.
RESOLVED = [(5e-8, 0.02, 5.0), (5e-8, 0.01, 2.5), (5e-7, 0.02, 1.0), (5e-7, 0.01, 0.5), (1e-6, 0.02, 0.5)]
THIN = [(5e-8, 0.004, 1.25), (5e-7, 0.004, 0.5), (1e-6, 0.01, 0.5), (1e-6, 0.004, 0.25)]


@pytest.mark.scheme
@pytest.mark.parametrize('flux, dz, days', RESOLVED)
def test_front_matches_oracle(monkeypatch, synthetic_column, flux, dz, days):
    assert_matches_oracle(run_front(monkeypatch, synthetic_column, flux, dz, days * DAY), flux)


@pytest.mark.scheme
@pytest.mark.xfail(strict=True, reason='at thin nodes the sub-step stops at dtmin_darcy, and qlim sets the flux')
@pytest.mark.parametrize('flux, dz, days', THIN)
def test_front_matches_oracle_at_thin_nodes(monkeypatch, synthetic_column, flux, dz, days):
    assert_matches_oracle(run_front(monkeypatch, synthetic_column, flux, dz, days * DAY), flux)


@pytest.mark.scheme
@pytest.mark.parametrize('flux, dz, days', RESOLVED[:1] + THIN[-1:])
def test_budget_and_input(monkeypatch, synthetic_column, flux, dz, days):
    '''All the water stays in the column. The input enters at the top node as a flux.'''
    run = run_front(monkeypatch, synthetic_column, flux, dz, days * DAY)
    assert run.water == pytest.approx(flux * days * DAY, rel=1e-10)
    assert run.runoff == 0.0
    assert run.bottom_water == 0.0
    assert run.substeps_spread == 0


@pytest.mark.scheme
def test_spread_count_sees_a_large_input(monkeypatch, synthetic_column):
    '''
    The count of test_budget_and_input can see spreading. With this flux, far above any melt rate,
    the limiter fills the top nodes. The water still stays in the column.
    '''
    flux, duration = 1e-4, 3600.0
    run = run_front(monkeypatch, synthetic_column, flux, 0.004, duration)
    assert run.substeps_spread > 0
    assert run.runoff == 0.0
    assert run.water == pytest.approx(flux * duration, rel=1e-10)


@pytest.mark.scheme
@pytest.mark.parametrize('flux, dz, days, dtsub', [(5e-8, 0.02, 2.5, 60.0), (1e-6, 0.02, 0.125, 5.0)])
def test_short_fixed_substep_gives_the_law(monkeypatch, synthetic_column, flux, dz, days, dtsub):
    '''With qstep well below qlim, the plateau is the root of the law.'''
    run = run_front(monkeypatch, synthetic_column, flux, dz, days * DAY, dtmin_darcy=dtsub, dtmax_darcy=dtsub)
    assert run.plateau == pytest.approx(oracle(flux).water_behind, rel=TOLERANCE_SHORT_STEP)


@pytest.mark.scheme
def test_shorter_dtmin_recovers_thin_nodes(monkeypatch, synthetic_column):
    '''The same column as the worst thin case, with a shorter dtmin_darcy. So the floor causes the error.'''
    flux, dz = 1e-6, 0.004
    # Calls of one hour. Each call starts at dtsub_darcy (60 s in the base config), and
    # dtmin_darcy does not bound that first sub-step. So short calls keep part of the error.
    run = run_front(monkeypatch, synthetic_column, flux, dz, 0.125 * DAY, calls=3, dtmin_darcy=1.0)
    assert_matches_oracle(run, flux)


def limited_plateau(flux, dz, dtmin):
    '''
    The plateau when each sub-step is at dtmin. The solve for qlim goes from the bottom up, so the
    lower node of an interface has passed on what it gains: h(theta - qlim/dz) - h(theta) = dz.
    Then flux*dtmin = qlim*(1 - exp(-dtmin*K/qlim)).
    '''
    alpha, n, m = vG_Yama_params(RADIUS, DENSITY)

    def head(theta):
        return phead_vG(alpha, n, m, (theta - THETA_IRREDUCIBLE) / (THETA_S - THETA_IRREDUCIBLE))

    def limited_flux(theta):
        largest = (theta - THETA_IRREDUCIBLE - 1e-9) * dz
        qlim = brentq(lambda g: head(theta - g / dz) - head(theta) - dz, 0.0, largest)
        return qlim * (1 - np.exp(-dtmin * conductivity(theta) / qlim)) / dtmin

    return brentq(lambda theta: limited_flux(theta) - flux, THETA_IRREDUCIBLE + 2e-3, 0.3)


@pytest.mark.scheme
@pytest.mark.parametrize('flux, dz, days', [(5e-7, 0.004, 0.5), (1e-6, 0.004, 0.25)])
def test_limiter_sets_the_plateau_at_thin_nodes(monkeypatch, synthetic_column, flux, dz, days):
    '''The plateau follows qlim and dtmin, far from the law.'''
    run = run_front(monkeypatch, synthetic_column, flux, dz, days * DAY)
    assert run.plateau == pytest.approx(limited_plateau(flux, dz, 60.0), rel=0.01)
    assert run.plateau > 1.5 * oracle(flux).water_behind


def run_with_ten_times_k(monkeypatch, synthetic_column, flux, dz, days):
    monkeypatch.setattr(melt, 'hydrconducsat_Calonne', lambda *args: 10 * hydrconducsat_Calonne(*args))
    return run_front(monkeypatch, synthetic_column, flux, dz, days * DAY)


@pytest.mark.scheme
def test_ten_times_k_follows_the_law_at_resolved_nodes(monkeypatch, synthetic_column):
    '''The law with 10 K carries the flux R where the law with K carries R/10.'''
    flux, dz, days = 1e-6, 0.02, 0.5
    base = run_front(monkeypatch, synthetic_column, flux, dz, days * DAY)
    changed = run_with_ten_times_k(monkeypatch, synthetic_column, flux, dz, days)
    law_ratio = oracle(flux / 10).water_behind / oracle(flux).water_behind
    assert changed.plateau / base.plateau == pytest.approx(law_ratio, rel=0.01)


@pytest.mark.scheme
def test_ten_times_k_has_no_effect_at_thin_nodes(monkeypatch, synthetic_column):
    '''The limiter caps the flux, so a larger K does not change the plateau. The law says -20 %.'''
    flux, dz, days = 1e-6, 0.004, 0.25
    base = run_front(monkeypatch, synthetic_column, flux, dz, days * DAY)
    changed = run_with_ten_times_k(monkeypatch, synthetic_column, flux, dz, days)
    assert changed.plateau == pytest.approx(base.plateau, rel=1e-3)
