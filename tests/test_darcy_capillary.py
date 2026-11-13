'''
Tests of the capillary step of the implicit gravity flow of darcyscheme (config key
capillary_implicit_darcy, darcy_implicit.py).

The unit tests check capillary_interface_terms and capillary_implicit_step on their own. The scheme
tests call darcyscheme on synthetic columns. The oracle is the travelling wave of Meyer and Hewitt (2017)
for constant rain on uniform firn at 0 C, with the power laws for K_sat and k_r. The tolerances of the
oracle tests are set from the measured errors, which are written next to them.
'''

import numpy as np
import pytest

import melt
from cfm_testing import SyntheticColumn
from constants import GRAVITY, RHO_I, RHO_I_MGM, RHO_W_KGM
from darcy_funcs import hydrconducsat_power_law
from darcy_implicit import capillary_implicit_step, capillary_interface_terms
from test_darcy_implicit import EDGE_CASES, IMPLICIT, edge_column, lens_column

CAPILLARY = {'capillary_implicit_darcy': 'power_law'}


### The function ###

def random_column(rng, n=30, closed_fraction=0.0, sink_fraction=0.0):
    '''Water, floor, saturated water, capacity, conductivity and thickness of a random column.'''
    dz = rng.uniform(0.004, 0.03, n)
    saturated = dz * rng.uniform(0.3, 0.5, n)
    floor = saturated * rng.uniform(0., 0.3, n)
    lwc = floor + (saturated - floor) * rng.uniform(0., 1., n) * (rng.uniform(size=n) < 0.6)
    sink = rng.uniform(size=n) < sink_fraction
    lwc[sink] = floor[sink] * rng.uniform(0., 0.9, sink.sum())
    conductivity = 10 ** rng.uniform(-6, -3, n)
    closed = rng.uniform(size=n) < closed_fraction
    return lwc, floor, saturated, saturated.copy(), conductivity, dz, closed


def step(column, dtsub=600., head=0.05):
    lwc, floor, saturated, capacity, conductivity, dz, closed = column
    terms = capillary_interface_terms(conductivity, dz, dtsub, head)
    return capillary_implicit_step(lwc, floor, saturated, capacity, terms, closed)


@pytest.mark.unit
def test_interface_terms_are_the_thickness_weighted_harmonic_mean():
    conductivity = np.array([1e-4, 4e-4, 0., 2e-4])
    dz = np.array([0.01, 0.03, 0.02, 0.02])
    terms = capillary_interface_terms(conductivity, dz, 100., 0.07)
    harmonic = 0.04 / (0.01 / 1e-4 + 0.03 / 4e-4)
    assert terms[0] == pytest.approx(100. * harmonic * 0.07 / 0.02, rel=1e-14)
    assert terms[1] == 0. and terms[2] == 0.   # a conductivity of zero closes both interfaces of the node


@pytest.mark.unit
def test_a_node_of_low_conductivity_limits_the_exchange_through_the_step():
    '''A wet node over a node with a conductivity 1e-6 times lower: the harmonic mean keeps the water almost in place.'''
    saturated, floor = np.full(2, 0.004), np.zeros(2)
    lwc = np.array([0.004, 0.0])
    closed = np.zeros(2, dtype=bool)
    moved = []
    for low in (1e-3, 1e-9):
        terms = capillary_interface_terms(np.array([1e-3, low]), np.full(2, 0.01), 600., 0.07)
        new, _ = capillary_implicit_step(lwc, floor, saturated, saturated.copy(), terms, closed)
        moved.append(new[1])
    assert moved[0] > 0.0015              # equal conductivities: the nodes go towards S = 0.5
    assert moved[1] < 1e-2 * moved[0]     # the arithmetic mean would give about the same as the first case


def full_column_solve(lwc, floor, saturated, interface_terms, closed):
    '''The solve over the whole column, with no window and no reset of the tail. Returns the water after the solve.'''
    from scipy.linalg import solve_banded
    span = saturated - floor
    closed = closed | (span <= 0.)
    active = ~closed & (lwc >= floor)
    terms = np.where(closed[:-1] | closed[1:], 0., interface_terms)
    mobile = np.where(active, lwc - floor, 0.)
    banded = np.zeros((3, len(lwc)))
    banded[0, 1:] = -np.where(active[:-1], terms, 0.)
    banded[1] = np.where(active, span + np.concatenate(([0.], terms)) + np.concatenate((terms, [0.])), 1.)
    banded[2, :-1] = -np.where(active[1:], terms, 0.)
    saturation = solve_banded((1, 1), banded, mobile)
    flux = terms * (saturation[:-1] - saturation[1:])
    new = lwc.copy()
    new[:-1] -= flux
    new[1:] += flux
    return new


@pytest.mark.unit
def test_the_window_gives_the_result_of_the_solve_over_the_whole_column():
    '''With sinks and closed nodes around the wet part, the window is exact. Nothing outside it changes.'''
    rng = np.random.default_rng(11)
    for _ in range(40):
        n = 40
        lwc, floor, saturated, capacity, conductivity, dz, closed = random_column(rng, n=n, closed_fraction=0.05)
        # sinks everywhere except two groups of active nodes: the wet groups
        lwc = floor * 0.5
        floor = np.maximum(floor, 1e-5)
        first, second = int(rng.integers(3, 12)), int(rng.integers(22, 34))
        for start in (first, second):
            lwc[start:start + 4] = floor[start:start + 4] + 0.5 * (saturated[start:start + 4] - floor[start:start + 4])
        terms = capillary_interface_terms(conductivity, dz, 1800., 0.1)
        windowed, excess = capillary_implicit_step(lwc, floor, saturated, 10. * saturated, terms, closed)   # no capacity pass
        reference = full_column_solve(lwc, floor, saturated, terms, closed)
        assert excess == 0.
        assert np.allclose(windowed, reference, rtol=1e-12, atol=1e-18)
        assert np.sum(windowed) == pytest.approx(lwc.sum(), rel=1e-13)
        far = np.ones(n, dtype=bool)
        far[max(first - 2, 0):first + 6] = False
        far[second - 2:second + 6] = False
        assert np.array_equal(windowed[far], lwc[far])


@pytest.mark.unit
def test_the_window_does_not_solve_a_column_with_a_wet_group_between_sinks():
    '''One wet group between two sinks: only the nodes of the group and the two sinks change.'''
    n = 12
    floor = np.full(n, 0.0005)
    saturated = floor + 0.004
    lwc = np.full(n, 0.0002)                  # all sinks
    lwc[5:7] = [0.0045, 0.003]               # one wet group of two nodes
    terms = capillary_interface_terms(np.full(n, 1e-4), np.full(n, 0.01), 600., 0.07)
    new, excess = capillary_implicit_step(lwc, floor, saturated, saturated.copy(), terms, np.zeros(n, dtype=bool))
    assert excess == 0.
    assert np.array_equal(new[:4], lwc[:4]) and np.array_equal(new[8:], lwc[8:])
    assert new[4] > lwc[4] and new[7] > lwc[7]
    assert new.sum() == pytest.approx(lwc.sum(), rel=1e-14)


@pytest.mark.unit
def test_the_tail_in_dry_nodes_of_floor_zero_is_set_back_and_the_budget_closes():
    '''
    A wet top and dry nodes of floor zero below: the backward Euler solve puts water in all of them.
    The part below CAPILLARY_WATER_THRESHOLD is set to zero and goes to the deepest node that keeps water.
    '''
    from darcy_implicit import CAPILLARY_WATER_THRESHOLD
    n = 200
    floor = np.zeros(n)
    saturated = np.full(n, 0.0056)
    lwc = np.zeros(n)
    lwc[:10] = 0.0012
    terms = capillary_interface_terms(np.full(n, 1e-4), np.full(n, 0.01), 60., 0.0714)
    full = full_column_solve(lwc, floor, saturated, terms, np.zeros(n, dtype=bool))
    assert full[-1] > 0. and full[-1] < CAPILLARY_WATER_THRESHOLD   # the solve itself puts a tail down to the bottom
    new, excess = capillary_implicit_step(lwc, floor, saturated, saturated.copy(), terms, np.zeros(n, dtype=bool))
    assert excess == 0.
    assert new.sum() == pytest.approx(lwc.sum(), rel=1e-14)
    wet = np.flatnonzero(new > 0.)
    assert new[wet].min() >= CAPILLARY_WATER_THRESHOLD or wet.max() < n - 1
    assert np.count_nonzero(new) < np.count_nonzero(full)
    assert np.all(new[wet[-1] + 1:] == 0.)


@pytest.mark.unit
def test_water_above_the_capacity_does_not_cross_a_closed_node_upward():
    '''
    Node 1 is a closed lens with no water. Node 2 is a sink with a floor near saturation, and node 3 is
    saturated. The sink fills above its capacity. The excess goes back down in the stretch of nodes 2 and 3, and
    does not jump the lens to node 0.
    '''
    dz = np.full(4, 0.01)
    saturated = dz * 0.5
    saturated[1] = 1e-9 * 0.01
    floor = np.array([0., 0., 0.0045, 0.])
    lwc = np.array([0., 0., 0.0040, 0.005])
    capacity = saturated.copy()
    capacity[1] = lwc[1]
    closed = np.array([False, True, False, False])
    terms = capillary_interface_terms(np.full(4, 1e-4), dz, 1800., 0.0714)
    new, excess = capillary_implicit_step(lwc, floor, saturated, capacity, terms, closed)
    assert new[0] == 0.
    assert new[1] == 0.
    assert excess == 0.
    assert new.sum() == pytest.approx(lwc.sum(), rel=1e-13)
    assert np.all(new[2:] <= saturated[2:] + 1e-15)


@pytest.mark.unit
def test_step_conserves_the_water_with_the_surface_excess():
    rng = np.random.default_rng(3)
    for _ in range(60):
        column = random_column(rng, closed_fraction=0.1, sink_fraction=0.2)
        lwc, floor, saturated, capacity, conductivity, dz, closed = column
        new, excess = step(column, dtsub=3600., head=0.3)
        assert new.sum() + excess == pytest.approx(lwc.sum(), rel=1e-12)
        assert excess >= 0.
        assert np.all(new <= np.maximum(capacity, lwc) + 1e-15)
        assert np.all(new >= -1e-15)


@pytest.mark.unit
def test_water_above_the_capacity_of_a_sink_goes_up_to_the_surface_excess():
    '''
    The top node is a sink that is almost at its floor, and the floor is almost at saturation. The full
    node below gives it more than it can take, and the water backs up to the surface.
    '''
    saturated = np.full(2, 0.004)
    floor = np.array([0.0039, 0.])
    lwc = np.array([0.0038, 0.004])
    terms = capillary_interface_terms(np.full(2, 1e-3), np.full(2, 0.01), 3600., 0.3)
    new, excess = capillary_implicit_step(lwc, floor, saturated, saturated.copy(), terms, np.zeros(2, dtype=bool))
    assert new[0] == pytest.approx(0.004)
    assert excess > 0.
    assert new.sum() + excess == pytest.approx(lwc.sum(), rel=1e-14)


@pytest.mark.unit
def test_no_water_passes_a_closed_node():
    '''The part above and the part below a closed node evolve as two separate columns.'''
    rng = np.random.default_rng(4)
    for _ in range(30):
        n, j = 24, int(rng.integers(5, 19))
        column = random_column(rng, n=n)
        lwc, floor, saturated, capacity, conductivity, dz, closed = column
        closed[j] = True
        new, excess = step(column, dtsub=3600., head=0.3)
        assert new[j] == lwc[j]
        assert new[:j].sum() + excess == pytest.approx(lwc[:j].sum(), rel=1e-12)
        assert new[j + 1:].sum() == pytest.approx(lwc[j + 1:].sum(), rel=1e-12)
        upper = tuple(a[:j] for a in column[:6]) + (closed[:j],)
        lower = tuple(a[j + 1:] for a in column[:6]) + (closed[j + 1:],)
        assert np.allclose(new[:j], step(upper, dtsub=3600., head=0.3)[0], rtol=1e-12, atol=1e-18)
        assert np.allclose(new[j + 1:], step(lower, dtsub=3600., head=0.3)[0], rtol=1e-12, atol=1e-18)


@pytest.mark.unit
def test_a_zero_conductivity_closes_the_interface_like_a_closed_node():
    rng = np.random.default_rng(5)
    lwc, floor, saturated, capacity, conductivity, dz, closed = random_column(rng, n=12)
    conductivity[6] = 0.
    new, _ = step((lwc, floor, saturated, capacity, conductivity, dz, closed))
    assert new[:6].sum() == pytest.approx(lwc[:6].sum(), rel=1e-12)   # the node 6 and the node 5 exchange nothing
    assert new[6:].sum() == pytest.approx(lwc[6:].sum(), rel=1e-12)


@pytest.mark.unit
def test_nodes_at_the_same_saturation_exchange_no_water():
    '''Also with different thickness, porosity and floor.'''
    rng = np.random.default_rng(6)
    _, floor, saturated, capacity, conductivity, dz, closed = random_column(rng, n=20)
    lwc = floor + 0.37 * (saturated - floor)
    new, excess = step((lwc, floor, saturated, capacity, conductivity, dz, closed), dtsub=3600., head=0.5)
    assert np.allclose(new, lwc, rtol=1e-10, atol=0.)
    assert excess == 0.


@pytest.mark.unit
def test_an_active_node_stays_between_the_values_of_its_neighbours():
    '''
    The new saturation of a node is a weighted mean of its old value and the new values of its
    two neighbours. So it stays between them. All the new values stay between the extremes of the old ones.
    '''
    rng = np.random.default_rng(7)
    for _ in range(40):
        lwc, floor, saturated, capacity, conductivity, dz, closed = random_column(rng, n=25)
        span = saturated - floor
        old = (lwc - floor) / span
        new_lwc, excess = step((lwc, floor, saturated, capacity, conductivity, dz, closed), dtsub=1800., head=0.1)
        assert excess == 0.
        new = (new_lwc - floor) / span
        assert new.min() >= old.min() - 1e-12 and new.max() <= old.max() + 1e-12
        inner = np.arange(1, 24)
        lowest = np.minimum(old[inner], np.minimum(new[inner - 1], new[inner + 1]))
        highest = np.maximum(old[inner], np.maximum(new[inner - 1], new[inner + 1]))
        assert np.all(new[inner] >= lowest - 1e-12) and np.all(new[inner] <= highest + 1e-12)


@pytest.mark.unit
def test_a_closed_column_goes_to_one_uniform_saturation():
    '''A step with a very large coefficient: the saturation is the mean, weighted by the span. A uniform column stays as it is.'''
    rng = np.random.default_rng(8)
    lwc, floor, saturated, capacity, conductivity, dz, closed = random_column(rng, n=20)
    span = saturated - floor
    conductivity[:] = 1e-3
    new, excess = step((lwc, floor, saturated, capacity, conductivity, dz, closed), dtsub=1e7, head=1.)
    assert excess == 0.
    uniform = (lwc - floor).sum() / span.sum()
    assert np.allclose((new - floor) / span, uniform, rtol=1e-5)
    # at each sub-step the steady state is exact
    for dtsub in (1., 600., 3600., 1e6):
        steady = floor + 0.4 * span
        again, _ = step((steady, floor, saturated, capacity, conductivity, dz, closed), dtsub=dtsub)
        assert np.allclose(again, steady, rtol=1e-9, atol=0.)   # the largest error is 2.6e-11, at 1e6 s


@pytest.mark.unit
@pytest.mark.parametrize('mode', [1, 2, 5])
@pytest.mark.parametrize('dtsub', [60., 1800.])
def test_a_cosine_mode_decays_by_the_backward_euler_factor(mode, dtsub):
    '''A uniform column with closed ends: S of a cosine mode is multiplied by 1/(1 + dtsub D lambda),
    with D = K h_c / (span / dz) and lambda the eigenvalue of the discrete Laplacian.'''
    n, dz, conductivity, head = 40, 0.01, 5e-5, 0.07
    span = np.full(n, 0.004)
    floor = np.full(n, 0.0005)
    saturated = floor + span
    amplitude = 0.3
    shape = np.cos(mode * np.pi * (np.arange(n) + 0.5) / n)
    lwc = floor + span * (0.5 + amplitude * shape)
    new, _ = capillary_implicit_step(lwc, floor, saturated, saturated.copy(),
                                     capillary_interface_terms(np.full(n, conductivity), np.full(n, dz), dtsub, head),
                                     np.zeros(n, dtype=bool))
    eigenvalue = 2. - 2. * np.cos(mode * np.pi / n)   # of the discrete Laplacian with no flux at the ends
    diffusivity = conductivity * head / (span[0] / dz)   # K h_c over the water content at S = 1 [m2 s-1]
    factor = 1. / (1. + dtsub * diffusivity / dz ** 2 * eigenvalue)
    assert (new - floor) / span - 0.5 == pytest.approx(amplitude * factor * shape, abs=1e-12)


@pytest.mark.unit
def test_a_sink_node_pays_its_floor_first():
    '''
    A node below its floor is held at S = 0. It gets G S of its active neighbour, all of it as water that
    pays its floor: its new water is its old water plus that amount, and it is not an unknown of the solve.
    '''
    n = 3
    floor = np.array([0.0005, 0.002, 0.0005])
    saturated = floor + 0.004
    lwc = np.array([saturated[0], 0.0015, 0.0005])   # full node, sink (below its floor), node at its floor
    conductivity, dz = np.full(n, 1e-5), np.full(n, 0.01)
    terms = capillary_interface_terms(conductivity, dz, 1., 0.07)   # 7e-5 m, below the deficit of the sink, 5e-4 m
    new, excess = capillary_implicit_step(lwc, floor, saturated, saturated.copy(), terms, np.zeros(n, dtype=bool))
    assert excess == 0.
    # the sink 1 holds S = 0, so the node 0 gives it terms[0] S_0 and the node 2 (S = 0, active) gives or takes nothing
    # of its own: it is next to the sink, which is at S = 0
    saturation_0 = (new[0] - floor[0]) / 0.004
    assert new[1] == pytest.approx(0.0015 + terms[0] * saturation_0, rel=1e-12)
    assert new[2] == 0.0005
    assert new[1] < floor[1]   # the water of this step did not take the node above its floor
    assert new.sum() == pytest.approx(lwc.sum(), rel=1e-14)


@pytest.mark.unit
def test_a_cold_node_and_a_node_below_its_irreducible_water_are_sinks_and_a_dry_node_is_not():
    '''
    Three drier neighbours of one wet node, with the same water below the wet node: a dry node with a
    floor of zero is active, so its saturation rises and it takes less than a sink at the same water.
    '''
    saturated = np.full(2, 0.004)
    conductivity, dz = np.full(2, 1e-4), np.full(2, 0.01)
    terms = capillary_interface_terms(conductivity, dz, 600., 0.07)
    no_closed = np.zeros(2, dtype=bool)

    def received(floor_below, water_below):
        lwc = np.array([0.004, water_below])
        floor = np.array([0., floor_below])
        new, _ = capillary_implicit_step(lwc, floor, saturated, saturated.copy(), terms, no_closed)
        return new[1] - water_below

    active = received(0., 0.)        # a dry node with a floor of zero
    cold = received(0.001, 0.)       # a cold node: the floor is the water that its cold content freezes
    irreducible = received(0.0005, 0.0003)   # a node below its irreducible water
    assert active > 0.
    assert cold > active
    assert irreducible > 0.
    # a sink takes G S_0 for the new S_0, and its new water is its old water plus that, as in the test above
    assert cold == pytest.approx(terms[0] * (0.004 - cold) / 0.004, rel=1e-10)


### The scheme: the oracle ###

DENSITY = 400.
POROSITY = 1. - DENSITY / RHO_I
THETA_S = POROSITY - 1e-6          # theta_s_factor_darcy 1
K0 = 5.56e-11
VISCOSITY = 1e-3
CONDUCTIVITY = float(hydrconducsat_power_law(POROSITY, K0, VISCOSITY))   # K_sat of the run [m s-1]
PAPER = {'ColeouLesaffre': False, 'IrrVal': 0., 'theta_s_factor_darcy': 1.0, 'Ksat_darcy': 'power_law',
         'Ksat_k0_darcy': K0, 'viscosity_darcy': VISCOSITY, 'krel_darcy': 'power_law', 'krel_exponent_darcy': 2}
FLUX = 5e-6          # [m s-1]
PRESSURE = 200.      # [Pa]
HOURS = 12
DEPTH = 3.0          # [m]


class Wave:
    '''
    The travelling wave. Water balance of the wave: -v theta_s S + K S^2 - K h_c S' = 0, with S = 0 ahead.
    Behind the front, K S_+^2 = flux and v = flux / (theta_s S_+). Then S' = S (S - S_+) / h_c, so
    S = S_+ / (1 + exp((z - z_front) / w)) with w = h_c / S_+. The mass of the water puts the half
    height at z_front = v t (the profile has formed, and z_front is far above w).
    '''

    def __init__(self, flux, pressure, duration):
        self.plateau = np.sqrt(flux / CONDUCTIVITY)
        self.head = pressure / (RHO_W_KGM * GRAVITY)
        self.width = self.head / self.plateau
        self.speed = flux / (THETA_S * self.plateau)
        self.front = self.speed * duration

    def saturation(self, depth):
        return self.plateau / (1. + np.exp((depth - self.front) / self.width))


def crossing(depth, saturation, level):
    '''The depth where the saturation falls through `level`, between node centres.'''
    below = np.where(saturation > level)[0][-1]
    return depth[below] + (saturation[below] - level) / (saturation[below] - saturation[below + 1]) \
        * (depth[below + 1] - depth[below])


def run_wave(dz, dtsub, flux=FLUX, pressure=PRESSURE, hours=HOURS, depth=DEPTH, **config):
    '''Rain at the surface of dry uniform firn at 0 C, one call of darcyscheme for each hour.'''
    calls = int(hours)
    dt = hours * 3600. / calls
    config = {**PAPER, **IMPLICIT, **CAPILLARY, 'dtsub_implicit_darcy': dtsub, 'capillary_pressure_darcy': pressure,
              **config}
    column = SyntheticColumn(n=int(round(depth / dz)), rho_profile=DENSITY, T_profile=273.15, dz=dz, r2=1e-6,
                             rain=flux * dt / RHO_I_MGM, dt=dt, steps=calls, config=config)
    for call in range(calls):
        melt.darcyscheme(column, 0)
    centre = np.cumsum(column.dz) - column.dz / 2
    return column, centre, column.LWC / column.dz / THETA_S


def compare_with_the_wave(dz, dtsub, **kwargs):
    '''Relative error of the front depth, the width ratio, the plateau error and the largest profile error (over S_+).'''
    column, centre, saturation = run_wave(dz, dtsub, **kwargs)
    wave = Wave(kwargs.get('flux', FLUX), kwargs.get('pressure', PRESSURE), kwargs.get('hours', HOURS) * 3600.)
    front = crossing(centre, saturation, 0.5 * wave.plateau) / wave.front - 1.
    width = (crossing(centre, saturation, 0.1 * wave.plateau) - crossing(centre, saturation, 0.9 * wave.plateau)) \
        / (2. * np.log(9.) * wave.width)
    plateau = np.mean(saturation[2:len(saturation) // 4]) / wave.plateau - 1.
    profile = np.max(np.abs(saturation - wave.saturation(centre))) / wave.plateau
    return front, width, plateau, profile, column


# dz [m], sub-step [s], then the tolerance of the front depth, of the width ratio minus 1, of the plateau and of
# the profile. Measured (front, width - 1, plateau, profile), 12 h of rain at 5e-6 m/s, c = 200 Pa (S_+ 0.226, w 0.090 m):
#   4 mm,  60 s:  0.0015, 0.040, -7e-6, 0.0104
#   20 mm, 60 s:  0.0040, 0.128, -2e-5, 0.0324
#   20 mm, 600 s: 0.0097, 0.295, -1.6e-4, 0.0696
#   20 mm, 1800 s: 0.0183, 0.626, -1.1e-3, 0.1323
#   0.1 m, 60 s:  0.0128, 0.582, -5.6e-4, 0.1164
# The width is wider than the wave because the gravity sweep (upwind, backward Euler) adds numerical diffusion.
# Without the capillary step the width ratio is 0.14 to 0.65, so the check 1 <= width ratio < 1 + tolerance sees the
# step: every measured width with the step on is above the exact width.
WAVE_CASES = [
    (0.004, 60., 0.003, 0.08, 1e-4, 0.015),
    (0.02, 60., 0.006, 0.17, 1e-4, 0.045),
    (0.02, 600., 0.013, 0.35, 5e-4, 0.09),
    (0.02, 1800., 0.025, 0.75, 2e-3, 0.17),
    (0.1, 60., 0.017, 0.65, 1.5e-3, 0.15),
]


@pytest.mark.scheme
@pytest.mark.parametrize('dz, dtsub, front_tolerance, width_tolerance, plateau_tolerance, profile_tolerance', WAVE_CASES)
def test_front_and_width_match_the_travelling_wave(dz, dtsub, front_tolerance, width_tolerance, plateau_tolerance,
                                                   profile_tolerance):
    front, width, plateau, profile, column = compare_with_the_wave(dz, dtsub)
    assert abs(front) < front_tolerance
    assert 0. <= width - 1. < width_tolerance
    assert abs(plateau) < plateau_tolerance
    assert profile < profile_tolerance
    assert column.LWC.sum() == pytest.approx(FLUX * HOURS * 3600., rel=1e-10)   # the budget closes
    assert column.melt_diagnostics['runoff'] < 1e-12   # roundoff of the sweep, 5e-14 at most


@pytest.mark.scheme
def test_the_width_of_the_wave_goes_with_the_capillary_pressure():
    '''c of 100 Pa and 400 Pa: w is proportional to c. Measured width ratios (width / exact width): 1.13 and 1.04.'''
    ratios = []
    for pressure in (100., 400.):
        _, width, _, _, _ = compare_with_the_wave(0.01, 60., pressure=pressure)
        ratios.append(width)
    assert ratios[0] == pytest.approx(1., abs=0.3)
    assert ratios[1] == pytest.approx(1., abs=0.15)


@pytest.mark.scheme
def test_the_capillary_step_spreads_the_water_with_calonne():
    '''
    With Ksat_darcy "Calonne", K_sat is far above that of the power law, and the capillary flux goes with
    K_sat. The comparison is with and without the capillary step: gravity alone already spreads the water.
    '''
    wet = {}
    for name, extra in (('gravity', {}), ('capillary', CAPILLARY)):
        column = SyntheticColumn(n=500, rho_profile=DENSITY, T_profile=273.15, dz=0.02, r2=(1e-3) ** 2,
                                 rain=FLUX * 3600. / RHO_I_MGM, dt=3600., steps=1,
                                 config={**PAPER, 'Ksat_darcy': 'Calonne', **IMPLICIT, **extra,
                                         'dtsub_implicit_darcy': 60., 'capillary_pressure_darcy': PRESSURE})
        melt.darcyscheme(column, 0)
        saturation = column.LWC / column.dz / THETA_S
        wet[name] = np.count_nonzero(saturation > 1e-3 * saturation.max())   # measured: 295 nodes with gravity, 498 with the step
        assert column.LWC.sum() + column.melt_diagnostics['runoff'] == pytest.approx(FLUX * 3600., rel=1e-10)
    assert wet['capillary'] > wet['gravity']


### The scheme: the budget and the limits ###

@pytest.mark.scheme
@pytest.mark.parametrize('case', EDGE_CASES)
def test_edge_cases_keep_the_budget_with_the_capillary_step(case):
    '''With lenses, cold firn, a saturated column and no input. With melt_strict a failed check raises.'''
    column = edge_column(case, 1800.)
    column.c.update(CAPILLARY)
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
def test_the_capillary_step_changes_a_column_with_a_gradient_of_saturation():
    '''The step is not a no-operation: the water of an edge column differs with and without it.'''
    results = []
    for extra in ({}, CAPILLARY):
        column = edge_column('zero_input', 1800.)
        column.c.update(extra)
        melt.darcyscheme(column, 0)
        results.append(column.LWC.copy())
    assert not np.array_equal(results[0], results[1])


@pytest.mark.scheme
def test_no_water_passes_an_impermeable_lens_with_the_capillary_step():
    flux, days = 2e-7, 3
    column, below = lens_column(0.01, 0.2, thick_imp=0.1, days=days, **CAPILLARY)
    for day in range(days):
        melt.darcyscheme(column, day)
    assert column.LWC[below].sum() == 0.
    assert column.melt_diagnostics['runoff'] == pytest.approx(flux * 86400., rel=0.05)


### The keys ###

@pytest.mark.scheme
def test_the_key_at_none_gives_the_arrays_of_no_key():
    '''Bit for bit. The step with the key at "power_law" gives other arrays.'''
    results = {}
    for name, value in (('no key', None), ('none', 'none'), ('power_law', 'power_law')):
        column = edge_column('base', 1800.)
        column.c.pop('capillary_implicit_darcy', None)
        if value:
            column.c['capillary_implicit_darcy'] = value
        for day in range(2):
            melt.darcyscheme(column, day)
        results[name] = column
    for name in ('LWC', 'Tz', 'rho', 'dz'):
        assert np.array_equal(getattr(results['no key'], name), getattr(results['none'], name))
    assert not np.array_equal(results['no key'].LWC, results['power_law'].LWC)


@pytest.mark.scheme
def test_the_capillary_pressure_is_not_read_without_the_step(capsys):
    melt._darcy_missing_reported.clear()
    column = edge_column('base', 1800.)
    column.c['capillary_pressure_darcy'] = -1.   # not read: the step is off
    melt.darcyscheme(column, 0)
    assert 'capillary_pressure_darcy' not in capsys.readouterr().out


@pytest.mark.scheme
def test_the_key_has_no_effect_with_the_explicit_step_and_a_message_is_printed_once(capsys):
    results = []
    for value in ('none', 'power_law'):
        column = edge_column('base', 1800.)
        column.c.update({'flow_step_darcy': 'explicit', 'capillary_implicit_darcy': value})
        capsys.readouterr()
        melt.darcyscheme(column, 0)
        melt.darcyscheme(column, 1)
        results.append((column, capsys.readouterr().out))
    assert np.array_equal(results[0][0].LWC, results[1][0].LWC)
    assert np.array_equal(results[0][0].Tz, results[1][0].Tz)
    assert 'capillary_implicit_darcy' not in results[0][1]
    assert results[1][1].count('capillary_implicit_darcy is "power_law" with flow_step_darcy "explicit".') == 1
    assert 'The key has no effect with the explicit flow step.' in results[1][1]


@pytest.mark.scheme
@pytest.mark.parametrize('laws, message', [
    ({'krel_darcy': 'vanGenuchten'}, True),
    ({'krel_darcy': 'power_law', 'krel_exponent_darcy': 3.}, True),
    ({'krel_darcy': 'power_law', 'krel_exponent_darcy': 2.}, False),
])
def test_the_message_for_other_krel_laws_is_printed_once(capsys, laws, message):
    column = edge_column('base', 1800.)
    column.c.update({**CAPILLARY, **laws})
    capsys.readouterr()
    for day in range(2):
        melt.darcyscheme(column, day)
    text = capsys.readouterr().out
    assert text.count('The capillary flow uses k_r = S^2') == (1 if message else 0)
    assert 'has no effect' not in text


@pytest.mark.scheme
@pytest.mark.parametrize('value', ['yes', 'van_genuchten', True])
def test_a_wrong_value_of_the_key_stops_the_run(value):
    column = edge_column('base', 1800.)
    column.c['capillary_implicit_darcy'] = value
    with pytest.raises(ValueError, match='capillary_implicit_darcy'):
        melt.darcyscheme(column, 0)


@pytest.mark.scheme
@pytest.mark.parametrize('pressure', [0., -700.])
def test_a_pressure_of_zero_or_below_stops_the_run(pressure):
    column = edge_column('base', 1800.)
    column.c.update({**CAPILLARY, 'capillary_pressure_darcy': pressure})
    with pytest.raises(ValueError, match='capillary_pressure_darcy'):
        melt.darcyscheme(column, 0)


@pytest.mark.unit
def test_the_defaults_of_the_keys():
    assert melt._DARCY_DEFAULTS['capillary_implicit_darcy'] == 'none'
    assert melt._DARCY_DEFAULTS['capillary_pressure_darcy'] == 700.


@pytest.mark.scheme
def test_the_pressure_key_sets_the_capillary_head():
    '''A larger c gives a wider wave in the same column: the key reaches the step.'''
    widths = []
    for pressure in (100., 400.):
        _, centre, saturation = run_wave(0.01, 600., pressure=pressure, hours=6., depth=1.5)
        widths.append(crossing(centre, saturation, 0.1 * saturation[3]) - crossing(centre, saturation, 0.9 * saturation[3]))
    assert widths[1] > 2. * widths[0]
