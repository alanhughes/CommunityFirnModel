'''
Tests for the densification laws HLdynamic and KuipersMunneke2015 in
physics.py against the equations of Stevens et al. (2020, Geosci. Model Dev.
13, 4355-4377).

The reference values are calculated in this file from the published equations,
with literal constants. They do not use constants.py.

    Herron and Langway (1980), Stevens et al. Eqs. 4-6, rho in Mg m-3, b in m w.e. a-1:
        drho/dt = c (rho_i - rho)
        c0 = 11 exp(-10160 / RT) b,  c1 = 575 exp(-21400 / RT) b^0.5
    Arthern et al. (2010) steady state, Stevens et al. Eq. 18, rho in kg m-3, b in kg m-2 a-1:
        c0 = 0.07 b g exp(-Ec / RT + Eg / RTm),  c1 = 0.03 b g exp(-Ec / RT + Eg / RTm)
        Ec = 60 kJ mol-1, Eg = 42.4 kJ mol-1
    Kuipers Munneke et al. (2015), Stevens et al. Eq. 20:
        M0 = 1.042 - 0.0916 ln(b),  M1 = 1.734 - 0.2039 ln(b),  c0(KM) = M0 c0,  c1(KM) = M1 c1

Choices in the code that the papers do not give:

- Zone 1 is rho < 550 kg m-3 and zone 2 is rho >= 550 kg m-3. Eqs. 6 and 18 put
  rho = 550 kg m-3 in zone 1. The rate is not continuous at 550 kg m-3.
- g = 9.8 m s-2 and one year is 365.25 days.
- KuipersMunneke2015 limits M0 and M1 to at least 0.25. Eq. 20 has no limit.
- KuipersMunneke2015 uses the mean annual temperature Tm in the grain-growth
  term, after Steger et al. (2017). Kuipers Munneke et al. (2015) used T.

The accumulation input of the tests is in m ice eq. a-1, as in the model
(bdotSec and bdot_mean). The model converts it with rho_i = 917 kg m-3.
'''

import numpy as np
import pytest

from physics import FirnPhysics

R_GAS = 8.314  # J mol-1 K-1
RHO_ICE = 917.0  # kg m-3
GRAVITY = 9.8  # m s-2
SECONDS_PER_YEAR = 31557600.0  # 365.25 d
STEPS_PER_YEAR = 12
ITERATION = 1  # time-step index iii. The time series have a different value at each index.

TEMPERATURES = [243.15, 263.15]  # K
ACCUMULATIONS = [0.1, 0.5]  # m ice eq. a-1
TEMPERATURE_OFFSETS = np.array([0.0, -1.5, 1.0, -0.5, 2.0])  # K, so that each node has its own T
ZONE_DENSITIES = np.array([320.0, 420.0, 540.0, 560.0, 760.0])  # kg m-3, zone 1 and zone 2


def _drho_dt(law, rho, Tz, bdot_type='mean', bdot_instant=0.3, bdot_mean=None,
             T_mean=250.0, sigma=None, dz=None, LWC=None):
    '''
    drho_dt [kg m-3 s-1] from FirnPhysics(PhysParams).<law>() for a column of
    len(rho) nodes. The keys of PhysParams are the keys that FirnDensityNoSpin
    gives.

    bdot_instant [m ice eq. a-1] is the accumulation rate of step ITERATION.
    bdot_mean [m ice eq. a-1] is the accumulation rate of each node. T_mean [K]
    is the mean annual temperature of step ITERATION.
    '''
    rho = np.asarray(rho, dtype=float)
    n = rho.size
    Tz = np.broadcast_to(np.asarray(Tz, dtype=float), (n,)).copy()
    dz = np.full(n, 0.1) if dz is None else np.asarray(dz, dtype=float)
    mass = rho * dz
    if sigma is None:
        sigma = np.cumsum(mass) * GRAVITY
    if bdot_mean is None:
        bdot_mean = bdot_instant
    bdot_mean = np.broadcast_to(np.asarray(bdot_mean, dtype=float), (n,)).copy()
    LWC = np.zeros(n) if LWC is None else np.asarray(LWC, dtype=float)
    # Wrong values at the other time indices, so that a wrong index changes the result.
    bdotSec = np.array([0.5, 1.0, 2.0]) * bdot_instant / SECONDS_PER_YEAR / STEPS_PER_YEAR
    T_series = np.array([-7.0, 0.0, 7.0]) + T_mean
    PhysParams = {
        'iii':           ITERATION,
        'steps':         STEPS_PER_YEAR,
        'gridLen':       n,
        'bdotSec':       bdotSec,
        'bdot_mean':     bdot_mean,
        'bdot_av':       bdot_mean,
        'bdot_type':     bdot_type,
        'Tz':            Tz,
        'T_mean':        T_series,
        'T10m':          T_mean,
        'T50':           T_mean,
        'rho':           rho,
        'mass':          mass,
        'sigma':         np.asarray(sigma, dtype=float),
        'dt':            SECONDS_PER_YEAR / STEPS_PER_YEAR,
        'Ts':            T_series,
        'r2':            np.full(n, 1.0e-8),
        'age':           np.arange(n) * SECONDS_PER_YEAR,
        'physGrain':     False,
        'calcGrainSize': False,
        'r2s0':          1.0e-8,
        'GrGrowPhysics': 'Arthern',
        'z':             np.cumsum(dz) - dz / 2,
        'rhos0':         rho[0],
        'dz':            dz,
        'LWC':           LWC,
        'MELT':          bool(np.any(LWC > 0)),
        'FirnAir':       False,
    }
    return getattr(FirnPhysics(PhysParams), law)()['drho_dt']


# ---------------------------------------------------------------------------
# Reference equations. rho [kg m-3], T [K], b_ice [m ice eq. a-1]. They return kg m-3 s-1.


def _herron_langway(rho, T, b_ice, zone_1_limit=550.0):
    b = b_ice * 0.917  # m w.e. a-1
    c = np.where(rho < zone_1_limit,
                 11.0 * np.exp(-10160.0 / (R_GAS * T)) * b,
                 575.0 * np.exp(-21400.0 / (R_GAS * T)) * np.sqrt(b))
    return c * (0.917 - rho / 1000.0) * 1000.0 / SECONDS_PER_YEAR


def _km_multipliers(b):
    '''M0 and M1 of Eq. 20 for b [kg m-2 a-1], before the limit of 0.25.'''
    return 1.042 - 0.0916 * np.log(b), 1.734 - 0.2039 * np.log(b)


def _kuipers_munneke(rho, T, T_mean, b_ice, zone_1_limit=550.0, b=None):
    '''Eqs. 18 and 20. b [kg m-2 a-1] replaces b_ice when it is given.'''
    if b is None:
        b = b_ice * RHO_ICE
    M0, M1 = (np.maximum(0.25, M) for M in _km_multipliers(b))
    arrhenius = np.exp(-60.0e3 / (R_GAS * T) + 42.4e3 / (R_GAS * T_mean))
    c = np.where(rho < zone_1_limit, M0 * 0.07, M1 * 0.03) * b * GRAVITY * arrhenius
    return c * (RHO_ICE - rho) / SECONDS_PER_YEAR


def _accumulation_inputs(bdot_type, b_ice):
    '''
    The accumulation inputs of _drho_dt and the rate of each node that the law
    must use. The input that bdot_type does not select has a different value,
    so that the wrong input changes the result.
    '''
    per_node = b_ice * np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    if bdot_type == 'instant':
        return dict(bdot_instant=b_ice, bdot_mean=3.0 * per_node), np.full(5, b_ice)
    return dict(bdot_instant=3.0 * b_ice, bdot_mean=per_node), per_node


# ---------------------------------------------------------------------------
# HLdynamic


@pytest.mark.unit
@pytest.mark.parametrize('bdot_type', ['instant', 'mean'])
@pytest.mark.parametrize('T', TEMPERATURES)
@pytest.mark.parametrize('b_ice', ACCUMULATIONS)
def test_HLdynamic_matches_herron_langway(bdot_type, T, b_ice):
    Tz = T + TEMPERATURE_OFFSETS
    inputs, b_node = _accumulation_inputs(bdot_type, b_ice)
    got = _drho_dt('HLdynamic', ZONE_DENSITIES, Tz, bdot_type, **inputs)
    expected = _herron_langway(ZONE_DENSITIES, Tz, b_node)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('law', ['HLdynamic', 'KuipersMunneke2015'])
def test_a_node_at_550_uses_the_zone_2_equation(law):
    '''Eqs. 6 and 18 put rho = 550 kg m-3 in zone 1. The code puts it in zone 2.'''
    rho = np.array([549.0, 550.0, 551.0])
    T, T_mean, b_ice = 253.15, 250.0, 0.3
    got = _drho_dt(law, rho, T, 'mean', bdot_mean=b_ice, T_mean=T_mean)
    if law == 'HLdynamic':
        reference = lambda limit: _herron_langway(rho, T, b_ice, zone_1_limit=limit)
    else:
        reference = lambda limit: _kuipers_munneke(rho, T, T_mean, b_ice, zone_1_limit=limit)
    np.testing.assert_allclose(got, reference(550.0), rtol=1e-12, atol=0)
    paper = reference(np.nextafter(550.0, np.inf))  # rho <= 550 is zone 1
    assert got[1] != pytest.approx(paper[1], rel=1e-2)


# ---------------------------------------------------------------------------
# KuipersMunneke2015


@pytest.mark.unit
@pytest.mark.parametrize('bdot_type', ['instant', 'mean'])
@pytest.mark.parametrize('T', TEMPERATURES)
@pytest.mark.parametrize('b_ice', ACCUMULATIONS)
def test_KuipersMunneke2015_matches_arthern_with_km_multipliers(bdot_type, T, b_ice):
    Tz = T + TEMPERATURE_OFFSETS
    T_mean = T - 4.0  # Tm differs from each Tz, so a swap of T and Tm changes the result
    inputs, b_node = _accumulation_inputs(bdot_type, b_ice)
    got = _drho_dt('KuipersMunneke2015', ZONE_DENSITIES, Tz, bdot_type, T_mean=T_mean, **inputs)
    expected = _kuipers_munneke(ZONE_DENSITIES, Tz, T_mean, b_node)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('bdot_type', ['instant', 'mean'])
@pytest.mark.parametrize('b_ice, floored', [(2.0, 'M1'), (8.0, 'M0 and M1')])
def test_KuipersMunneke2015_multipliers_are_at_least_0_25(bdot_type, b_ice, floored):
    '''
    At 2 m ice eq. a-1 (1834 kg m-2 a-1), M1 of Eq. 20 is less than 0.25. At
    8 m ice eq. a-1, M0 and M1 are both less than 0.25.
    '''
    M0, M1 = _km_multipliers(b_ice * RHO_ICE)
    assert M1 < 0.25 and (M0 < 0.25) == (floored == 'M0 and M1')
    T = 253.15 + TEMPERATURE_OFFSETS
    if bdot_type == 'instant':
        inputs = dict(bdot_instant=b_ice, bdot_mean=0.1)
    else:
        inputs = dict(bdot_instant=0.1, bdot_mean=b_ice)
    got = _drho_dt('KuipersMunneke2015', ZONE_DENSITIES, T, bdot_type, T_mean=250.0, **inputs)
    expected = _kuipers_munneke(ZONE_DENSITIES, T, 250.0, b_ice)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
    zone_2 = ZONE_DENSITIES >= 550.0
    np.testing.assert_allclose(  # the limit holds with the multiplier 0.25
        got[zone_2],
        0.25 * 0.03 * b_ice * RHO_ICE * GRAVITY * (RHO_ICE - ZONE_DENSITIES[zone_2]) / SECONDS_PER_YEAR
        * np.exp(-60.0e3 / (R_GAS * T[zone_2]) + 42.4e3 / (R_GAS * 250.0)),
        rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('bdot_type', ['instant', 'mean'])
def test_KuipersMunneke2015_grain_growth_term_uses_the_mean_annual_temperature(bdot_type):
    '''
    Steger et al. (2017): Eg is divided by R Tm, not by R T. A change of Tm
    alone multiplies the rate by exp(Eg/R (1/Tm2 - 1/Tm1)). A change of T alone
    multiplies it by exp(-Ec/R (1/T2 - 1/T1)).
    '''
    T, b_ice = 253.15 + TEMPERATURE_OFFSETS, 0.3

    def rate(Tz, T_mean):
        return _drho_dt('KuipersMunneke2015', ZONE_DENSITIES, Tz, bdot_type,
                        bdot_instant=b_ice, bdot_mean=b_ice, T_mean=T_mean)

    np.testing.assert_allclose(rate(T, 255.0) / rate(T, 245.0),
                               np.exp(42.4e3 / R_GAS * (1 / 255.0 - 1 / 245.0)), rtol=1e-12)
    np.testing.assert_allclose(rate(T + 10.0, 250.0) / rate(T, 250.0),
                               np.exp(-60.0e3 / R_GAS * (1 / (T + 10.0) - 1 / T)), rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('T', TEMPERATURES)
@pytest.mark.parametrize('dz', [0.05, 0.2])
def test_KuipersMunneke2015_stress_uses_ten_times_the_layer_mass(T, dz):
    '''
    bdot_type "stress" is not a published equation, and docs/running/json.rst
    says that it is in progress. The code uses 10 rho dz [kg m-2] in place of b
    [kg m-2 a-1] in Eqs. 18 and 20. It does not use sigma. This test records
    that behaviour.
    '''
    Tz = T + TEMPERATURE_OFFSETS
    dz = np.full(5, dz)
    got = _drho_dt('KuipersMunneke2015', ZONE_DENSITIES, Tz, 'stress', bdot_mean=0.3,
                   T_mean=250.0, dz=dz, sigma=np.full(5, 1.0e5))
    expected = _kuipers_munneke(ZONE_DENSITIES, Tz, 250.0, None, b=10.0 * ZONE_DENSITIES * dz)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason='KuipersMunneke2015 "stress" uses 10 x the layer mass, not the stress')
def test_KuipersMunneke2015_stress_rate_does_not_depend_on_the_layer_thickness():
    '''
    A densification rate is a property of the firn at a node: rho, T and the
    stress. Two columns with the same rho, T and sigma at each node, but with
    a different node thickness, must have the same rate.
    '''
    T = 253.15 + TEMPERATURE_OFFSETS
    sigma = np.array([1.0e3, 5.0e3, 2.0e4, 5.0e4, 1.0e5])  # Pa
    thin, thick = (_drho_dt('KuipersMunneke2015', ZONE_DENSITIES, T, 'stress', bdot_mean=0.3,
                            T_mean=250.0, dz=np.full(5, dz), sigma=sigma) for dz in (0.05, 0.1))
    np.testing.assert_allclose(thin, thick, rtol=1e-12)
