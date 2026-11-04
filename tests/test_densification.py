'''
Tests for the densification laws in physics.py.

Equation tests: HLdynamic, KuipersMunneke2015 and Crocus against the equations
of Stevens et al. (2020, Geosci. Model Dev. 13, 4355-4377) and Verjans et al.
(2019, The Cryosphere 13, 1819-1842).

Property tests: each law in physics.py that a model run can use, against these
properties, which do not need a reference equation:

- The rate is not negative.
- The rate is finite: for rho from 300 to 917 kg m-3, in a layered column, and
  at the melting point.
- The rate goes to 0 as rho goes to 917 kg m-3.
- The rate increases with the temperature at constant rho, stress and
  accumulation.
- bdot_type "mean" and "instant" give a rate at each node below the ice density,
  or stop the run on purpose.

Where a law breaks a property, the test leaves out that law or that condition.
This file lists each one, with the cause in the code.

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
        M0 and M1 are at least 0.25 (Kuipers Munneke et al., 2015, Sect. 2.1, after
        Ligtenberg et al., 2011). Eq. 20 does not show this limit.
    Crocus (Vionnet et al., 2012), Stevens et al. Eq. 22 and Verjans et al. Eqs. 1-3:
        drho/dt = rho sigma / eta
        eta = f1 f2 eta0 (rho / c_eta) exp(a_eta (T_melt - T) + b_eta rho)
        f1 = 1 / (1 + 60 theta),  f2 = 4,  eta0 = 7.62237e6 kg s-1 m-1,
        a_eta = 0.1 K-1,  b_eta = 0.023 m3 kg-1,  c_eta = 358 kg m-3,  T_melt = 273.15 K

Choices in the code that the papers do not give:

- Zone 1 is rho < 550 kg m-3 and zone 2 is rho >= 550 kg m-3. Eqs. 6 and 18 put
  rho = 550 kg m-3 in zone 1. The rate is not continuous at 550 kg m-3.
- g = 9.8 m s-2 and one year is 365.25 days.
- KuipersMunneke2015 uses the mean annual temperature Tm in the grain-growth
  term, after Steger et al. (2017). Kuipers Munneke et al. (2015) used T.
- Crocus is a hybrid. Eq. 22 applies only for rho < 450 kg m-3. For
  rho >= 450 kg m-3, the code uses the zone 2 rate c1(KM) (rho_i - rho), with
  bdot_mean. The liquid-water factor f1 has no effect there.

The accumulation input of the tests is in m ice eq. a-1, as in the model
(bdotSec and bdot_mean). The model converts it with rho_i = 917 kg m-3.
'''

import inspect

import numpy as np
import pytest

from config_validation import ALLOWED_VALUES
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


def _node_age(n):
    '''The age [s] of each node. The surface node is one year old, so each age is above 0.'''
    return (np.arange(n) + 1.0) * SECONDS_PER_YEAR


def _drho_dt(law, rho, Tz, bdot_type='mean', bdot_instant=0.3, bdot_mean=None,
             T_mean=250.0, sigma=None, dz=None, LWC=None, extra_inputs=None):
    '''
    drho_dt [kg m-3 s-1] from FirnPhysics(PhysParams).<law>() for a column of
    len(rho) nodes. The keys of PhysParams are the keys that FirnDensityNoSpin
    gives.

    bdot_instant [m ice eq. a-1] is the accumulation rate of step ITERATION.
    bdot_mean [m ice eq. a-1] is the accumulation rate of each node. T_mean [K]
    is the mean annual temperature of step ITERATION. extra_inputs adds keys to
    PhysParams or replaces them, for the laws that need more inputs.
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
        'age':           _node_age(n),
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
    PhysParams.update(extra_inputs or {})
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


def _crocus(rho, T, sigma, theta):
    f1 = 1.0 / (1.0 + 60.0 * theta)
    eta = f1 * 4.0 * 7.62237e6 * rho / 358.0 * np.exp(0.1 * (273.15 - T) + 0.023 * rho)
    return rho * sigma / eta


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


# ---------------------------------------------------------------------------
# Crocus

CROCUS_DENSITIES = np.array([300.0, 340.0, 380.0, 420.0, 449.0])  # kg m-3, all below 450
CROCUS_STRESS = np.array([3.0e2, 2.0e3, 6.0e3, 1.5e4, 3.0e4])  # Pa


@pytest.mark.unit
@pytest.mark.parametrize('T', TEMPERATURES)
@pytest.mark.parametrize('b_ice', ACCUMULATIONS)
def test_Crocus_dry_matches_vionnet(T, b_ice):
    '''Below 450 kg m-3 the rate does not depend on the accumulation rate.'''
    Tz = T + TEMPERATURE_OFFSETS
    got = _drho_dt('Crocus', CROCUS_DENSITIES, Tz, 'mean', bdot_instant=b_ice, bdot_mean=b_ice,
                   sigma=CROCUS_STRESS)
    expected = _crocus(CROCUS_DENSITIES, Tz, CROCUS_STRESS, theta=0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('T', [273.15, 268.15])
@pytest.mark.parametrize('b_ice', ACCUMULATIONS)
def test_Crocus_wet_viscosity_factor_uses_the_volumetric_water_content(T, b_ice):
    '''
    Verjans et al. (2019), Eq. 3: f1 = 1 / (1 + 60 theta), with theta the
    volumetric water content [m3 m-3]. The CFM gives LWC as a water thickness
    [m] for each node, so theta = LWC / dz. The node thicknesses are not 1 m, so
    a factor 1 / (1 + 60 LWC) gives a different result.
    '''
    dz = np.array([0.05, 0.1, 0.2, 0.08, 0.15])  # m
    theta = np.array([0.0, 0.005, 0.02, 0.05, 0.08])  # m3 m-3
    Tz = np.full(5, T)
    got = _drho_dt('Crocus', CROCUS_DENSITIES, Tz, 'mean', bdot_instant=b_ice, bdot_mean=b_ice,
                   sigma=CROCUS_STRESS, dz=dz, LWC=theta * dz)
    np.testing.assert_allclose(got, _crocus(CROCUS_DENSITIES, Tz, CROCUS_STRESS, theta),
                               rtol=1e-12, atol=0)
    dry = _drho_dt('Crocus', CROCUS_DENSITIES, Tz, 'mean', bdot_instant=b_ice, bdot_mean=b_ice,
                   sigma=CROCUS_STRESS, dz=dz)
    np.testing.assert_allclose(got / dry, 1.0 + 60.0 * theta, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('bdot_type', ['instant', 'mean'])
@pytest.mark.parametrize('T', TEMPERATURES)
@pytest.mark.parametrize('b_ice', ACCUMULATIONS)
def test_Crocus_uses_kuipers_munneke_zone_2_at_450_and_above(bdot_type, T, b_ice):
    '''
    The code is a hybrid of Crocus and KuipersMunneke2015. For rho >= 450 kg m-3
    the rate is c1(KM) (rho_i - rho), with M1, 0.03 and bdot_mean, also for
    450 <= rho < 550 kg m-3 and for bdot_type "instant". The wet factor f1 has
    no effect there. Stevens et al. (2020) and Verjans et al. (2019) give
    Eq. 22 for all densities.
    '''
    rho = np.array([350.0, 449.0, 450.0, 500.0, 600.0, 800.0])
    Tz = T + np.array([0.0, -1.5, 1.0, -0.5, 2.0, 0.5])
    dz = np.full(6, 0.1)
    theta = np.full(6, 0.03)
    sigma = np.array([2.0e3, 1.5e4, 1.6e4, 3.0e4, 8.0e4, 3.0e5])
    T_mean = T - 4.0
    b_node = b_ice * np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.05])
    got = _drho_dt('Crocus', rho, Tz, bdot_type, bdot_instant=3.0 * b_ice, bdot_mean=b_node,
                   T_mean=T_mean, sigma=sigma, dz=dz, LWC=theta * dz)
    below = rho < 450.0
    expected = np.where(below, _crocus(rho, Tz, sigma, theta),
                        _kuipers_munneke(rho, Tz, T_mean, b_node, zone_1_limit=0.0))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Properties of all laws

LAWS_WITH_A_RATE = [
    'Arthern2010S', 'Arthern2010T', 'Barnola1991', 'Breant2017', 'Brils2022', 'Crocus',
    'GSFC2020', 'Goujon2003', 'HLSigfus', 'HLdynamic', 'Helsen2008', 'KuipersMunneke2015',
    'Li2004', 'Li2011', 'Li2015', 'Ligtenberg2011', 'MaxSP',
    'Morris2014', 'Simonsen2013', 'Veldhuijsen2023', 'Yamazaki1993',
]
LAWS_WITH_NO_RATE = ['Marshall1999']  # not yet written: the method returns None
# The validator rejects these laws, and no other code calls them, so a model run cannot use them.
LAWS_THAT_CANNOT_BE_SELECTED = ['ArthernWingham1998', 'Max2018', 'Max2018b']
FIRNPHYSICS_METHODS_THAT_ARE_NOT_LAWS = {'__init__', 'surfacegrain', 'graincalc'}

# Inputs that FirnDensityNoSpin gives only to some laws. The node spacing of the
# columns below is 0.5 kg m-3, so that a node falls in the narrow density range
# that Goujon2003 needs at the zone 1 to zone 2 transition.
GOUJON_INPUTS = {'Gamma_Gou': 0.0, 'Gamma_old_Gou': 0.0, 'Gamma_old2_Gou': 0.0, 'ind1_old': 0}
LAW_INPUTS = {
    'Arthern2010T': {'physGrain': True},
    'Breant2017': GOUJON_INPUTS,  # FirnDensityNoSpin does not give them to this law
    'Goujon2003': GOUJON_INPUTS,
}
MORRIS_ACTIVATION_ENERGY = 110.0e3  # J mol-1, the default of FirnDensityNoSpin

NODE_THICKNESS = 0.03  # m, so that rho = 550 kg m-3 is at about 15 m
SMOOTH_COLUMN = np.concatenate([np.arange(300.25, 907.0, 0.5),
                                [907.0, 912.0, 916.0, 916.9, 916.99, RHO_ICE]])  # kg m-3
LOWEST_ACTIVATION_ENERGY = 10.0e3  # J mol-1. Q1 of Herron and Langway (1980) is the lowest, 10.16 kJ mol-1.

# The conditions of the sign and NaN tests.
PROPERTY_T_MEANS = [233.15, 253.15, 258.15, 263.15, 271.15]  # K
PROPERTY_ACCUMULATIONS = [0.1, 0.5, 1.0]  # m ice eq. a-1
PROPERTY_CONDITIONS = [(T_mean, b_ice) for T_mean in PROPERTY_T_MEANS for b_ice in PROPERTY_ACCUMULATIONS]

# The conditions in which a law gives a negative rate, with the cause. The test leaves them out.
NEGATIVE_RATES = {
    'Breant2017': (lambda T_mean, b_ice: T_mean >= 271.15,
                   'the ice density of the law is below 917 kg m-3 for Tm above 269.6 K'),
    'Goujon2003': (lambda T_mean, b_ice: T_mean >= 271.15,
                   'the ice density of the law is below 917 kg m-3 for Tm above 269.6 K'),
    'Helsen2008': (lambda T_mean, b_ice: T_mean >= 263.15,
                   'the factor 76.138 - 0.28965 Tm is negative for Tm above 262.86 K'),
    'Li2004': (lambda T_mean, b_ice: T_mean >= 258.15,
               'the factor 139.21 - 0.542 Tm is negative for Tm above 256.85 K'),
    'Li2011': (lambda T_mean, b_ice: T_mean >= 271.15 or (T_mean >= 263.15 and b_ice <= 0.1),
               'beta1 is negative for a warm Tm and a low accumulation rate'),
    'Li2015': (lambda T_mean, b_ice: T_mean >= 271.15 or b_ice >= 1.0,
               'beta1 is negative for Tm above 270.13 K, and beta2 at a high accumulation rate'),
}


def _law_rate(law, rho=SMOOTH_COLUMN, T=253.15, bdot_type='mean', b_ice=0.3, bdot_instant=None, T_mean=250.0):
    '''
    drho_dt [kg m-3 s-1] of any law in physics.py, for a column at the
    accumulation rate b_ice [m ice eq. a-1]. bdot_instant replaces b_ice for the
    current time step. The stress is the weight of the column. The viscosity calculations of some laws divide by a rate of 0 at the
    ice density. The tests here do not read the viscosity, so they ignore this.
    '''
    rho = np.asarray(rho, dtype=float)
    extra_inputs = dict(LAW_INPUTS.get(law, {}))
    if law == 'Morris2014':
        # The form of the thermal history in FirnDensitySpin. It uses T_mean, so a change of T does not change it.
        extra_inputs['QMorris'] = MORRIS_ACTIVATION_ENERGY
        extra_inputs['Hx'] = np.exp(-MORRIS_ACTIVATION_ENERGY / (R_GAS * T_mean)) * _node_age(rho.size)
    with np.errstate(divide='ignore', invalid='ignore'):
        return _drho_dt(law, rho, T, bdot_type, bdot_instant=b_ice if bdot_instant is None else bdot_instant,
                        bdot_mean=b_ice, T_mean=T_mean, dz=np.full(rho.size, NODE_THICKNESS),
                        extra_inputs=extra_inputs)


def _accumulation_inputs_with_a_rate(law):
    '''
    The values of bdot_type that the property tests run. Each law has one branch of
    code for each value. GSFC2020 stops on purpose with "instant".
    '''
    return ['mean'] if law == 'GSFC2020' else ['mean', 'instant']


def _laws_except(left_out):
    '''LAWS_WITH_A_RATE without the laws in the dict left_out, which gives the cause for each.'''
    return [law for law in LAWS_WITH_A_RATE if law not in left_out]


@pytest.mark.unit
def test_the_law_lists_have_each_law_in_physics_py():
    '''
    A new law in physics.py makes this test fail until a list includes it. A law
    that the validator accepts must have the property tests.
    '''
    methods = {name for name, _ in inspect.getmembers(FirnPhysics, inspect.isfunction)}
    laws = set(LAWS_WITH_A_RATE) | set(LAWS_WITH_NO_RATE) | set(LAWS_THAT_CANNOT_BE_SELECTED)
    assert methods - FIRNPHYSICS_METHODS_THAT_ARE_NOT_LAWS == laws
    assert ALLOWED_VALUES['physRho'] <= set(LAWS_WITH_A_RATE)


@pytest.mark.unit
@pytest.mark.parametrize('law', LAWS_WITH_A_RATE)
def test_each_law_gives_a_rate_for_each_node(law):
    assert _law_rate(law).shape == SMOOTH_COLUMN.shape


def _conditions_with_a_rate_that_is_not_negative(law):
    if law not in NEGATIVE_RATES:
        return PROPERTY_CONDITIONS
    is_negative, _cause = NEGATIVE_RATES[law]
    return [condition for condition in PROPERTY_CONDITIONS if not is_negative(*condition)]


@pytest.mark.unit
@pytest.mark.parametrize('law', LAWS_WITH_A_RATE)
def test_rate_is_not_negative(law):
    for T_mean, b_ice in _conditions_with_a_rate_that_is_not_negative(law):
        for bdot_type in _accumulation_inputs_with_a_rate(law):
            rate = _law_rate(law, T=T_mean, bdot_type=bdot_type, T_mean=T_mean, b_ice=b_ice)
            assert np.all(rate >= 0.0), (f'{bdot_type}, Tm = {T_mean} K, b = {b_ice} m ice eq. a-1, '
                                         f'rho = {SMOOTH_COLUMN[~(rate >= 0.0)]}')


@pytest.mark.unit
@pytest.mark.parametrize('law', LAWS_WITH_A_RATE)
def test_rate_is_finite_for_each_density(law):
    for T_mean, b_ice in PROPERTY_CONDITIONS:
        for bdot_type in _accumulation_inputs_with_a_rate(law):
            rate = _law_rate(law, T=T_mean, bdot_type=bdot_type, T_mean=T_mean, b_ice=b_ice)
            assert np.all(np.isfinite(rate)), f'{bdot_type}, Tm = {T_mean} K, b = {b_ice} m ice eq. a-1'


def _layered_column(kind):
    '''
    Melt makes a column that is not monotonic. "thin_layers" has only layers of
    +30 and -30 kg m-3. "lenses" adds dense firn near the surface and ice lenses.
    "crust" also adds a refrozen node at the surface.
    '''
    n = 600
    rho = np.linspace(320.0, 880.0, n) + 30.0 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    if kind in ('lenses', 'crust'):
        rho[20] = 545.0  # dense firn, still in zone 1
        rho[[60, 61, 150, 300]] = 900.0  # ice lenses
    if kind == 'crust':
        rho[0] = 850.0
    return rho


# The columns that a law does not accept, with the cause. The test leaves them out.
GOUJON_ZONES = 'the zones are found from the first node above the transition density'
MORRIS_LAYER = 'the factor 1 - 3.3 m is negative for a node far above the fitted density'
LAYERED_COLUMNS_LEFT_OUT = {
    **{(law, kind): GOUJON_ZONES for law in ['Breant2017', 'Goujon2003'] for kind in ['thin_layers', 'lenses', 'crust']},
    ('HLSigfus', 'crust'): 'with a zone 2 node at the surface, the stress at 550 kg m-3 reads the bottom node',
    ('Morris2014', 'lenses'): MORRIS_LAYER,
    ('Morris2014', 'crust'): MORRIS_LAYER,
}


@pytest.mark.unit
@pytest.mark.parametrize('law, kind', [
    (law, kind) for law in LAWS_WITH_A_RATE for kind in ('thin_layers', 'lenses', 'crust')
    if (law, kind) not in LAYERED_COLUMNS_LEFT_OUT])
def test_rate_is_finite_and_not_negative_in_a_layered_column(law, kind):
    rho = _layered_column(kind)
    rate = _law_rate(law, rho=rho, T=253.15, T_mean=253.15)
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0.0), f'rho = {rho[rate < 0.0]}'


INFINITE_AT_THE_MELTING_POINT = dict.fromkeys(
    ['Helsen2008', 'Li2004', 'Li2011', 'Li2015'], 'the factor (273.15 - T)^-2.061 is infinite at the melting point')


@pytest.mark.unit
@pytest.mark.parametrize('law', _laws_except(INFINITE_AT_THE_MELTING_POINT))
def test_rate_is_finite_at_the_melting_point(law):
    assert np.all(np.isfinite(_law_rate(law, T=273.15, T_mean=253.15)))


@pytest.mark.unit
@pytest.mark.parametrize('law', _laws_except(
    {'Yamazaki1993': 'a snow law: the model uses it only for rho below 300 kg m-3'}))
def test_rate_goes_to_zero_at_the_ice_density(law):
    for bdot_type in _accumulation_inputs_with_a_rate(law):
        rate = _law_rate(law, bdot_type=bdot_type)
        near_ice = rate[-6:-1]  # 907, 912, 916, 916.9 and 916.99 kg m-3
        assert np.all(np.diff(near_ice) < 0.0), bdot_type
        assert rate[-1] <= 1.0e-3 * np.median(rate), bdot_type  # 917 kg m-3


@pytest.mark.unit
@pytest.mark.parametrize('law', LAWS_WITH_A_RATE)
@pytest.mark.parametrize('T', [233.15, 253.15, 263.15])
def test_rate_increases_with_temperature(law, T):
    '''
    A 5 K change of T alone, at the same rho, stress, accumulation and Tm. The
    increase is at least that of the lowest activation energy of the laws. So a
    law with a rate that is almost independent of T fails.
    '''
    warmer = T + 5.0
    below_ice = SMOOTH_COLUMN < RHO_ICE
    lowest = np.exp(LOWEST_ACTIVATION_ENERGY / R_GAS * (1.0 / T - 1.0 / warmer))
    for bdot_type in _accumulation_inputs_with_a_rate(law):
        ratio = (_law_rate(law, T=warmer, bdot_type=bdot_type)[below_ice]
                 / _law_rate(law, T=T, bdot_type=bdot_type)[below_ice])
        assert np.all(ratio >= lowest), f'{bdot_type}: lowest ratio {ratio.min()}, not {lowest}'


NODE_AT_550_COLUMN = np.concatenate([np.arange(300.25, 550.0, 0.5), [550.0], np.arange(550.25, 900.0, 0.5)])


@pytest.mark.unit
@pytest.mark.parametrize('law', LAWS_WITH_A_RATE)
def test_a_node_at_550_has_a_rate(law):
    for bdot_type in _accumulation_inputs_with_a_rate(law):
        rate = _law_rate(law, rho=NODE_AT_550_COLUMN, bdot_type=bdot_type)
        assert rate[NODE_AT_550_COLUMN == 550.0] > 0.0, bdot_type


@pytest.mark.unit
def test_HLSigfus_node_at_550_uses_the_transition_equation():
    '''From 550 to 551 kg m-3, the law uses the zone 2 rate of Herron and Langway (1980).'''
    T, b_ice = 253.15, 0.3
    rate = _law_rate('HLSigfus', rho=NODE_AT_550_COLUMN, T=T, b_ice=b_ice)
    at_550 = NODE_AT_550_COLUMN == 550.0
    np.testing.assert_allclose(rate[at_550], _herron_langway(NODE_AT_550_COLUMN[at_550], T, b_ice), rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('law, bdot_type', [
    (law, bdot_type) for law in sorted(ALLOWED_VALUES['physRho']) for bdot_type in ('mean', 'instant')])
def test_each_bdot_type_gives_a_rate_or_stops_the_run(law, bdot_type):
    '''
    A law must densify each node below the ice density, or stop with a message. A
    law can leave one zone at 0, and the other zone then looks correct. bdot_type
    "stress" is not here, because docs/running/json.rst says that it is not
    complete.
    '''
    try:
        rate = _law_rate(law, bdot_type=bdot_type)
    except SystemExit:  # GSFC2020 stops for "instant"
        return
    assert np.all(np.isfinite(rate))
    no_rate = (rate == 0.0) & (SMOOTH_COLUMN < RHO_ICE)
    assert not np.any(no_rate), f'{no_rate.sum()} nodes have a rate of 0, for rho from {SMOOTH_COLUMN[no_rate][0]}'


@pytest.mark.unit
def test_Li2015_instant_agrees_with_mean_at_one_accumulation_rate():
    '''beta1 and beta2 do not depend on bdot_type, so the same rate gives the same result.'''
    np.testing.assert_allclose(_law_rate('Li2015', bdot_type='instant'), _law_rate('Li2015', bdot_type='mean'),
                               rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize('law', LAWS_WITH_A_RATE)
def test_zero_instant_accumulation_gives_a_finite_rate(law):
    '''With daily forcing, many time steps have no accumulation.'''
    try:
        rate = _law_rate(law, bdot_type='instant', bdot_instant=0.0)
    except SystemExit:  # GSFC2020 stops for "instant"
        return
    assert np.all(np.isfinite(rate))


@pytest.mark.unit
@pytest.mark.parametrize('law', ['Brils2022', 'KuipersMunneke2015', 'Ligtenberg2011', 'Simonsen2013', 'Veldhuijsen2023'])
def test_rate_at_zero_instant_accumulation_is_the_limit_of_the_rate(law):
    '''
    These laws have ln(b) or 1/sqrt(b). The product with b still goes to 0 as b
    goes to 0, so 0 is the rate at b = 0.
    '''
    below_ice = SMOOTH_COLUMN < RHO_ICE
    typical = _law_rate(law, bdot_type='instant', bdot_instant=0.3)[below_ice]
    near_zero = _law_rate(law, bdot_type='instant', bdot_instant=1.0e-9)[below_ice]
    assert np.all(near_zero <= 1.0e-3 * typical)
    assert np.all(_law_rate(law, bdot_type='instant', bdot_instant=0.0) == 0.0)
