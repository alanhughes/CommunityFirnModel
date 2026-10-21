'''
Tests for the densification law HLdynamic in physics.py against the
equations of Stevens et al. (2020, Geosci. Model Dev. 13, 4355-4377).

The reference values are calculated in this file from the published equations,
with literal constants. They do not use constants.py.

    Herron and Langway (1980), Stevens et al. Eqs. 4-6, rho in Mg m-3, b in m w.e. a-1:
        drho/dt = c (rho_i - rho)
        c0 = 11 exp(-10160 / RT) b,  c1 = 575 exp(-21400 / RT) b^0.5

Choices in the code that the papers do not give:

- Zone 1 is rho < 550 kg m-3 and zone 2 is rho >= 550 kg m-3. Eq. 6 puts
  rho = 550 kg m-3 in zone 1. The rate is not continuous at 550 kg m-3.
- One year is 365.25 days.

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
