'''
The laws for the hydraulic conductivity of darcyscheme. Ksat_darcy selects the saturated
conductivity and krel_darcy the relative conductivity. The unit tests check each law against
values computed by hand. The scheme tests show that each key reaches the law in a call of
darcyscheme. The front test of the laws is in test_darcy_front_speed.py.
'''

import numpy as np
import pytest

import melt
from cfm_testing import example_config
from config_validation import KNOWN_KEYS, ConfigValidationError, validate_config
from darcy_funcs import hydrconducsat_Calonne, hydrconducsat_power_law, krel_power_law

NEW_KEYS = {'Ksat_darcy': 'Calonne', 'Ksat_k0_darcy': 5.56e-11, 'viscosity_darcy': 0.001792,
            'krel_darcy': 'vanGenuchten', 'krel_exponent_darcy': 2.0}

# [kg m-3], K_sat [m s-1], computed by hand with g = 9.8. power_law: K_h = k0 rho_w g / mu = 5e-4 m s-1,
# the rounded value of Shadab et al. (2025, Table 1). Calonne: radius 0.5 mm and mu = 1.79e-3 Pa s.
POWER_LAW_VALUES = [(350, 1.2e-4), (500, 4.7e-5), (650, 1.2e-5), (800, 1.0e-6), (830, 4.3e-7), (880, 3.3e-8)]
CALONNE_VALUES = [(350, 4.3e-2), (500, 6.2e-3), (650, 8.8e-4), (800, 1.3e-4), (830, 8.5e-5), (880, 4.4e-5),
                  (917, 2.7e-5)]
ROUNDING = 0.06  # the values have two significant figures
K0_FOR_TABLE = 5e-4 * 1e-3 / (1000.0 * 9.8)  # [m2], with mu = 1e-3 Pa s and the g of the CFM


@pytest.mark.unit
@pytest.mark.parametrize('rho, expected', POWER_LAW_VALUES)
def test_power_law_ksat_values(rho, expected):
    assert hydrconducsat_power_law(1 - rho / 917.0, K0_FOR_TABLE, 1e-3) == pytest.approx(expected, rel=ROUNDING)


@pytest.mark.unit
@pytest.mark.parametrize('porosity', [0.0, -0.01])
def test_power_law_ksat_is_zero_at_zero_porosity_or_below(porosity):
    # The porosity is below zero above 917 kg m-3, if rholim_dcy is above 917.
    assert hydrconducsat_power_law(porosity, 5.56e-11, 1e-3) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize('rho, expected', CALONNE_VALUES)
def test_calonne_ksat_values(rho, expected):
    assert hydrconducsat_Calonne(5e-4, rho) == pytest.approx(expected, rel=ROUNDING)


@pytest.mark.unit
def test_calonne_default_viscosity_is_the_scheme_default():
    rho = np.array([350.0, 600.0, 900.0])
    assert np.array_equal(hydrconducsat_Calonne(5e-4, rho),
                          hydrconducsat_Calonne(5e-4, rho, melt._DARCY_DEFAULTS['viscosity_darcy']))


@pytest.mark.unit
def test_both_ksat_laws_scale_as_one_over_viscosity():
    assert hydrconducsat_Calonne(5e-4, 500.0, 2e-3) == pytest.approx(hydrconducsat_Calonne(5e-4, 500.0, 1e-3) / 2)
    assert hydrconducsat_power_law(0.4, 5.56e-11, 2e-3) == pytest.approx(
        hydrconducsat_power_law(0.4, 5.56e-11, 1e-3) / 2)


@pytest.mark.unit
@pytest.mark.parametrize('saturation, n, expected', [(0.5, 2, 0.25), (0.5, 3, 0.125), (0.2, 2.5, 0.2 ** 2.5),
                                                     (0.0, 2, 0.0), (1.0, 2, 1.0)])
def test_power_law_krel_values(saturation, n, expected):
    assert krel_power_law(saturation, n) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_defaults_give_the_laws_of_the_scheme_before_the_keys():
    assert {key: melt._DARCY_DEFAULTS[key] for key in NEW_KEYS} == NEW_KEYS


@pytest.mark.unit
def test_validator_knows_the_keys():
    assert set(NEW_KEYS) <= KNOWN_KEYS


@pytest.mark.unit
@pytest.mark.parametrize('key', ['Ksat_darcy', 'krel_darcy'])
def test_validator_accepts_the_law_names_only(key):
    validate_config({**example_config(), key: 'power_law'})
    with pytest.raises(ConfigValidationError, match=key):
        validate_config({**example_config(), key: 'powerlaw'})


def _centre_of_water(synthetic_column, **config):
    '''
    Rain for 3 hours on temperate firn of 400 kg m-3, in one call of darcyscheme. Returns the depth
    of the centre of mass of the water [m]. A law with a larger conductivity moves it down.
    '''
    column = synthetic_column(n=60, rho_profile=400.0, T_profile=273.15, dz=0.02, r2=(5e-4) ** 2, rain=0.005,
                              dt=3 * 3600.0, config={'ColeouLesaffre': False, 'IrrVal': 0.0, **config})
    melt.darcyscheme(column, 0)
    depth = np.cumsum(column.dz) - column.dz / 2
    return float((column.LWC * depth).sum() / column.LWC.sum())


@pytest.mark.scheme
@pytest.mark.parametrize('base, changed', [
    ({}, {'Ksat_darcy': 'power_law'}),  # about 1000 times smaller at r = 0.5 mm
    ({'Ksat_darcy': 'power_law'}, {'Ksat_darcy': 'power_law', 'Ksat_k0_darcy': 0.5 * 5.56e-11}),
    ({}, {'viscosity_darcy': 2 * 0.001792}),
    ({'Ksat_darcy': 'power_law'}, {'Ksat_darcy': 'power_law', 'viscosity_darcy': 2 * 0.001792}),
    ({'krel_darcy': 'power_law', 'krel_exponent_darcy': 3}, {'krel_darcy': 'power_law', 'krel_exponent_darcy': 4}),
])
def test_a_smaller_conductivity_keeps_the_water_higher(synthetic_column, base, changed):
    assert _centre_of_water(synthetic_column, **changed) < 0.9 * _centre_of_water(synthetic_column, **base)


@pytest.mark.scheme
def test_power_law_uses_the_porosity_of_the_capped_density(synthetic_column, monkeypatch):
    '''A node of solid ice has the porosity of rholim_dcy in the law, not zero.'''
    porosities = []

    def record(phi, k0, mu):
        porosities.append(np.copy(phi))
        return hydrconducsat_power_law(phi, k0, mu)

    monkeypatch.setattr(melt, 'hydrconducsat_power_law', record)
    rho = np.full(20, 400.0)
    rho[10] = 917.0
    column = synthetic_column(n=20, rho_profile=rho, T_profile=273.15, rain=0.001,
                              config={'Ksat_darcy': 'power_law', 'rholim_dcy': 910.0})
    melt.darcyscheme(column, 0)
    assert porosities
    assert min(phi.min() for phi in porosities) == pytest.approx((917.0 - 910.0) / 917.0, rel=1e-12)


@pytest.mark.scheme
def test_krel_option_changes_the_flux(synthetic_column):
    base = _centre_of_water(synthetic_column)
    assert _centre_of_water(synthetic_column, krel_darcy='power_law') != pytest.approx(base, rel=0.1)


@pytest.mark.scheme
@pytest.mark.parametrize('key', ['Ksat_darcy', 'krel_darcy'])
def test_an_unknown_law_stops_the_scheme(synthetic_column, key):
    column = synthetic_column(n=20, rho_profile=400.0, T_profile=273.15, rain=0.001, config={key: 'powerlaw'})
    with pytest.raises(ValueError, match=key):
        melt.darcyscheme(column, 0)
