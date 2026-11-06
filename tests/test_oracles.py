'''
The oracles against values calculated by hand. The values come from plain arithmetic
outside the module, so these tests do not compare the module with itself.
'''

import pytest

from darcy_funcs import hydrconducsat_Calonne, krel_vG, vG_Yama_params
from oracles.kinematic_wave import (FirnConstants, freezing_water, front_speed, porosity_drop,
                                    power_law_front, power_law_saturation, solve_front)

pytestmark = pytest.mark.unit

DAY = 86400.0
# Meyer and Hewitt (2017), Table 1: one density for ice and water.
ONE_DENSITY = FirnConstants(rho_ice=917.0, rho_water=917.0, specific_heat=2050.0, latent_heat=334000.0)
TWO_DENSITIES = FirnConstants(rho_ice=917.0, rho_water=1000.0, specific_heat=2050.0, latent_heat=334000.0)
CONDUCTIVITY, POROSITY, FLUX = 5e-4, 0.5, 1e-6


def power_law(theta, porosity):
    return CONDUCTIVITY * porosity**3 * (theta / porosity)**2


@pytest.mark.parametrize('constants', [ONE_DENSITY, TWO_DENSITIES])
def test_power_law_at_the_melting_temperature(constants):
    '''The densities have no effect when the firn is not cold.'''
    front = power_law_front(FLUX, CONDUCTIVITY, POROSITY, constants)
    assert power_law_saturation(FLUX, CONDUCTIVITY, POROSITY) == pytest.approx(0.126491, rel=1e-5)
    assert front.speed * DAY == pytest.approx(1.36610, rel=1e-5)
    assert front.porosity_behind == POROSITY


@pytest.mark.parametrize('constants, speed', [(ONE_DENSITY, 0.900288), (TWO_DENSITIES, 0.924835)])
def test_power_law_at_minus_ten(constants, speed):
    '''One density is eq. C17 of Meyer and Hewitt. Two densities is the derived form.'''
    front = power_law_front(FLUX, CONDUCTIVITY, POROSITY, constants, cold_degrees=10.0)
    assert front.speed * DAY == pytest.approx(speed, rel=1e-5)
    assert front.porosity_behind == pytest.approx(0.469311, rel=1e-5)
    assert front.water_behind == pytest.approx(0.0652806, rel=1e-5)


def test_density_ratio_is_only_in_the_freezing_water():
    assert porosity_drop(POROSITY, TWO_DENSITIES, 10.0) == porosity_drop(POROSITY, ONE_DENSITY, 10.0)
    assert porosity_drop(POROSITY, TWO_DENSITIES, 10.0) == pytest.approx(0.0306886, rel=1e-5)
    assert freezing_water(POROSITY, TWO_DENSITIES, 10.0) == pytest.approx(0.0281415, rel=1e-5)


@pytest.mark.parametrize('constants', [ONE_DENSITY, TWO_DENSITIES])
@pytest.mark.parametrize('cold_degrees', [0.0, 10.0])
def test_root_solve_agrees_with_the_closed_form(constants, cold_degrees):
    solved = solve_front(power_law, FLUX, POROSITY, constants, cold_degrees)
    closed = power_law_front(FLUX, CONDUCTIVITY, POROSITY, constants, cold_degrees)
    assert solved.speed == pytest.approx(closed.speed, rel=1e-9)
    assert solved.water_behind == pytest.approx(closed.water_behind, rel=1e-9)


def test_front_closes_the_three_balances():
    '''Put the result back in the water, ice and enthalpy balances of the derivation.'''
    c = TWO_DENSITIES
    cold_degrees = 10.0
    front = solve_front(power_law, FLUX, POROSITY, c, cold_degrees)
    frozen_mass_rate = c.rho_ice * front.speed * (POROSITY - front.porosity_behind)
    assert c.rho_water * front.speed * front.water_behind == pytest.approx(c.rho_water * FLUX - frozen_mass_rate, rel=1e-9)
    enthalpy_ahead = -c.rho_ice * (1 - POROSITY) * c.specific_heat * cold_degrees
    enthalpy_behind = c.rho_water * c.latent_heat * front.water_behind
    assert front.speed * (enthalpy_behind - enthalpy_ahead) == pytest.approx(c.rho_water * c.latent_heat * FLUX, rel=1e-9)


def test_immobile_water_is_in_the_jump():
    '''q = K * (theta - 0.02) for theta above 0.02: theta+ = 0.03 and V = R / 0.03.'''
    law = lambda theta, porosity: 1e-4 * max(theta - 0.02, 0.0)
    front = solve_front(law, FLUX, POROSITY, ONE_DENSITY)
    assert front.water_behind == pytest.approx(0.03, rel=1e-9)
    assert front.speed == pytest.approx(FLUX / 0.03, rel=1e-9)


def test_front_speed_with_water_ahead():
    '''A wet state ahead of the front: V = (4e-6 - 1e-6) / (0.10 - 0.04).'''
    assert front_speed(4e-6, 0.10, flux_ahead=1e-6, water_ahead=0.04) == pytest.approx(5e-5, rel=1e-12)


@pytest.mark.parametrize('flux, water, speed', [(5e-8, 0.0241725, 0.178716), (1e-6, 0.0315837, 2.73559)])
def test_darcy_law_from_the_cfm(flux, water, speed):
    '''hydrconducsat_Calonne times krel_vG at 400 kg m-3 and 1 mm radius, with theta_i = 0.02.'''
    radius, theta_s_factor, theta_i = 1e-3, 0.917, 0.02

    def law(theta, porosity):
        rho = 917.0 * (1 - porosity)
        theta_s = theta_s_factor * porosity
        theta_e = min(max((theta - theta_i) / (theta_s - theta_i), 1e-9), 1 - 1e-9)
        return hydrconducsat_Calonne(radius, rho) * krel_vG(vG_Yama_params(radius, rho)[2], theta_e)

    front = solve_front(law, flux, 1 - 400.0 / 917.0, TWO_DENSITIES, saturated_fraction=theta_s_factor)
    assert front.water_behind == pytest.approx(water, rel=1e-5)
    assert front.speed * DAY == pytest.approx(speed, rel=1e-5)


@pytest.mark.parametrize('oracle', [
    lambda: power_law_front(FLUX, CONDUCTIVITY, 0.05, ONE_DENSITY, cold_degrees=10.0),
    lambda: solve_front(power_law, FLUX, 0.05, ONE_DENSITY, cold_degrees=10.0),
])
def test_no_front_when_the_pores_freeze_shut(oracle):
    '''The drop at 0.05 porosity and -10 C is 0.0583, more than the porosity.'''
    with pytest.raises(ValueError, match='freeze shut'):
        oracle()


@pytest.mark.parametrize('oracle', [
    lambda: power_law_front(1e-3, CONDUCTIVITY, POROSITY, ONE_DENSITY),
    lambda: solve_front(power_law, 1e-3, POROSITY, ONE_DENSITY),
])
def test_no_front_when_the_law_cannot_carry_the_flux(oracle):
    '''The largest flux of the power law at 0.5 porosity is 6.25e-5 m/s.'''
    with pytest.raises(ValueError, match='cannot carry'):
        oracle()


def test_saturated_fraction_limits_the_water_content():
    '''The power law carries 5.26e-5 m/s at S = 0.917, so 6e-5 m/s fits only at full saturation.'''
    assert solve_front(power_law, 6e-5, POROSITY, ONE_DENSITY).water_behind == pytest.approx(0.489898, rel=1e-5)
    with pytest.raises(ValueError, match='cannot carry'):
        solve_front(power_law, 6e-5, POROSITY, ONE_DENSITY, saturated_fraction=0.917)


def test_limit_of_the_law_uses_the_porosity_behind_a_cold_front():
    '''The largest flux at -10 C is 5.17e-5 m/s at phi+ = 0.4693, less than 6.25e-5 m/s at phi0.'''
    with pytest.raises(ValueError, match='cannot carry'):
        solve_front(power_law, 5.5e-5, POROSITY, ONE_DENSITY, cold_degrees=10.0)


def test_negative_cold_degrees_is_an_error():
    with pytest.raises(ValueError, match='must not be negative'):
        solve_front(power_law, FLUX, POROSITY, ONE_DENSITY, cold_degrees=-10.0)
