'''
A wetting front that moves down into uniform firn, from the balances across the front.

The problem. A constant water flux R enters uniform firn at the top. The firn ahead of
the front has porosity phi0 and temperature Tm - dT. Capillary gradients are zero. Behind
the front, the firn is at the melting temperature Tm. The ice does not move.

Meyer and Hewitt (2017, App. C) solve this problem with rho_i = rho_w. This module keeps
the two densities distinct, because the CFM conserves mass with rho_i != rho_w.

Derivation. Let theta be the volume of water per volume of firn, and q the water flux in
m/s. Let M be the mass that freezes at the front, per area and per time. A front at speed
V changes the amount of a quantity u in a fixed interval at the rate V*(u_behind - u_ahead).
The amount also changes by the flux in at the top minus the flux out at the bottom. So:

    water:    rho_w * V * (theta+ - theta-) = rho_w * (q+ - q-) - M
    ice:      rho_i * V * (phi- - phi+)     = M
    enthalpy: V * (H+ - H-)                 = rho_w * L * (q+ - q-)

Here + is behind the front and - is ahead of it. The enthalpy per firn volume, relative to
ice at Tm, is H = rho_i*(1-phi)*c*(T-Tm) + rho_w*L*theta. Water enters at Tm, so it brings
only its latent heat. The freezing is internal to the enthalpy balance, so M is not in it.

The enthalpy balance gives the front speed:

    V = (q+ - q-) / (theta+ - theta- + E)
    E = (rho_i/rho_w) * (1-phi0) * c * dT / L

E is the water that freezes to warm one volume of firn to Tm. Because rho_i*(1-phi0) is the
firn density, E is the cold content divided by L, in water equivalent per firn thickness.

The water balance then gives M = rho_i * V * (1-phi0) * c * dT / L. So the ice balance gives
the porosity drop:

    phi0 - phi+ = (1-phi0) * c * dT / L

The density ratio cancels in the drop. It remains only in E. With rho_i = rho_w, theta- = 0
and q- = 0, the speed is eq. C17 of Meyer and Hewitt in dimensional form. Their eq. 38 is
printed without c. With dT = 0, the speed is [q]/[theta].

Conduction spreads the warming ahead of the front. It does not change these results in the
travelling wave, because the conductive flux is zero at both limits of the balance.

The state behind the front comes from q(theta+, phi+) = R. Water that the law holds
immobile is part of theta, so it is in the jump. If phi+ is zero or less, the pores freeze
shut and the front stops. No front exists then, so the functions raise ValueError.

The law is an argument. This module does not import the CFM.
'''

from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class FirnConstants:
    '''The constants of the balances, in SI units.'''
    rho_ice: float
    rho_water: float
    specific_heat: float # of ice
    latent_heat: float


@dataclass(frozen=True)
class Front:
    '''The state behind a front, and its speed [m/s].'''
    speed: float
    water_behind: float # theta+
    porosity_behind: float # phi+


def porosity_drop(porosity_ahead, constants, cold_degrees):
    '''phi0 - phi+. It does not depend on the densities. cold_degrees is Tm - T, so it is not negative.'''
    if cold_degrees < 0:
        raise ValueError(f'cold_degrees must not be negative: {cold_degrees}')
    return (1 - porosity_ahead) * constants.specific_heat * cold_degrees / constants.latent_heat


def freezing_water(porosity_ahead, constants, cold_degrees):
    '''E: the water that freezes to warm one volume of firn to the melting temperature.'''
    return constants.rho_ice / constants.rho_water * porosity_drop(porosity_ahead, constants, cold_degrees)


def porosity_behind_front(porosity_ahead, constants, cold_degrees):
    '''phi+. Raises ValueError if the pores freeze shut.'''
    porosity = porosity_ahead - porosity_drop(porosity_ahead, constants, cold_degrees)
    if porosity <= 0:
        raise ValueError(f'The pores freeze shut: the porosity behind the front is {porosity}')
    return porosity


def front_speed(flux_behind, water_behind, flux_ahead=0.0, water_ahead=0.0, frozen_water=0.0):
    '''V = [q] / ([theta] + E). Give E as frozen_water, from freezing_water().'''
    return (flux_behind - flux_ahead) / (water_behind - water_ahead + frozen_water)


def solve_front(flux_law, flux, porosity_ahead, constants, cold_degrees=0.0, saturated_fraction=1.0):
    '''
    The front for a general law, from a root solve of q(theta+, phi+) = flux.

    flux_law(theta, porosity) gives q [m/s]. It must increase with theta up to saturation,
    and it must be less than flux at theta = 0. It must also be convex, so that the front is
    one shock. It can be flat at zero first, as with immobile water. The largest water content is
    saturated_fraction * phi+. Raises ValueError if the law cannot carry the flux there.
    '''
    porosity = porosity_behind_front(porosity_ahead, constants, cold_degrees)
    water_max = saturated_fraction * porosity
    if flux_law(water_max, porosity) < flux:
        raise ValueError(f'The law cannot carry the flux {flux} at the water content {water_max}')
    water = brentq(lambda theta: flux_law(theta, porosity) - flux, 0.0, water_max, xtol=1e-14, rtol=1e-12)
    frozen_water = freezing_water(porosity_ahead, constants, cold_degrees)
    return Front(front_speed(flux, water, frozen_water=frozen_water), water, porosity)


def power_law_saturation(flux, conductivity, porosity):
    '''S+ = (R / (K * phi^3))^(1/2), for q = K * phi^3 * S^2.'''
    return (flux / (conductivity * porosity**3))**0.5


def power_law_front(flux, conductivity, porosity_ahead, constants, cold_degrees=0.0):
    '''The closed form of solve_front for q = K * phi^3 * S^2. With dT = 0, V = R/(phi*S).'''
    porosity = porosity_behind_front(porosity_ahead, constants, cold_degrees)
    saturation = power_law_saturation(flux, conductivity, porosity)
    if saturation > 1:
        raise ValueError(f'The law cannot carry the flux {flux}: the saturation is {saturation}')
    water = porosity * saturation
    frozen_water = freezing_water(porosity_ahead, constants, cold_degrees)
    return Front(front_speed(flux, water, frozen_water=frozen_water), water, porosity)
