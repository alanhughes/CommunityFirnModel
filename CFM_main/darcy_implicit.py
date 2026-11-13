#!/usr/bin/env python
'''
Experiment: an implicit gravity step for the Darcy scheme (option "implicit_gravity" of the
config key flow_step_darcy). It replaces the limiting-flux loop of darcyscheme in melt.py.
It has no capillary flow.

Caution: the step is stable at any sub-step, but darcyscheme removes the melt of a sub-step
before the flow of that sub-step. Use a dtsub_darcy of 3600 s or less. A sub-step of a day can
give wrong runoff and refreezing when melt reaches an ice lens (see the docstring of darcyscheme).
'''

import math
import numpy as np
from scipy.optimize import brentq


def _krel_vG_scalar(saturation, m):
    '''Relative conductivity of van Genuchten-Mualem for one node (as krel_vG in darcy_funcs.py).'''
    return math.sqrt(saturation) * (1. - (1. - saturation ** (1. / m)) ** m) ** 2


def water_after_outflow(water, irreducible, saturated, rate, krel_law, krel_parameter):
    '''
    Backward Euler for one node that loses water at the rate K: find x in

        x + rate * k_r((x - irreducible) / (saturated - irreducible)) = water

    Parameters
    ----------
    water : float
        Water in the node before the outflow [m]: the content at the start of the sub-step plus
        the inflow.
    irreducible, saturated : float
        Water in the node at the irreducible content and at saturation [m].
    rate : float
        Largest possible outflow in the sub-step [m]: the sub-step times the saturated
        conductivity (times the slope, for lateral runoff).
    krel_law : str
        'vanGenuchten' or 'power_law'.
    krel_parameter : float
        m of van Genuchten, or the exponent of the power law.

    Returns
    -------
    x : float
        Water in the node after the outflow [m]. It can be more than `saturated`: the caller
        moves that excess.
    solves : int
        1 if the function did a root solve or used the closed form, 0 if not.
    '''
    if water <= irreducible or rate <= 0.:
        return water, 0
    if water - rate >= saturated:
        return water - rate, 0     # the node stays full: it loses water at the saturated rate
    span = saturated - irreducible
    if krel_law == 'power_law' and krel_parameter == 2.:
        mobile = water - irreducible
        return irreducible + 2. * mobile / (1. + math.sqrt(1. + 4. * rate * mobile / span ** 2)), 1

    if krel_law == 'power_law':
        def residual(x):
            return x + rate * ((x - irreducible) / span) ** krel_parameter - water
    else:
        def residual(x):
            return x + rate * _krel_vG_scalar((x - irreducible) / span, krel_parameter) - water
    upper = min(water, saturated)
    if residual(upper) <= 0.:
        return upper, 1
    return brentq(residual, irreducible, upper, xtol=1e-16, rtol=1e-10), 1


def implicit_gravity_sweep(lwc, capacity, irreducible, saturated, rate, krel_law, krel_parameter,
                           surface_input, blocked, held):
    '''
    One sub-step of gravity flow, implicit in time, with upwind conductivity. The sweep goes from
    the top node down. The outflow of each node is the conductivity at its water content at the
    end of the sub-step. Then a second pass, from the bottom up, moves water above the capacity
    of a node to the node above it.

    Parameters
    ----------
    lwc : ndarray
        Water in each node at the start of the sub-step [m].
    capacity : ndarray
        Largest water content of each node [m].
    irreducible, saturated : ndarray
        Water at the irreducible content and at saturation [m].
    rate : ndarray
        Sub-step times the saturated conductivity [m].
    krel_law : str
        'vanGenuchten' or 'power_law'.
    krel_parameter : ndarray or float
        m of van Genuchten for each node, or the exponent of the power law.
    surface_input : float
        Water that enters the top node in the sub-step [m].
    blocked : ndarray of bool
        True for a node with no outflow.
    held : ndarray
        Water that the cold content of each node freezes in this sub-step [m]. It does not flow.

    Returns
    -------
    lwc_new : ndarray
        Water in each node at the end of the sub-step [m].
    surface_excess : float
        Water that the column cannot take [m].
    solves : int
        Number of root solves.
    '''
    ncv = len(lwc)
    lwc_new = lwc.copy()
    per_node = np.ndim(krel_parameter) > 0
    mobile_nodes = np.flatnonzero((lwc - held > irreducible) & ~blocked)
    solves = 0
    inflow = surface_input
    j = 0
    while j < ncv:
        if inflow <= 0.:
            # no water arrives from above: go to the next node that has mobile water
            k = np.searchsorted(mobile_nodes, j)
            if k == len(mobile_nodes):
                break
            j = mobile_nodes[k]
        water = lwc[j] + inflow
        if blocked[j]:
            out = 0.
        else:
            frozen = min(water, held[j])
            parameter = krel_parameter[j] if per_node else krel_parameter
            left, solved = water_after_outflow(water - frozen, irreducible[j], saturated[j], rate[j],
                                               krel_law, parameter)
            solves += solved
            out = water - frozen - left
        lwc_new[j] = water - out
        inflow = out
        j += 1

    # water above the capacity of a node goes back to the node above it
    limit = np.maximum(capacity, lwc)
    too_full = np.flatnonzero(lwc_new > limit)
    excess = 0.
    if len(too_full) > 0:
        for j in range(too_full[-1], -1, -1):
            lwc_new[j] += excess
            excess = max(0., lwc_new[j] - limit[j])
            lwc_new[j] -= excess
            if excess == 0. and j <= too_full[0]:
                break
    return lwc_new, excess, solves


def implicit_lateral_runoff(lwc, irreducible, saturated, rate, krel_law, krel_parameter, nodes):
    '''
    Lateral runoff of runoffDarcy (darcy_funcs.py), implicit in time: the conductivity is that
    of the water content at the end of the sub-step. `rate` is the sub-step times the saturated
    conductivity times the slope [m]. Returns the runoff of each node [m] (zero outside `nodes`).
    '''
    runoff = np.zeros(len(lwc))
    per_node = np.ndim(krel_parameter) > 0
    for j in nodes:
        parameter = krel_parameter[j] if per_node else krel_parameter
        left, _ = water_after_outflow(lwc[j], irreducible[j], saturated[j], rate[j], krel_law, parameter)
        runoff[j] = lwc[j] - min(left, lwc[j])
    return runoff


def nodes_with_lateral_runoff(imptop, saturation, ncv):
    '''
    Nodes that can have lateral runoff, by the rule of darcyscheme: the node above each
    impermeable lens, each node of a stack at 0.95 of saturation or more above the lens, and the
    node above that stack.
    '''
    nodes = []
    for lens_top in imptop:
        j = lens_top - 1
        if j < 0:
            continue
        nodes.append(j)
        while j >= 0 and saturation[j] >= 0.95:
            j -= 1
            if j >= 0:
                nodes.append(j)
    return np.unique(np.array(nodes, dtype=int))
