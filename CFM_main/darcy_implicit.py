#!/usr/bin/env python
'''
An implicit gravity step for the Darcy scheme (option "implicit_gravity" of the
config key flow_step_darcy). It replaces the limiting-flux loop of darcyscheme in melt.py.
The gravity sweep has no capillary flow. The config key capillary_implicit_darcy adds the capillary
flow of Meyer and Hewitt (2017), as one linear step after the sweep (capillary_implicit_step).

Caution: the step is stable at any sub-step, but darcyscheme removes the melt of a sub-step
before the flow of that sub-step. Use a dtsub_implicit_darcy of 3600 s or less. A sub-step of a day can
give wrong runoff and refreezing when melt reaches an ice lens (see the docstring of darcyscheme).
'''

import math
import numpy as np
from scipy.linalg import solve_banded
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
    excess = move_water_above_capacity_up(lwc_new, limit)
    return lwc_new, excess, solves


def move_water_above_capacity_up(lwc_new, limit):
    '''
    Move the water above the limit of a node to the node above it, from the bottom up. It changes
    `lwc_new` in place and returns the water that leaves the top node [m]. `limit` is the largest
    water content of each node [m].
    '''
    too_full = np.flatnonzero(lwc_new > limit)
    excess = 0.
    if len(too_full) > 0:
        for j in range(too_full[-1], -1, -1):
            lwc_new[j] += excess
            excess = max(0., lwc_new[j] - limit[j])
            lwc_new[j] -= excess
            if excess == 0. and j <= too_full[0]:
                break
    return excess


def capillary_interface_terms(conductivity, thickness, dtsub, capillary_head):
    '''
    Water that passes each interface in one sub-step for a unit difference of saturation [m]:

        G = dtsub * K_face * h_c / dz_face

    K_face is the harmonic mean of the saturated conductivity of the two nodes, weighted by their
    thickness. It is zero if one conductivity is zero. dz_face is the distance between the node
    centres. The capillary pressure is p_c = c / S, so the capillary flux is -K_sat h_c dS/dz,
    with h_c = c / (rho_w g) (Meyer and Hewitt, 2017).

    Parameters
    ----------
    conductivity : ndarray
        Saturated hydraulic conductivity of each node [m s-1].
    thickness : ndarray
        Node thickness [m].
    dtsub : float
        Sub-step [s].
    capillary_head : float
        h_c [m].

    Returns
    -------
    terms : ndarray
        One value for each interface (length n - 1) [m].
    '''
    k_upper, k_lower = conductivity[:-1], conductivity[1:]
    dz_upper, dz_lower = thickness[:-1], thickness[1:]
    both_open = (k_upper > 0.) & (k_lower > 0.)
    resistance = np.divide(dz_upper, k_upper, out=np.zeros(len(k_upper)), where=both_open) \
        + np.divide(dz_lower, k_lower, out=np.zeros(len(k_lower)), where=both_open)
    k_face = np.divide(dz_upper + dz_lower, resistance, out=np.zeros(len(k_upper)), where=both_open)
    return dtsub * k_face * capillary_head / (0.5 * (dz_upper + dz_lower))


CAPILLARY_WATER_THRESHOLD = 1e-14   # [m of water] Water that the capillary solve puts in a dry node of floor zero, below this, is a numerical tail of the diffusion (it goes down to 1e-85 m). Without the reset, every dry node of a column wakes the gravity sweep. The value is far below any physical amount of water, and far above the round-off of a wet node (about 1e-18 m at 1e-2 m). A value from 1e-16 to 1e-12 m gives practically the same results.


def capillary_implicit_step(lwc, floor, saturated, capacity, interface_terms, closed):
    '''
    One sub-step of capillary flow, backward Euler, for the saturation of the mobile water. The
    saturation is S = (water - floor) / (saturated - floor). Water crosses an interface from the
    wetter node to the drier node, G (S_upper - S_lower). The step is one linear tridiagonal
    solve, with no iteration.

    Each node is one of three kinds:

    - Active: the water is at or above the floor. It is an unknown of the solve.
    - Sink: the water is below the floor. The solve holds it at S = 0 for the whole sub-step. It
      takes the water that its active neighbours give it, and that water pays its floor first. Two
      sinks exchange no water. The floor pass (below) handles a sink that the solve fills above
      its floor.
    - Closed: an impermeable node, or a node with no span (a floor at or above saturation). It
      exchanges no water.

    A dry node with a floor of zero is active, with S = 0.

    Floor pass. The solve holds a sink at S = 0, so a sink drains the mobile water of the wet nodes
    next to it. A sink can end above its floor, with a pile of water. After the solve, for each
    sink with water above its floor, the excess (water minus floor) goes away from the nodes that
    gave it:

    - Down, if the sink received water from above. Up, if it received water from below. If it
      received water from both sides, the excess is split in proportion to the water from each
      side.
    - In that direction, the excess pays the floor deficit of each next node. The walk stops at
      a closed node, an active node, a sink that is at its floor, the end of the column, or when
      the excess is spent. The walk can go past the window of the solve.
    - The water that the walk moves leaves the first sink. The rest stays in it. It is above the
      floor, so it is mobile in the next sub-step.

    The pass moves water between nodes, so it keeps the water exactly. It acts on the whole
    column, not only on the window.

    The solve covers only the part of the column that can change. A sink or a closed node passes
    no water on in the sub-step. So the solve takes each group of active nodes (between sinks and
    closed nodes) that has mobile water, with the sinks next to it, from the first to the last of
    these nodes. The other nodes keep their water. This is exact. A column with no mobile water
    returns at once.

    A dry node of floor zero in a group takes a tail of water from the diffusion. If a node had no
    water and takes less than CAPILLARY_WATER_THRESHOLD, its water is set to zero and added to the
    deepest node above it that keeps its water, with no closed node between them. A closed node is
    never that node. If there is no such node (for example, the tail rises to the top of the
    solve, under a closed node or at the surface), the tail keeps its water.

    Water above the capacity of a node goes to the node above it, as in implicit_gravity_sweep, but
    inside each stretch of nodes between closed nodes. Water that reaches the top node of a stretch
    under a closed node does not pass it. It goes back down in the same stretch, into the room
    that the nodes below gave. Only the stretch that reaches the surface gives surface excess. If
    a stretch has no room, the rest stays in its lowest node. This occurs only by round-off,
    because the solve keeps the water of each stretch.

    Parameters
    ----------
    lwc : ndarray
        Water in each node at the start of the step [m].
    floor : ndarray
        Water that does not flow: the irreducible water plus the water that the cold content
        freezes [m].
    saturated : ndarray
        Water in each node at saturation [m].
    capacity : ndarray
        Largest water content of each node [m].
    interface_terms : ndarray
        G of each interface, from capillary_interface_terms (length n - 1) [m]. A later term for
        the grain size of each node can change these values or add to the matrix here.
    closed : ndarray of bool
        True for a node that exchanges no water (the impermeable nodes). A node with no span is
        also closed.

    Returns
    -------
    lwc_new : ndarray
        Water in each node at the end of the step [m].
    surface_excess : float
        Water that the top node cannot take [m].
    '''
    span = saturated - floor
    closed = closed | (span <= 0.)
    active = ~closed & (lwc >= floor)
    terms = np.where(closed[:-1] | closed[1:], 0., interface_terms)

    mobile = np.where(active, lwc - floor, 0.)   # span times S at the start [m]
    if not np.any(mobile > 0.):
        return lwc.copy(), 0.   # no water to move: S = 0 everywhere is the exact solution

    ### The nodes of the solve: the groups of active nodes with mobile water, and the sinks next to them
    group = np.cumsum(~active)                                        # the same value for the active nodes of one group
    in_wet_group = active & np.isin(group, np.unique(group[mobile > 0.]))
    next_to_group = ~closed & ~active & (np.concatenate(([False], in_wet_group[:-1]))
                                         | np.concatenate((in_wet_group[1:], [False])))
    needed = np.flatnonzero(in_wet_group | next_to_group)
    lo, hi = needed[0], needed[-1] + 1
    span_w, active_w, terms_w, mobile_w = span[lo:hi], active[lo:hi], terms[lo:hi - 1], mobile[lo:hi]

    nw = hi - lo
    upper_terms = np.where(active_w[:-1], terms_w, 0.)    # coupling of node i to node i+1
    lower_terms = np.where(active_w[1:], terms_w, 0.)     # coupling of node i+1 to node i
    face_above = np.concatenate(([0.], terms_w))
    face_below = np.concatenate((terms_w, [0.]))
    banded = np.zeros((3, nw))
    banded[0, 1:] = -upper_terms
    banded[1] = np.where(active_w, span_w + face_above + face_below, 1.)
    banded[2, :-1] = -lower_terms
    saturation = solve_banded((1, 1), banded, mobile_w)   # S of each node at the end; zero for sink and closed nodes

    flux = terms_w * (saturation[:-1] - saturation[1:])   # water that crosses each interface downward [m]
    lwc_new = lwc.copy()
    window = lwc_new[lo:hi]                                # a view: changes here change lwc_new
    window[:-1] -= flux
    window[1:] += flux

    ### Water above the floor of a sink: it goes away from the nodes that gave it, and pays the floors of the sinks there
    sink = ~closed & ~active
    flux_all = np.zeros(len(lwc) - 1)
    flux_all[lo:hi - 1] = flux
    for i in np.flatnonzero(sink & (lwc_new > floor)):
        excess = lwc_new[i] - floor[i]
        from_above = max(flux_all[i - 1], 0.) if i > 0 else 0.
        from_below = max(-flux_all[i], 0.) if i < len(lwc) - 1 else 0.
        if from_above + from_below <= 0.:
            continue
        down = excess * from_above / (from_above + from_below)   # the share of the water that came from above
        up = excess - down
        for direction, remaining in ((1, down), (-1, up)):
            j = i + direction
            while remaining > 0. and 0 <= j < len(lwc) and sink[j] and lwc_new[j] < floor[j]:
                put = min(remaining, floor[j] - lwc_new[j])
                lwc_new[j] += put
                lwc_new[i] -= put
                remaining -= put
                j += direction

    ### Tail of the diffusion in dry nodes of floor zero: set it back to zero in the node above
    dry = (lwc[lo:hi] <= 0.) & (floor[lo:hi] <= 0.) & ~closed[lo:hi]
    tail = np.flatnonzero(dry & (window < CAPILLARY_WATER_THRESHOLD))
    if len(tail) > 0:
        keeps = np.flatnonzero(~closed[lo:hi] & (~dry | (window >= CAPILLARY_WATER_THRESHOLD)))   # a closed node is never a host
        host = np.searchsorted(keeps, tail) - 1                 # the deepest node above that keeps its water
        closed_count = np.cumsum(closed[lo:hi])
        has_host = host >= 0
        host[~has_host] = 0
        movable = has_host & (closed_count[keeps[host]] == closed_count[tail])   # no closed node between
        tail, host = tail[movable], keeps[host[movable]]
        np.add.at(window, host, window[tail])
        window[tail] = 0.

    ### Water above the capacity of a node: up inside each stretch between closed nodes
    limit = np.maximum(capacity, lwc)
    surface_excess = 0.
    if np.any(lwc_new > limit):
        edges = np.flatnonzero(closed)
        starts = np.concatenate(([0], edges + 1))
        stops = np.concatenate((edges, [len(lwc)]))
        for first, stop in zip(starts, stops):
            if stop <= first or not np.any(lwc_new[first:stop] > limit[first:stop]):
                continue
            stretch = lwc_new[first:stop]   # a view
            excess = move_water_above_capacity_up(stretch, limit[first:stop])
            if first == 0:
                surface_excess = excess
                continue
            for j in range(stop - first):   # back down into the room of this stretch
                put = min(excess, max(0., limit[first + j] - stretch[j]))
                stretch[j] += put
                excess -= put
                if excess <= 0.:
                    break
            if excess > 0.:
                stretch[-1] += excess   # no room left: the lowest node of the stretch keeps it
    return lwc_new, surface_excess


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
