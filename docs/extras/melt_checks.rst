Meltwater checks and strict mode
--------------------------------

The **bucket** and **darcy** meltwater schemes (``bucket`` and ``darcyscheme`` in ``melt.py``) do checks at the end of each call:

- A **budget** check. The water at the start of the call (liquid water in the column, plus melt, plus rain) must agree with the water at the end (liquid water, plus refreezing, plus runoff).
- A **cold-layer** check. No layer below 0 °C can hold liquid water.

The **prefsnowpack** and **resingledomain** schemes do not do these checks. Before the model year in **bucket_until_year**, the **darcy**, **prefsnowpack** and **resingledomain** options use the **bucket** scheme, which does the checks.

The checks
==========

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Scheme
     - Check
     - What it checks
   * - ``bucket``
     - ``cold_1``
     - After percolation and the first refreezing step: no layer below 0 °C holds liquid water.
   * - ``bucket``
     - ``budget_1``
     - After runoff: the water budget closes to 1e-3 m w.e.
   * - ``bucket``
     - ``cold_2``
     - After the temperature correction and the second refreezing step: no layer below 0 °C holds liquid water.
   * - ``bucket``
     - ``budget_2``
     - At the end of the call: the water budget closes to 1e-5 m w.e.
   * - ``darcy``
     - ``budget``
     - At the end of the call (all sub-steps): the water budget closes to 1e-12 m w.e.
   * - ``darcy``
     - ``cold``
     - At the end of the call: no layer below 0 °C holds liquid water.

Before each cold-layer check, the scheme removes round-off liquid water from cold layers. If each cold layer holds less than 1e-9 m w.e., the scheme sets the liquid water of all cold layers to zero. It records the removed amount as ``cold_lwc_removed``. If one or more cold layers hold more than this, the scheme removes nothing and the check fails.

The result of a failed check
============================

The **melt_strict** key in the configuration file (see :doc:`../running/json`) sets what occurs when a check fails:

- **melt_strict** false (default). The scheme prints a message and the run continues.
- **melt_strict** true. The scheme raises ``melt.MeltCheckError`` and the run stops. The message gives the scheme, the check, the time step, and the recorded values, for example::

	MeltCheckError: bucket: check 'budget_1' failed at step 7: {'ok': False, 'residual': 0.002, 'threshold': 0.001}

Strict mode is useful for ensemble runs, because a run with a wrong water budget then does not complete.

The diagnostics record
======================

Each call of ``bucket`` or ``darcyscheme`` writes a new dictionary ``melt_diagnostics`` on the model object. It holds the results of that call only. It is not a time series.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Value
   * - ``scheme``
     - ``'bucket'`` or ``'darcy'``.
   * - ``step``
     - The time-step index ``iii``.
   * - ``water_in``
     - The water at the start of the call: liquid water, plus melt, plus rain [m w.e.].
   * - ``lwc_end``
     - The liquid water in the column at the end of the call [m w.e.].
   * - ``refrozen``
     - The water that refroze in the call [m w.e.].
   * - ``runoff``
     - The runoff in the call [m w.e.].
   * - ``checks``
     - One entry for each check, with the check name as key. Each entry has ``ok`` (True or False). A budget check also has ``residual`` and ``threshold`` [m w.e.]. A cold-layer check also has ``n_wet_cold_layers`` and ``cold_lwc_removed`` [m w.e.].

For example, after a call of ``bucket``::

	>>> model.melt_diagnostics['checks']['budget_2']
	{'ok': True, 'residual': 1.7e-17, 'threshold': 1e-05}

The limit of darcy at thin nodes
================================

In each sub-step, ``darcyscheme`` computes the flux at each interface as ``qlim*(1-exp(-dtsub*q0/qlim))``, from Hirashima et al. (2010). ``q0`` is the Darcy flux, which follows the permeability. ``qlim`` is the water that brings the two nodes to hydrostatic equilibrium. The sub-step gets shorter when the flux is near ``qlim``, down to **dtmin_darcy**.

Where the limiter acts for a long time, the error is in the water content. Take a steady flux ``R`` through uniform firn, with each sub-step at **dtmin_darcy**. Each sub-step then moves ``R*dtmin_darcy``, and the water content rises until ``qlim`` can carry it. The water content does not follow the permeability, and the wetting front is too slow. ``qlim`` falls with the node thickness, so thinner nodes give a larger error, not a smaller one. A change of a permeability law can then have no effect on the result.

For 400 kg m-3 firn and 1 mm grain radius at the default **dtmin_darcy** of 60 s, the error is more than 4 % below these node thicknesses. The values are calculated from the expression for ``qlim``, and ``test_darcy_front_speed.py`` measures the cases around them.

The thicknesses depend on the irreducible water content :math:`\theta_i`, the volume fraction of water that the firn holds against gravity. Only the water above :math:`\theta_i` can flow, so :math:`\theta_i` changes the effective saturation, the head and ``qlim``. The table uses :math:`\theta_i = 0.02`. To set this value, use **ColeouLesaffre** false and **IrrVal** 0.0387.

**IrrVal** is a fraction of the pore space, not a volume fraction of the node. The pore space, as a volume of water per volume of firn, is

.. math::

   \phi \frac{\rho_i}{\rho_w}, \qquad \phi = 1 - \frac{\rho}{\rho_i}.

At :math:`\rho` = 400 kg m-3, :math:`\phi` = 0.5638 and the pore space is 0.5170. So :math:`\theta_i = 0.02` needs **IrrVal** = 0.02 / 0.5170 = 0.0387.

The default is **ColeouLesaffre** true. :math:`\theta_i` then follows Coléou and Lesaffre (1998), in the form of Langen et al. (2017), Eqs. 3-4. The irreducible water as a mass fraction of the wet firn is

.. math::

   W_{mi} = 0.057 \frac{\rho_i - \rho}{\rho} + 0.017.

As a fraction of the pore space, this is

.. math::

   S_{wi} = \frac{W_{mi}}{1 - W_{mi}} \frac{\rho_i \rho}{\rho_w (\rho_i - \rho)},

and :math:`\theta_i = S_{wi}\,\phi\,\rho_i/\rho_w`. At :math:`\rho` = 400 kg m-3, :math:`W_{mi}` = 0.0907, :math:`S_{wi}` = 0.0707 and :math:`\theta_i` = 0.0366. That is almost twice the value of the table. So with the default, the node thicknesses differ from the table. They are not calculated here.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Flux (m s-1)
     - Node thickness (mm)
   * - 5e-8
     - 4
   * - 5e-7
     - 12
   * - 1e-6
     - 16

A shorter **dtmin_darcy** moves these thicknesses down, and a run then needs more sub-steps. To see if the limit acts in a run, compare it with a run at a shorter **dtmin_darcy**.
