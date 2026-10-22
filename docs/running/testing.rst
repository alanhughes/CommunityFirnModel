.. _testing-page:

Testing the CFM
===============

The CFM has a test suite in the folder ``tests/`` of the repository. It uses `pytest <https://pytest.org>`_.

Run the tests
-------------

Run the tests from the repository root, not from ``CFM_main/``:

.. code-block:: bash

	pip install -r requirements.txt
	python -m pytest

The file ``pytest.ini`` in the repository root sets the test folder and the markers.

The CFM modules in ``CFM_main/`` are flat scripts, not an installable package. The tests put ``CFM_main/`` on ``sys.path`` and run each test in that folder. You do not have to do this yourself.

Test groups
-----------

Each test has one marker. To select a group, use ``-m``:

.. code-block:: bash

	python -m pytest -m "unit or scheme"    # the fast tests
	python -m pytest -m conservation        # the short model runs

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Marker
     - What it tests
     - Run time
   * - ``unit``
     - Single functions and equations
     - Seconds
   * - ``scheme``
     - One meltwater scheme on a synthetic column
     - Seconds
   * - ``conservation``
     - Mass and energy budgets over a short model run
     - Minutes

What the tests check
--------------------

A test file can contain tests of more than one group. The sections below list the tests by group.

Unit tests (``unit``)
~~~~~~~~~~~~~~~~~~~~~

A unit test calls one function with a small, specified input. Where a published equation or an analytic solution exists, the test compares the result with it.

In ``test_hl_analytic.py``, ``test_liquid_water.py`` and ``test_densification.py``, the test file calculates the reference values from the published equations, with literal constants. These tests do not call the CFM function under test to get the reference, and they do not import ``constants.py``. Thus an error in the code cannot also be in the reference. A change to the code that changes a result makes the test fail, so a change to the physics must also change the test. This makes an accidental change less likely.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``test_hl_analytic.py``
     - ``hl_analytic`` (the spin-up profile) against the Herron and Langway (1980) density and age equations, in zone 1 and zone 2. A node on the boundary between the two zones stays in the profile.
   * - ``test_liquid_water.py``
     - The liquid-water functions of the **bucket** and **darcy** schemes against the equations in Verjans et al. (2019): water-holding capacity (Coléou and Lesaffre, 1998), saturated hydraulic conductivity (Calonne et al., 2012), van Genuchten parameters (Yamaguchi et al., 2012), Mualem relative conductivity, and Zuo and Oerlemans (1996) runoff. The Darcy flux solvers stay within the available water and pore space.
   * - ``test_solvers.py``
     - The tridiagonal solver against ``scipy.linalg.solve_banded``. ``transient_solve_TR`` against the erfc solution for a step change of temperature at the surface of a semi-infinite slab. The four meltwater solvers: energy conservation with latent heat, temperature at or below 0 °C, mass conservation, agreement with ``transient_solve_TR`` in a dry column, and the position of a refreezing front. The ``ahc`` energy and front tests are expected failures (see :doc:`../extras/meltwater_solver`).
   * - ``test_densification.py``
     - The **physRho** laws ``HLdynamic``, ``KuipersMunneke2015`` and ``Crocus`` against the equations in Stevens et al. (2020), with both accumulation inputs: **bdot_type** ``instant`` and ``mean`` (see :doc:`json`). The tests give different values to inputs that the code can mix up: the instant and the mean accumulation rate, the values at different time steps, and the node temperature and the mean annual temperature. If the code reads the wrong input, the result changes and the test fails. ``HLdynamic``: Herron and Langway (1980). ``KuipersMunneke2015``: Arthern et al. (2010) with the Kuipers Munneke et al. (2015) multipliers, the lower limit of 0.25 on the multipliers, and the mean annual temperature in the grain-growth term (Steger et al., 2017). ``Crocus``: the viscosity of Vionnet et al. (2012), with the liquid-water factor of Verjans et al. (2019). At exactly 550 kg m-3, the code uses the zone 2 equation. Two tests record the present behaviour of the code where it differs from the papers. At 450 kg m-3 and above, ``Crocus`` uses the zone 2 rate of ``KuipersMunneke2015``. With **bdot_type** ``stress``, ``KuipersMunneke2015`` uses 10 times the node mass in place of the accumulation rate. A test that the ``stress`` rate does not depend on the node thickness is an expected failure.
   * - ``test_enthalpy_solver.py``
     - The ``enthalpy`` solver with liquid water in a layer below 0 °C. The solver refreezes this water with its latent heat.
   * - ``test_melt_diagnostics.py``
     - The record of a failed meltwater check, with and without **melt_strict** (see :doc:`../extras/melt_checks`).
   * - ``test_fixtures.py``
     - The ``synthetic_column`` fixture gives the state arrays that the meltwater schemes read.

Scheme tests (``scheme``)
~~~~~~~~~~~~~~~~~~~~~~~~~

A scheme test calls one meltwater scheme once, on a synthetic column (see `Fixtures for new tests`_). It does not run the model.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``test_melt_schemes.py``
     - ``bucket`` and ``darcyscheme``: the melted thickness ``dh_melt`` agrees with the melt mass divided by the density. ``darcyscheme`` gives the two interface nodes to the flux solvers, keeps each flux within the available water and pore space, and uses bisection when the first guess is far from equilibrium.
   * - ``test_fixtures.py``
     - ``bucket`` closes its water budget on a synthetic column and passes all its checks in strict mode. ``darcyscheme`` runs on a synthetic column and passes its checks.

Conservation tests (``conservation``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A conservation test runs the model (spin-up and ``time_evolve()``) for two years of the example forcing (see `Fixtures for new tests`_). It checks that the run completes, and some tests also check the water budget.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``test_short_runs.py``
     - The model completes a run with each of these configurations: **MELT** true with **SUBLIM** true and with **SUBLIM** false; **darcy**, **prefsnowpack** and **resingledomain**, each with output from 1980 and from the first step.
   * - ``test_water_budget.py``
     - The water budget for each of the four **liquid** schemes, with the DYE-2 forcing. Melt plus rain over the run agrees with refreezing plus runoff plus the liquid water in the column at the end.
   * - ``test_fixtures.py``
     - A short run with **melt_strict** true completes.

Fixtures for new tests
----------------------

The file ``tests/conftest.py`` gives two fixtures. The helpers are in ``tests/cfm_testing.py``.

``synthetic_column``
  Makes a small firn column with the state arrays that the meltwater schemes read. The density, temperature and liquid water content can be a scalar, an array, or a function of depth. The configuration starts from ``example_df.json``. Call a scheme on the column directly:

  .. code-block:: python

	import melt

	def test_bucket_refreezes(synthetic_column):
	    column = synthetic_column(n=60, rho_profile=lambda z: 350.0 + 40.0 * z,
	                              T_profile=265.0, melt=0.02,
	                              config={'melt_strict': True})
	    melt.bucket(column, 0)
	    assert column.melt_diagnostics['refrozen'] > 0.0

``short_config``
  Makes a short model run from ``example_df.json``. It cuts the forcing to a few years and writes the output to a temporary folder. Keyword arguments replace .json keys. Call ``.run()`` to run the spin-up and ``time_evolve()``:

  .. code-block:: python

	def test_short_run_completes_in_strict_mode(short_config):
	    run = short_config(years=2, melt_strict=True)
	    run.run()
	    assert run.results_path.exists()

Continuous integration
----------------------

The GitHub Actions workflow ``.github/workflows/tests.yml`` runs the test suite on each push to ``main`` and on each pull request. It has two jobs, which run in parallel:

- **Fast tests**: ``python -m pytest -m "unit or scheme"``.
- **Full tests**: ``python -m pytest -m conservation``.

To run the workflow manually, use **Run workflow** in the **Actions** tab of the repository on GitHub.

**References**

*Arthern, R. J., Vaughan, D. G., Rankin, A. M., Mulvaney, R., and Thomas, E. R. (2010), In situ measurements of Antarctic snow compaction compared with predictions of models, J. Geophys. Res., 115, F03011, doi:10.1029/2009JF001306.*

*Calonne, N., Geindreau, C., Flin, F., Morin, S., Lesaffre, B., Rolland du Roscoat, S., and Charrier, P. (2012), 3-D image-based numerical computations of snow permeability: links to specific surface area, density, and microstructural anisotropy, The Cryosphere, 6, 939–951, doi:10.5194/tc-6-939-2012.*

*Coléou, C., and Lesaffre, B. (1998), Irreducible water saturation in snow: experimental results in a cold laboratory, Ann. Glaciol., 26, 64–68, doi:10.3189/1998AoG26-1-64-68.*

*Herron, M. M., and Langway, C. C. (1980), Firn densification: an empirical model, J. Glaciol., 25(93), 373–385, doi:10.3189/S0022143000015239.*

*Kuipers Munneke, P., Ligtenberg, S. R. M., Noël, B. P. Y., Howat, I. M., Box, J. E., Mosley-Thompson, E., McConnell, J. R., Steffen, K., Harper, J. T., Das, S. B., and van den Broeke, M. R. (2015), Elevation change of the Greenland Ice Sheet due to surface mass balance and firn processes, 1960–2014, The Cryosphere, 9, 2009–2025, doi:10.5194/tc-9-2009-2015.*

*Steger, C. R., Reijmer, C. H., van den Broeke, M. R., Wever, N., Forster, R. R., Koenig, L. S., Kuipers Munneke, P., Lehning, M., Lhermitte, S., Ligtenberg, S. R. M., Miège, C., and Noël, B. P. Y. (2017), Firn meltwater retention on the Greenland Ice Sheet: a model comparison, Front. Earth Sci., 5, 3, doi:10.3389/feart.2017.00003.*

*Stevens, C. M., Verjans, V., Lundin, J. M. D., Kahle, E. C., Horlings, A. N., Horlings, B. I., and Waddington, E. D. (2020), The Community Firn Model (CFM) v1.0, Geosci. Model Dev., 13, 4355–4377, doi:10.5194/gmd-13-4355-2020.*

*Verjans, V., Leeson, A. A., Stevens, C. M., MacFerrin, M., Noël, B., and van den Broeke, M. R. (2019), Development of physically based liquid water schemes for Greenland firn-densification models, The Cryosphere, 13, 1819–1842, doi:10.5194/tc-13-1819-2019.*

*Vionnet, V., Brun, E., Morin, S., Boone, A., Faroux, S., Le Moigne, P., Martin, E., and Willemet, J.-M. (2012), The detailed snowpack scheme Crocus and its implementation in SURFEX v7.2, Geosci. Model Dev., 5, 773–791, doi:10.5194/gmd-5-773-2012.*

*Yamaguchi, S., Watanabe, K., Katsushima, T., Sato, A., and Kumakura, T. (2012), Dependence of the water retention curve of snow on snow characteristics, Ann. Glaciol., 53(61), 6–12, doi:10.3189/2012AoG61A001.*

*Zuo, Z., and Oerlemans, J. (1996), Modelling albedo and specific balance of the Greenland ice sheet: calculations for the Søndre Strømfjord transect, J. Glaciol., 42(141), 305–317, doi:10.3189/S0022143000004160.*
