# CFM test suite

Run the tests from the repository root:

```bash
pip install -r requirements.txt
python -m pytest
```

## Markers

Each test has one marker. Select a group with `-m`, for example `python -m pytest -m "unit or scheme"`. `pytest.ini` lists the markers.

| Marker | What it tests | Run time |
|---|---|---|
| `unit` | Single functions and equations | Seconds |
| `scheme` | One meltwater scheme on a synthetic column | Seconds |
| `conservation` | Mass and energy budgets over a short model run | Minutes |

## Fixtures

`conftest.py` gives these fixtures. The helpers are in `cfm_testing.py`.

- `synthetic_column`: makes a small firn column with the state attributes that the meltwater schemes read. Call a scheme on it directly, for example `melt.bucket(column, 0)`.
- `short_config`: makes a short model run from `example_df.json`, with the forcing cut to a few years and the output in a temporary folder. Call `.run()` to run it.

The CFM modules in `CFM_main/` are flat scripts, not an installable package. As a temporary workaround, the tests put `CFM_main/` on `sys.path` and run in that folder.

## Oracles

An oracle is a function in the test code that gives the exact answer to a simplified problem. It is not the truth, so a difference is an error only when the scheme and the oracle make the same assumptions about the quantity under test.

`oracles/kinematic_wave.py` gives a wetting front that moves down into uniform firn with no capillary gradients. Its docstring has the derivation, with distinct densities for ice and water.

- `solve_front`: the state behind the front and the front speed, for a general flux law. The test passes the law as `flux_law(theta, porosity)`, for example `hydrconducsat_Calonne` times `krel_vG`. The module does not import the CFM.
- `power_law_front`: the closed form for q = K φ³ S².
- `front_speed`: the speed from the states on the two sides of the front.

Both front functions take the cold degrees of the firn ahead of the front. They raise `ValueError` if the pores freeze shut, or if the law cannot carry the flux. `test_oracles.py` checks them against values calculated by hand.

`test_darcy_front_speed.py` compares `darcyscheme` with `solve_front`, for rain on dry firn at the melting temperature, with the flux law of the scheme. It calls `darcyscheme` directly, and it also counts the sub-steps that spread the input below the top node. To count them, it reads `i_input_bottom` from the frame that calls `vG_Yama_params`. So a new name for `i_input_bottom` fails the test with a `KeyError`.

- The plateau behind the front is the test of the law, because mass balance gives the speed from the plateau.
- The tolerance is 4 % at the default sub-steps. With a short fixed sub-step, the tolerance on the plateau is 0.5 %.
- The cases at thin nodes are strict `xfail` tests. There the sub-step stops at `dtmin_darcy`, and `qlim` sets the flux. Other tests show the cause, so change them with the fix.
- Two tests multiply K_sat by 10 in the scheme. At resolved nodes the plateau follows the law. At thin nodes it does not change, because the limiter caps the flux.
- The last tests set `Ksat_darcy` and `krel_darcy`, at 20 mm nodes. With both laws at `power_law` (n = 2), `viscosity_darcy` 1e-3 and the paper settings (`ColeouLesaffre` false, `IrrVal` 0, `theta_s_factor_darcy` 1), the oracle is `power_law_front`. With short fixed sub-steps the tolerance is 0.5 %. At the default sub-steps the plateau is 6.7 % high, because with k_r = S² the plateau goes as the square root of the flux. One mixed case, the power law for K_sat with the van Genuchten k_r, uses `solve_front`.

`test_darcy_implicit.py` checks the implicit gravity flow step of `darcyscheme` (`flow_step_darcy` "implicit_gravity", `darcy_implicit.py`).

- The unit tests check the root solve (`water_after_outflow`) against its equation, for van Genuchten and for the power law with exponents 2 (closed form) and 3 (`brentq`). They also check the limits and the sweep: mass, capacity, blocked nodes, the back-up of water, and the hold of cold nodes.
- The front tests run the cases that are strict `xfail` for the explicit step in `test_darcy_front_speed.py`, at a sub-step of 1800 s, with the same 4 % tolerance. They also run a column with nodes of 4 mm over nodes of 20 mm. They import the oracle functions from that file.
- The sub-step test compares 60 s, 1800 s and 3600 s on one column. The cold firn test compares the front with the kinematic wave with refreezing, at 1800 s and at one sub-step for each day. The second case sees the hold of cold nodes.
- The edge cases (lens at the surface, lens at node 1, no input, `ZuoOerlemans`, both `krel_darcy` laws, no lateral runoff) run with `melt_strict` on. The lens tests check an impermeable lens, and that a lens of one node and of five nodes pass the same water.
- The key tests check that the explicit step is the default (bit for bit), that a wrong `flow_step_darcy` or a `dtsub_implicit_darcy` of 0 s or below stops the run, that the implicit step uses `dtsub_implicit_darcy` only, the counters `substeps` and `root_solves`, and the message for a sub-step above 3600 s. The message is printed once in each run (a flag on the model instance).

`test_darcy_capillary.py` checks the capillary step of the implicit gravity flow (`capillary_implicit_darcy` "power_law", `darcy_implicit.py`).

- The unit tests check `capillary_interface_terms` and `capillary_implicit_step`: the harmonic mean, the water budget with the surface excess, no water through a closed node or a conductivity of zero, no exchange between nodes at the same saturation, the bounds of the new saturation, the uniform steady state, the decay of a cosine mode against the backward Euler factor, and the rules for a sink node (a node below its floor is held at S = 0 and its new water pays the floor first), and the reset of the small tail in dry nodes of floor zero, which never moves water into or across a closed node.
- The oracle is the travelling wave of constant rain on uniform firn at 0 C, with the power laws for K_sat and k_r and the paper settings. The front is at v t, and S = S_+ / (1 + exp(z/w)) with w = h_c / S_+ and K_sat S_+^2 = flux. The tests compare the front depth, the width of the front, the plateau and the largest profile error at node thicknesses of 4 mm to 0.1 m and sub-steps of 60 s and 600 s. The tolerances come from the measured errors, which are in the comments. The width is larger than the exact width because the gravity sweep adds numerical diffusion.
- The floor pass (the excess of a sink that the solve fills above its floor goes away from the node that gave it, and pays the floors of the sinks there, at most G of each face that it crosses) has an oracle in the tests, `active_set_step`: the same step with an active-set iteration. The scheme tests compare the front and the profile of S with the oracle for a wet zone above or below dry firn (a cold floor and a small floor), at 1800 s and 3600 s: the front within two nodes, the largest S beyond the front within 0.03, and a difference of S under 0.06. At 60 s the difference stays under 0.03. With a layer of low conductivity (1e-3 to 4e-5 of the others, not closed) below the wet zone, the water that passes the layer in 4 h at 1800 s stays within 0.5 mm of the oracle. The unit tests check the exact water budget on random columns with sinks and closed nodes, that the walk stops at a closed node, at a face of zero conductivity, at an active node and at a sink above or at its floor, goes up for water from below, goes past the window of the solve, and that a sink with water from both sides keeps its excess. A node a round-off below its floor is at its floor, not a sink. The window tests compare the step with the solve over the whole column followed by the floor pass.
- The edge cases of `test_darcy_implicit.py` run again with the capillary step. A lens test checks that no water passes an impermeable lens.
- The key tests check that "none" gives the arrays of no key (bit for bit), that the key has no effect with the explicit step (one message in each run), the message for a `krel_darcy` other than `power_law` with exponent 2, and that a wrong value of the key or a `capillary_pressure_darcy` of zero or below stops the run.

`test_darcy_permeability_laws.py` checks the K_sat and k_r laws against values calculated by hand. Its scheme tests show that each of `Ksat_darcy`, `Ksat_k0_darcy`, `viscosity_darcy`, `krel_darcy` and `krel_exponent_darcy` changes the water profile after one call of `darcyscheme`. For all of them except `krel_darcy`, the test also checks the direction: a smaller conductivity keeps the water higher.

`test_darcy_snapshot.py` checks the output `darcy_snapshot`, the state of `darcyscheme` partway through a step (`snapshot_fraction_darcy`).

- The snapshot does not change the run: the end state is the same, bit for bit, with and without it.
- The snapshot is the end state of a call that stops at `snapshot_time`. With no melt, that call takes the same sub-steps.
- In a model run, the snapshot is NaN in each step where `darcyscheme` does not run. The test counts the calls with a wrapper of `darcyscheme` in `firn_density_nospin`.
- Gridded output uses the depth of the snapshot, not the end-of-step depth.

`test_step_timing.py` checks the key `step_timing` (`step_timing.py`).

- A short run with the implicit flow step and `bucket_until_year` in the run gives the same datasets, bit for bit, with the key on and off. With the key off the model has no `step_timing_stages` and there is no `step_timing.csv`.
- `step_timing.csv` has one row for each step. The five parts add up to the step (wall and CPU time) within 1e-9 s. The counts of the schemes `darcy` and `bucket` equal the calls that a wrapper of the two functions counts. `substeps`, `root_solves` and `idle_substeps` are empty where the scheme does not give them. Each `darcy` and `bucket` row has a `budget_residual`.
- The run prints one line for each model year, and the wall time of the line equals the sum of its rows. With the monthly forcing (dates at the end of each month), no line of the two full years says "(part year)". `step_timing_stages` has the four stages in order, and `switch` is `None` when no step reaches `bucket_until_year`.
- `idle_substeps` (a count in `darcyscheme`, implicit step only) equals `substeps` on a dry column with no input, is 0 on a column with melt, and is absent with the explicit step. The test imports `edge_column` from `test_darcy_implicit.py`.
- A run that stops inside the loop (a `darcyscheme` that raises) keeps `loop_start` and `switch` as plain finite floats, and `loop_end` and `write_end` stay `None`. `step_timing.csv` keeps the rows of each full year before the stop.
- With `liquid: bucket`, `switch` is set at the first step at or after `bucket_until_year`, although no scheme changes there.
- With `MELT: false` each row has the scheme `nomelt` and an empty `melt_mwe`, and `switch` is `None`.
- The label "(part year)" comes from the number of steps: a `StepTimer` on a stub model with monthly steps at the end of each month marks a full year of 12 steps as full and a last year of 6 steps as part.
- `step_timing: "yes"` stops the run in the validation.

`test_melt_split.py` checks the option `melt_split_darcy` of `darcyscheme` (the melt of a step falls in a burst, implicit flow step only).

- The schedule tests check `_melt_schedule` on its own. The sum equals the melt (relative 1e-12). The burst is rounded up to whole sub-steps, and the rate is at or below the peak rate. The window is centred on `melt_peak_hour_darcy` to half a sub-step. It is moved to fit at the start and at the end of the step. A melt that the peak rate cannot give in the step gives the uniform rule. Zero melt gives zeros and `None`. A shorter last sub-step takes melt in proportion to its length.
- The scheme tests check that the water budget closes with a burst, that a snapshot before the burst shows no surface lowering (and a uniform run does) and that the surface lowering of the step is close to that of the uniform run, that no sub-step of the burst is counted as idle, that the default keys give the arrays of no key (bit for bit), and that a burst that fills the step gives the arrays of the uniform run.
- The validator and `darcyscheme` stop the run for "burst" with the explicit step, a `melt_peak_rate_darcy` of 0 or below, a `melt_peak_hour_darcy` outside 0 to 24, and an unknown `melt_split_darcy`. The validator does not stop for "burst" with the explicit step if the darcy scheme does not run (`liquid` is not `darcy`, or `MELT` is false).

`test_heat_substep.py` checks the argument `dt` of `heatDiff` and `refreezeDiff` (`diffusion.py`).

- `dt=None` gives the same arrays, bit for bit, as `dt=self.dt[iii]`. The test covers `heatDiff` and `refreezeDiff` with `transient_solve_enthalpy`.
- On a dry column, one step of 86400 s and two steps of 43200 s differ by less than 0.01 K.
- On a wet column, 48 steps of 1800 s with each of the four solvers keep the total mass to 1e-9 kg/m², keep `Tz` at or below 273.15 K and keep `LWC` at or above zero.
- On the same wet column, the column liquid water of 48 steps of 1800 s with `transient_solve_enthalpy` is within 0.2 kg/m² of one step of 86400 s.

`test_conduction_in_loop.py` checks the option `conduction_interval_darcy` ("substep" runs heat conduction once in each implicit sub-step, "step" runs it once for the whole step). The scheme tests use a synthetic column with the heat attributes (`Ts`, `melt_solver`, `total_count`, `T10m`).

- A count of the calls of `heatDiff` and `refreezeDiff`: one call in each sub-step, each with `dt` equal to the sub-step, and none after the scheme. A short run on daily steps (the step is a little longer than a day, so the last sub-step is short) checks that the calls of a step with water cover the step, and that a step with no water makes one call with `dt=None`. "step" makes no call in the scheme.
- The water budget closes at 1e-12, `refrozen_conduction` is above 0, and `refrozen_flow` plus `refrozen_conduction` equals `refrozen`. The layer total of `self.refrozen` equals `refrozen`.
- The ghost node. With "substep", the heat solve has a ghost node above node 0 (see `_conduct_with_ghost_node` in `melt.py`). It has almost no mass (1e-6 of the thickness of node 0, not 0, because the solvers divide by `dz`) and no water, it has the density of node 0 (a near-zero density would give a conductivity at the 1e-12 floor, which would dominate the face), it ends at the top face of node 0, and its temperature is the surface value `Ts[iii]` (capped at 0 °C). The solvers hold the ghost node fixed, so node 0 is a solved node: it conducts, and it freezes its water with the latent heat counted. A fixed node 0 would freeze all its water for free when the surface is cold, and each reset of its temperature would take heat from an unlimited cold store. The tests check:
  - A wet node 0 at 0 °C, a column below at 0 °C and a surface 1 K colder: the latent heat of the water that freezes in one sub-step equals the heat through the surface (`enthalpy` and `ncz`), and node 0 stays wet.
  - The column arrays have the same shape after the call, and `z` and `dz` are unchanged.
  - A wet node 0 under a cold surface does not end cold and wet, and the cold-layer check of `darcyscheme` does not fire.
- The energy. The heat of the column changes by `surface_heat`, the heat through the face between the ghost node and node 0. `surface_heat` comes from the solved state, not from the energy change.
  - One conduction call, wet node 0 and cold surface, `enthalpy` and `ncz`, in the metric of the solver (constant specific heat of ice, plus latent): the error is below 3e-7 of the surface heat.
  - One conduction call with no water (`heatDiff`): the error is below 1e-9 of the surface heat. The heat capacity is that of the solver, the specific heat of ice at the temperature before the solve.
  - 48 sub-steps of the whole scheme with no melt and no runoff, in the metric of the model (`column_heat` integrates the specific heat of ice at the node temperature). It is a loose physical check: the measured error is 0.04 % to 0.18 % of the surface heat, and the tolerance is 0.5 %. The error comes from the metric, not from a conservation error. `refreezeDiff` uses a constant 2097 J kg-1 K-1, and `heatDiff` uses the specific heat at the temperature before the solve. In these cases `refreezeDiff` runs only 1 to 3 times, so the test mostly checks `heatDiff`. The calls are checked exactly in the solver metric by the two tests above.
  - The boundary distance: on a dry column with a short `dt`, one call gives a surface heat equal to K0 / (dz0 / 2) times (Ts minus the solved T0) times `dt`, to 1e-4. A ghost node as thick as node 0 gives half of that value.
  - `surface_heat` is the heat that the solver used only for the solvers with one implicit solve (`enthalpy`, `ncz`). `ahc` does not count the latent heat, and `decp` uses internal steps, so the validator stops "substep" with them.
- The first step with water: a run with melt on the first day and "substep" completes (the conduction block of `time_evolve()` is skipped, and its diagnostics still run). `time_evolve()` stops with an error if the scheme covered more time than the step.
- The default: a run with the key absent and a run with "step" give identical arrays (`np.array_equal`).
- The validator stops "substep" with the explicit step, `manualT`, `heatDiff` false, `SEB` true, any `meltwater_solver` other than "enthalpy" or "ncz" ("ahc" does not count the latent heat, "decp" has a surface heat record that does not match its internal steps) and an unknown value. `darcyscheme` stops for an unknown value and for "substep" with the explicit step.

`test_thin_surface_node.py` checks the thin partial node 0 of the `darcy` scheme with the implicit flow step. The tests call `darcyscheme` on a synthetic column of one sub-step of one day, with the flow step switched off, so that the melt removal is the only change.

- The key `min_partial_node_dz_darcy` merges a partial node 0 thinner than the key into the node below. Node 0 is then at least the key thick. Compared with no merge, the column has the same mass, thickness, liquid water and sensible heat, to 1e-12, apart from one more copy of the bottom node. The merged node has the density mass / thickness and the mass-weighted grain size and age.
- A partial node above the key does not merge. The key at 0 (and no key) gives the same arrays bit for bit. A negative, a text or a boolean key stops the scheme. The scheme warns once if `conduction_interval_darcy` is "substep" and the key is 0.

## Compare two commits

`compare_commits.py` checks that two commits give the same model results. Run it by hand before you close a change that must keep the results. Put the result in the PR description. Commit your work first, because the script reads commits only.

```bash
python tests/compare_commits.py <base of the PR> HEAD
python tests/compare_commits.py <base of the PR> HEAD --ignore refrozen    # the change alters this dataset on purpose
python tests/compare_commits.py <base of the PR> HEAD --start-year 1995    # the change affects runoff or liquid water
```

- For each ref, the script makes a temporary git worktree, runs the cases with the code of that ref, and removes the worktree.
- The cases are `ShortRun` with `liquid` set to `darcy` and to `bucket`, for 2 years of the DYE-2 forcing from 1980 or from the year of `--start-year`. All other keys come from `example_df.json` of the ref.
- The script turns on all outputs in `outputs_options` of both refs. It writes `float64` values at the nodes, because `float32` values on a grid can hide a small difference.
- It compares each dataset of `CFMresults.hdf5` exactly: the shape, the NaN positions and each value. There is no tolerance.
- It reports a pass, or the first dataset that differs, with the first index, both values and the number of values that differ.
- The exit code is 0 if the results are equal, 1 if a dataset differs, and 2 if a ref is not found or a run fails.
- It takes less than a minute from 1980, and about 4 minutes from 1995.

These two years have little water. Neither case has runoff, and the `darcy` case has no liquid water at the write times. So a change that affects only those results can pass.

From 1995, the DYE-2 cases have runoff and liquid water at the write times. Use `--start-year 1995` for a change that affects these results. A later year takes longer, because at the default `bucket_until_year`, `darcy` runs for each model year from 1980, and the spin-up repeats the window back in time.

No CI job runs this script, because a change that alters the results on purpose would fail it.
