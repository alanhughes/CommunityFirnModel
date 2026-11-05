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
