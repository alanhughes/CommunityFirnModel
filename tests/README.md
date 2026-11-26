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

## Restart test

`test_restart.py` runs a model for 12 years and saves a snapshot in the spin file (`spinUpdate`). It restarts from a copy of the spin file. It compares every output dataset after the snapshot, and the state at the end of the run, with no tolerance (bit for bit). One `bucket` case restarts with 3.5 years of forcing left, less than the 10-year window of the rolling means of the surface temperature and the accumulation rate. Dataset `DIP` is compared without its cumulative columns (they restart at zero) and without the dH values of its first row (the model writes zero there). A second test removes the restart datasets from the spin file and checks that the restart still runs and uses the mean steps per year of its own part of the record. A third test checks that `read_restart_value` returns nothing for a time that is not the snapshot time.
