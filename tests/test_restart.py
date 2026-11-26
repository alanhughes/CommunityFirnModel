'''
A restart from a spin file snapshot (spinUpdate) continues the run bit for bit.

A full run saves a snapshot in the middle of the second year. A second run restarts from a copy of
that spin file. For every output dataset, the rows after the snapshot have to be identical in the two
runs, with no tolerance. A spin file without the datasets massSpin, dzSpin and iceoutSpin (the
spin-up column, or a file from an earlier version) still restarts.
'''

import json
import shutil

import h5py
import numpy as np
import pytest

SNAPSHOT_DATE = 1980.5

# The restart has to have at least 10 years of forcing left: the rolling mean of the surface temperature
# and of the accumulation (T_mean, bdot_av) is a flat mean over a shorter forcing, which is another run.
# grid_output_max_depth is set so that both runs write the gridded outputs to the same depth.
RUN_SETTINGS = dict(years=12, start_year=1980, num_reps=1, spinUpdate=True, spinUpdateDate=SNAPSHOT_DATE,
                    grid_output_max_depth=20.0)

# State arrays that are compared at the end of the run (double precision).
FINAL_STATE = ('rho', 'Tz', 'age', 'z', 'dz', 'mass', 'LWC', 'bdot_mean', 'sigma')

# DIP columns: time, depth-integrated porosity, dH, cumulative dH, compaction, dH corrected, cumulative dH
# corrected, DIP horizon. The cumulative sums (columns 3 and 6) start at zero at the first write of a run, so
# a restart has another offset. The model also writes zero dH at the first write of a run (first row).
DIP_CUMULATIVE_COLUMNS = (3, 6)
DIP_FIRST_ROW_ZERO_COLUMNS = (2, 4, 5)


def _restart(full_run, folder, remove_datasets=()):
    '''Restart from a copy of the spin file of full_run. Return the path of the results file.'''
    import solver
    from firn_density_nospin import FirnDensityNoSpin

    folder.mkdir(parents=True, exist_ok=True)
    spin_copy = folder / full_run.config['spinFileName']
    shutil.copy(full_run.folder / full_run.config['spinFileName'], spin_copy)
    if remove_datasets:
        with h5py.File(spin_copy, 'a') as f:
            for name in remove_datasets:
                del f[name]
    config = dict(full_run.config, resultsFolder=str(folder), NewSpin=False)
    config_path = folder / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)
    solver._diag_reset()  # module-level list; it is not reset between runs
    model = FirnDensityNoSpin(str(config_path), climateTS=full_run.climateTS, NewSpin=False,
                              SEBfluxes=full_run.SEBfluxes)
    model.time_evolve()
    return folder / full_run.config['resultsFileName'], model


def _snapshot_time(spin_path):
    with h5py.File(spin_path, 'r') as f:
        times = f['densitySpin'][0, :]
    return times[np.abs(times - SNAPSHOT_DATE).argmin()]


CASES = [
    pytest.param(dict(liquid='bucket', DFfile='CFM_example_66.5_-46.25.pkl'), id='bucket_dye2'),
    pytest.param(dict(MELT=False), id='dry_summit'),
]


@pytest.mark.scheme
@pytest.mark.parametrize('overrides', CASES)
def test_restart_is_bit_exact(short_config, tmp_path, overrides):
    full_run = short_config(**RUN_SETTINGS, **overrides)
    full_model = full_run.run()
    snapshot_time = _snapshot_time(full_run.folder / full_run.config['spinFileName'])
    restart_path, restart_model = _restart(full_run, tmp_path / 'restart')

    with h5py.File(full_run.results_path, 'r') as full, h5py.File(restart_path, 'r') as restart:
        assert set(full.keys()) == set(restart.keys())
        for name in full.keys():
            a, b = full[name][:], restart[name][:]
            # rows are times, column 0 is the time. The restart repeats its first row, so align by time.
            a, b = a[a[:, 0] > snapshot_time], b[b[:, 0] > snapshot_time]
            assert a.shape[0] > 0, name
            if name == 'DIP':
                for column in DIP_FIRST_ROW_ZERO_COLUMNS:
                    assert b[0, column] == 0.0
                    a[0, column] = b[0, column] = 0.0
                keep = [column for column in range(a.shape[1]) if column not in DIP_CUMULATIVE_COLUMNS]
                a, b = a[:, keep], b[:, keep]
            assert np.array_equal(a, b, equal_nan=True), f'{name} differs after the restart'

    # The outputs have single precision. The state at the end of the run, in double precision, shows
    # roundoff differences of the restart that the outputs do not.
    for name in FINAL_STATE:
        assert np.array_equal(getattr(full_model, name), getattr(restart_model, name)), f'{name} differs at the end'


@pytest.mark.scheme
def test_restart_from_a_file_without_the_restart_datasets(short_config, tmp_path):
    full_run = short_config(**RUN_SETTINGS, MELT=False)
    full_run.run()
    spin_path = full_run.folder / full_run.config['spinFileName']
    with h5py.File(spin_path, 'r') as f:
        assert {'massSpin', 'dzSpin', 'iceoutSpin'} <= set(f.keys())
    restart_path, _ = _restart(full_run, tmp_path / 'restart', remove_datasets=('massSpin', 'dzSpin', 'iceoutSpin'))
    assert restart_path.exists()


@pytest.mark.scheme
def test_snapshot_of_another_time_is_not_used(short_config):
    '''read_restart_value returns a value only for the snapshot at the restart time.'''
    from reader import read_restart_value
    full_run = short_config(**RUN_SETTINGS, MELT=False)
    full_run.run()
    folder, name = str(full_run.folder), full_run.config['spinFileName']
    time = _snapshot_time(full_run.folder / name)
    assert read_restart_value(folder, name, 'massSpin', time)[0] == time
    assert read_restart_value(folder, name, 'massSpin', time + 0.001) is None
    assert read_restart_value(folder, name, 'massSpin', None) is None
