'''
The output 'darcy_snapshot': the state of the darcy scheme partway through a step.

The model writes its outputs at the end of a step. The darcy scheme moves the water in
sub-steps inside the step, so the end of the step can hide the state during the melt.
The snapshot records LWC, rho, Tz and z at the end of the first sub-step that reaches
snapshot_fraction_darcy of the step, and the time of that sub-step.
'''

import h5py
import numpy as np
import pytest

import firn_density_nospin
import melt
from ModelOutputs import ModelOutputs

DAY = 86400.
SNAPSHOT_STATE = ('LWC', 'rho', 'Tz', 'z')


def _draining_column(synthetic_column, dt=DAY, melt_amount=0.0, snapshot=False, fraction=0.5):
    '''
    Temperate firn at 400 kg/m3 with wet top nodes: the water drains during the step. Nothing
    refreezes, no node is above rholim_dcy and no node is cold, so the end of the routine
    changes nothing after the last sub-step.
    '''
    column = synthetic_column(n=60, rho_profile=400.0, T_profile=273.15, dz=0.05,
                              LWC=lambda z: np.where(z < 0.3, 0.01, 0.0), melt=melt_amount, dt=dt,
                              config={'snapshot_fraction_darcy': fraction})
    column.darcy_snapshot = snapshot
    return column


@pytest.mark.scheme
@pytest.mark.parametrize('melt_amount', [0.0, 0.01], ids=['drainage', 'melt'])
def test_snapshot_does_not_change_the_run(synthetic_column, melt_amount):
    # The sub-step lengths are the same with and without the snapshot, so the end state is bit for bit.
    with_snapshot = _draining_column(synthetic_column, melt_amount=melt_amount, snapshot=True)
    without = _draining_column(synthetic_column, melt_amount=melt_amount)
    out_with = melt.darcyscheme(with_snapshot, 0)
    out_without = melt.darcyscheme(without, 0)
    for a, b in zip(out_with, out_without):
        np.testing.assert_array_equal(a, b)


@pytest.mark.scheme
def test_snapshot_is_the_state_at_snapshot_time(synthetic_column):
    # A call that ends at snapshot_time takes the same sub-steps up to there, so its end state
    # is the snapshot. With no melt, the input does not depend on the step length.
    full = _draining_column(synthetic_column, snapshot=True)
    melt.darcyscheme(full, 0)
    dtmax = melt._DARCY_DEFAULTS['dtmax_darcy']
    assert 0.5 * DAY <= full.snapshot_time <= 0.5 * DAY + dtmax

    short = _draining_column(synthetic_column, dt=full.snapshot_time)
    melt.darcyscheme(short, 0)
    for name in SNAPSHOT_STATE:
        np.testing.assert_array_equal(getattr(full, name + '_snapshot'), getattr(short, name))
    # The water moves after the snapshot, so the snapshot is not the end state.
    assert np.max(np.abs(full.LWC_snapshot - full.LWC)) > 1e-6


@pytest.mark.scheme
def test_snapshot_at_the_end_of_the_step(synthetic_column):
    column = _draining_column(synthetic_column, snapshot=True, fraction=1.0)
    melt.darcyscheme(column, 0)
    assert column.snapshot_time == pytest.approx(DAY)
    for name in SNAPSHOT_STATE:
        np.testing.assert_array_equal(getattr(column, name + '_snapshot'), getattr(column, name))


@pytest.mark.scheme
@pytest.mark.parametrize('fraction', [-0.1, 1.5])
def test_snapshot_fraction_out_of_range(synthetic_column, fraction):
    column = _draining_column(synthetic_column, snapshot=True, fraction=fraction)
    with pytest.raises(ValueError, match='snapshot_fraction_darcy'):
        melt.darcyscheme(column, 0)


SNAPSHOT_DATASETS = ('LWC_snapshot', 'rho_snapshot', 'Tz_snapshot', 'z_snapshot', 'snapshot_time')
RUN_OUTPUTS = ['density', 'depth', 'temperature', 'age', 'LWC']


def _run(short_config, outputs, **overrides):
    run = short_config(years=2, DFfile='CFM_example_66.5_-46.25.pkl', TWriteInt=1, outputs=outputs, **overrides)
    model = run.run()
    with h5py.File(run.results_path, 'r') as f:
        results = {name: f[name][:] for name in f.keys()}
    results['step_length'] = model.dt   # [s]
    return results


@pytest.mark.conservation
def test_snapshot_output(short_config, monkeypatch):
    # Native grid and gridded runs, and a run without the output: the snapshot does not change
    # the results, it is NaN in the steps where darcyscheme does not run, and gridding uses the
    # depth of the snapshot.
    calls = []
    def counted_darcyscheme(self, iii):
        calls.append(iii)
        return melt.darcyscheme(self, iii)
    monkeypatch.setattr(firn_density_nospin, 'darcyscheme', counted_darcyscheme)
    native = _run(short_config, RUN_OUTPUTS + ['darcy_snapshot'], liquid='darcy', grid_outputs=False)
    native_calls = len(calls)
    gridded = _run(short_config, RUN_OUTPUTS + ['darcy_snapshot'], liquid='darcy', grid_outputs=True)
    without = _run(short_config, RUN_OUTPUTS, liquid='darcy', grid_outputs=False)

    for name in SNAPSHOT_DATASETS:
        assert name in native and name in gridded
        assert name not in without
    np.testing.assert_array_equal(native['step_length'], without['step_length'])
    for name in ('density', 'temperature', 'LWC', 'refreeze', 'runoff'):
        np.testing.assert_array_equal(native[name], without[name])

    snapshot_time = native['snapshot_time'][1:, 1]   # row 0 is the initial state
    taken = np.isfinite(snapshot_time)
    assert taken.sum() == native_calls               # every step is written
    assert taken.sum() > 0 and (~taken).sum() > 0    # the run has wet and dry steps
    # The snapshot comes at most one sub-step (dtmax_darcy) after the middle of the step.
    dtmax = melt._DARCY_DEFAULTS['dtmax_darcy']
    step_length = native['step_length']
    assert np.all(snapshot_time[taken] >= 0.5 * step_length.min() * (1 - 1e-6))
    assert np.all(snapshot_time[taken] <= 0.5 * step_length.max() + dtmax)
    for name in SNAPSHOT_DATASETS[:4]:
        rows = native[name][1:, 1:]
        assert np.all(np.isnan(rows[~taken]))
        assert np.all(np.isfinite(rows[taken]))
    np.testing.assert_array_equal(np.isfinite(gridded['snapshot_time'][1:, 1]), taken)

    # Gridded on the snapshot depth: the native snapshot regridded by hand, in the top 5 m.
    resolution = 0.25   # grid_output_res of example_df.json
    cells = np.arange(0., 5., resolution)
    wet_rows = np.where(np.nansum(native['LWC_snapshot'][:, 1:], axis=1) > 0)[0]
    assert len(wet_rows) > 0
    for row in wet_rows:
        expected = ModelOutputs.RGfun(None, native['z_snapshot'][row, 1:].astype(float),
                                      native['LWC_snapshot'][row, 1:].astype(float), cells, to_bottom=False)
        np.testing.assert_allclose(gridded['LWC_snapshot'][row, 1:len(cells) + 1], expected, rtol=1e-4, atol=1e-8)


@pytest.mark.conservation
def test_snapshot_output_needs_darcy(short_config, capsys):
    results = _run(short_config, RUN_OUTPUTS + ['darcy_snapshot'], liquid='bucket')
    assert "'darcy_snapshot' needs MELT true and liquid 'darcy'" in capsys.readouterr().out
    for name in SNAPSHOT_DATASETS:
        assert name not in results
