'''
The key step_timing: the results of a run do not change, and step_timing.csv has one row for each
step, with the parts that add up to the step and the counts of the melt schemes.
'''

import csv
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import firn_density_nospin
import melt
import step_timing
from config_validation import ConfigValidationError, validate_config
from test_darcy_implicit import edge_column

WET_FORCING = 'CFM_example_66.5_-46.25.pkl'  # DYE-2 has wet steps in each year.
IMPLICIT = dict(liquid='darcy', bucket_until_year=1981, flow_step_darcy='implicit_gravity',
                dtsub_implicit_darcy=1800., capillary_implicit_darcy='power_law', capillary_pressure_darcy=1000.,
                DFfile=WET_FORCING, num_reps=1)
PARTS = ('pre', 'melt', 'mid', 'heat', 'post')


def _run(short_config, monkeypatch, **overrides):
    calls = {'bucket': 0, 'darcy': 0}
    for name, function in (('bucket', 'bucket'), ('darcy', 'darcyscheme')):
        def wrapper(model, iii, original=getattr(firn_density_nospin, function), name=name):
            calls[name] += 1
            return original(model, iii)
        monkeypatch.setattr(firn_density_nospin, function, wrapper)
    run = short_config(years=2, **overrides)
    model = run.run()
    with h5py.File(run.results_path, 'r') as f:
        results = {name: f[name][:] for name in f.keys()}
    return run, model, results, calls


@pytest.mark.conservation
def test_key_on_gives_the_same_results_and_the_rows(short_config, monkeypatch, capsys):
    off_run, off_model, off, _ = _run(short_config, monkeypatch, **IMPLICIT)
    assert not hasattr(off_model, 'step_timing_stages')
    assert not (off_run.folder / 'step_timing.csv').exists()
    capsys.readouterr()
    on_run, on_model, on, calls = _run(short_config, monkeypatch, step_timing=True, **IMPLICIT)
    printed = capsys.readouterr().out

    # the same datasets, bit for bit
    assert off.keys() == on.keys()
    for name in off:
        assert np.array_equal(off[name], on[name], equal_nan=True), name

    # one row for each step, the parts add up to the step, the schemes and the sub-steps agree
    with open(on_run.folder / 'step_timing.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == on_model.stp
    assert [int(row['step']) for row in rows] == list(range(on_model.stp))
    for row in rows:
        parts = sum(float(row[f'wall_{part}_s']) for part in PARTS)
        assert abs(parts - float(row['wall_s'])) < 1e-9
        assert abs(sum(float(row[f'cpu_{part}_s']) for part in PARTS) - float(row['cpu_s'])) < 1e-9
        assert float(row['wall_s']) > 0.
    schemes = [row['scheme'] for row in rows]
    assert schemes.count('darcy') == calls['darcy'] > 0
    assert schemes.count('bucket') == calls['bucket'] > 0
    assert schemes.count('dry') > 0
    for row in rows:
        if row['scheme'] == 'darcy':
            assert int(row['substeps']) >= 1
            assert row['root_solves'] != ''
            assert 0 <= int(row['idle_substeps']) <= int(row['substeps'])
        else:
            assert row['root_solves'] == '' and row['idle_substeps'] == ''
        if row['scheme'] in ('darcy', 'bucket'):
            assert row['budget_residual'] != ''
        if row['scheme'] == 'bucket':
            assert row['substeps'] == ''
    assert sum(int(row['substeps']) for row in rows if row['scheme'] == 'darcy') > 0

    # one line for each model year, and the wall time of a year is the sum of its rows
    lines = [line for line in printed.splitlines() if ' step_timing year ' in line]
    assert len(lines) == len({int(float(row['modeltime'])) for row in rows})
    for line in lines:
        year = int(line.split(' step_timing year ')[1].split(':')[0].split()[0])
        wall = float(line.split('wall ')[1].split(' s')[0])
        year_rows = [row for row in rows if int(float(row['modeltime'])) == year]
        assert abs(wall - sum(float(row['wall_s']) for row in year_rows)) < 0.006   # the line has two decimals
    # the monthly forcing has dates at the end of each month, and the two years of the run are full years
    assert not any('(part year)' in line for line in lines)

    # the stages for the program that runs the model
    stages = on_model.step_timing_stages
    assert set(stages) == {'loop_start', 'switch', 'loop_end', 'write_end'}
    for stage in stages.values():
        assert set(stage) == {'wall', 'cpu'}
    assert stages['loop_start']['wall'] < stages['switch']['wall'] < stages['loop_end']['wall'] < stages['write_end']['wall']


@pytest.mark.scheme
def test_idle_substeps_count_the_sub_steps_with_no_input_and_no_water_above_the_irreducible_water():
    dry = edge_column('dry_no_input', 1800.)
    melt.darcyscheme(dry, 0)
    assert dry.melt_diagnostics['idle_substeps'] == dry.melt_diagnostics['substeps'] == 48
    wet = edge_column('base', 1800.)
    melt.darcyscheme(wet, 0)
    assert wet.melt_diagnostics['idle_substeps'] == 0
    explicit = edge_column('dry_no_input', 1800.)
    explicit.c['flow_step_darcy'] = 'explicit'
    melt.darcyscheme(explicit, 0)
    assert 'idle_substeps' not in explicit.melt_diagnostics


@pytest.mark.conservation
def test_switch_is_none_when_the_run_ends_before_the_year(short_config, monkeypatch):
    _, model, _, _ = _run(short_config, monkeypatch, step_timing=True, **dict(IMPLICIT, bucket_until_year=2100))
    assert model.step_timing_stages['switch'] is None


@pytest.mark.unit
def test_validation_needs_a_boolean():
    config = {'step_timing': 'yes'}
    with pytest.raises(ConfigValidationError, match="'step_timing' should be true or false"):
        validate_config(dict(config))


@pytest.mark.conservation
def test_stages_are_filled_as_the_run_goes(short_config, monkeypatch):
    # A run that stops inside the loop keeps loop_start and switch, as plain finite floats.
    def failing_darcyscheme(model, iii):
        raise RuntimeError('stop')
    monkeypatch.setattr(firn_density_nospin, 'darcyscheme', failing_darcyscheme)
    run = short_config(years=2, step_timing=True, **IMPLICIT)
    model = run.build()
    with pytest.raises(RuntimeError):
        model.time_evolve()
    stages = model.step_timing_stages
    assert stages['loop_end'] is None and stages['write_end'] is None
    for name in ('loop_start', 'switch'):
        assert set(stages[name]) == {'wall', 'cpu'}
        assert all(type(value) is float and np.isfinite(value) for value in stages[name].values())


@pytest.mark.unit
def test_part_year_label_comes_from_the_number_of_steps(tmp_path, capsys):
    # Monthly steps at the end of each month: 1980 is full, the run stops after June 1981.
    month_ends = np.cumsum([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30])
    modeltime = np.where(month_ends < 366, 1980. + month_ends / 366., 1981. + (month_ends - 366) / 365.)
    modeltime[11] = 1980.9999999    # 31 December 1980 stays in 1980
    model = SimpleNamespace(c={'resultsFolder': str(tmp_path)}, modeltime=modeltime, MELT=False,
                            dz=np.ones(5), LWC=np.zeros(5))
    timer = step_timing.StepTimer(model, 1.)
    for iii in range(len(modeltime)):
        timer.begin(iii)
        for part in PARTS[:-1]:
            timer.mark(part)
        timer.end_step(iii)
    timer.finish()
    lines = [line for line in capsys.readouterr().out.splitlines() if ' step_timing year ' in line]
    assert len(lines) == 2
    assert 'year 1980:' in lines[0] and 'steps 12 ' in lines[0]
    assert 'year 1981 (part year):' in lines[1] and 'steps 6 ' in lines[1]
