'''
The config key bucket_until_year: darcy, prefsnowpack and resingledomain run bucket
before this model year, a message says so, and null means no switch.
'''

import json
from pathlib import Path

import pytest

import firn_density_nospin
from config_validation import KNOWN_KEYS

SCHEME_FUNCTIONS = {'darcy': 'darcyscheme', 'prefsnowpack': 'prefflow', 'resingledomain': 'resingledomain'}
WET_FORCING = 'CFM_example_66.5_-46.25.pkl'  # DYE-2 has wet steps in each year. The default forcing has none in 1981.


def _record_scheme_calls(monkeypatch):
    '''
    Record each scheme call of time_evolve as (model year, function name, scheme label).
    Only bucket and darcyscheme set the label in melt_diagnostics.
    '''
    calls = []

    def recorded(name):
        original = getattr(firn_density_nospin, name)

        def wrapper(model, iii):
            outputs = original(model, iii)
            label = model.melt_diagnostics['scheme'] if hasattr(model, 'melt_diagnostics') else None
            calls.append((model.modeltime[iii], name, label))
            return outputs

        return wrapper

    for name in ['bucket', *SCHEME_FUNCTIONS.values()]:
        monkeypatch.setattr(firn_density_nospin, name, recorded(name))
    return calls


@pytest.mark.unit
@pytest.mark.parametrize('config_name', ['example.json', 'example_df.json', 'example_csv.json'])
def test_example_configs_set_the_default(config_name):
    config = json.loads((Path('.') / config_name).read_text())
    assert config['bucket_until_year'] == 1980


@pytest.mark.unit
def test_validator_knows_the_key():
    assert 'bucket_until_year' in KNOWN_KEYS


@pytest.mark.conservation
def test_darcy_switches_at_the_year_of_the_key(short_config, monkeypatch):
    calls = _record_scheme_calls(monkeypatch)
    short_config(years=2, num_reps=1, liquid='darcy', bucket_until_year=1981, DFfile=WET_FORCING).run()
    # The year 1980 is the important part: the old fixed switch would use darcy in it.
    assert any(1980 <= year < 1981 for year, _, _ in calls)
    assert any(year >= 1981 for year, _, _ in calls)
    for year, _, label in calls:
        assert label == ('bucket' if year < 1981 else 'darcy')


@pytest.mark.conservation
@pytest.mark.parametrize('liquid', ['darcy', 'prefsnowpack', 'resingledomain'])
def test_step_on_the_key_runs_the_scheme(short_config, monkeypatch, liquid):
    # A key of 1981 falls between two monthly steps. This key is the time of a wet step.
    model = short_config(years=2, num_reps=1, liquid=liquid, bucket_until_year=None, DFfile=WET_FORCING).build()
    wet_times = model.modeltime[model.snowmeltSec > 0]
    key = float(wet_times[wet_times >= 1981][0])

    calls = _record_scheme_calls(monkeypatch)
    short_config(years=2, num_reps=1, liquid=liquid, bucket_until_year=key, DFfile=WET_FORCING).run()
    assert any(1980 <= year < key for year, _, _ in calls)
    assert key in [year for year, _, _ in calls]
    for year, name, _ in calls:
        assert name == ('bucket' if year < key else SCHEME_FUNCTIONS[liquid])


@pytest.mark.conservation
@pytest.mark.parametrize('liquid', ['darcy', 'prefsnowpack', 'resingledomain'])
def test_null_runs_the_scheme_from_the_first_step(short_config, monkeypatch, liquid):
    calls = _record_scheme_calls(monkeypatch)
    short_config(years=2, num_reps=1, liquid=liquid, bucket_until_year=None).run()
    assert calls[0][0] < 1980  # A run from before the default year.
    assert {name for _, name, _ in calls} == {SCHEME_FUNCTIONS[liquid]}


@pytest.mark.unit
def test_config_without_the_key_uses_1980(short_config, capsys):
    run = short_config(years=2, num_reps=1, liquid='darcy')
    config = json.loads(run.config_path.read_text())
    del config['bucket_until_year']
    run.config_path.write_text(json.dumps(config))
    assert run.build().bucket_until_year == 1980
    assert 'before model year 1980' in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize('liquid', ['darcy', 'prefsnowpack', 'resingledomain'])
def test_message_for_a_run_that_switches(short_config, capsys, liquid):
    short_config(years=2, num_reps=1, liquid=liquid, bucket_until_year=1981).build()
    assert capsys.readouterr().out.count(f'this run uses bucket before model year 1981 (bucket_until_year), then {liquid}') == 1


@pytest.mark.unit
@pytest.mark.parametrize('liquid', ['darcy', 'prefsnowpack', 'resingledomain'])
def test_message_for_a_run_that_ends_before_the_year(short_config, capsys, liquid):
    short_config(years=2, num_reps=1, liquid=liquid, bucket_until_year=2000).build()
    output = capsys.readouterr().out
    assert output.count('all steps use bucket') == 1
    assert 'this run uses bucket before' not in output


@pytest.mark.unit
def test_message_for_a_key_on_the_last_step(short_config, capsys):
    # The last step runs the scheme, so the run does switch.
    model = short_config(years=2, num_reps=1, liquid='darcy', bucket_until_year=None).build()
    capsys.readouterr()
    short_config(years=2, num_reps=1, liquid='darcy', bucket_until_year=float(model.modeltime[-1])).build()
    output = capsys.readouterr().out
    assert 'this run uses bucket before' in output
    assert 'all steps use bucket' not in output


@pytest.mark.unit
def test_no_message_for_a_key_on_the_first_step(short_config, capsys):
    model = short_config(years=2, num_reps=1, liquid='darcy', bucket_until_year=None).build()
    capsys.readouterr()
    short_config(years=2, num_reps=1, liquid='darcy', bucket_until_year=float(model.modeltime[0])).build()
    assert 'bucket_until_year' not in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize('overrides', [
    {'liquid': 'darcy', 'bucket_until_year': None},  # No switch.
    {'liquid': 'darcy', 'bucket_until_year': 1950},  # The run starts after this year.
    {'liquid': 'bucket', 'bucket_until_year': 1981},  # The key does not apply to bucket.
    {'liquid': 'darcy', 'bucket_until_year': 1981, 'MELT': False},  # No meltwater scheme runs.
])
def test_no_message_for_a_run_with_no_switch(short_config, capsys, overrides):
    short_config(years=2, num_reps=1, **overrides).build()
    assert 'bucket_until_year' not in capsys.readouterr().out
