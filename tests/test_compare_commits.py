'''
Tests for compare_commits.py. The comparison tests use small HDF5 files. The full
check runs on a small git repository with a stub in place of the model, so these
tests take seconds. A run of the script on the model is a manual step.
'''

import subprocess

import h5py
import numpy as np
import pytest

import compare_commits
from compare_commits import compare_results, run_check

pytestmark = pytest.mark.unit


def _write(path, **datasets):
    with h5py.File(path, 'w') as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)
    return path


@pytest.fixture
def base_data():
    return np.array([[1980.0, 350.0, np.nan], [1981.0, 400.0, 917.0]])


def test_equal_files_with_nan_at_the_same_positions_pass(tmp_path, base_data):
    a = _write(tmp_path / 'a.hdf5', density=base_data)
    b = _write(tmp_path / 'b.hdf5', density=base_data)
    assert compare_results(a, b) is None


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_a_change_of_one_bit_is_reported(tmp_path, base_data, dtype):
    base_data = base_data.astype(dtype)
    changed = base_data.copy()
    changed[1, 1] = np.nextafter(changed[1, 1], np.inf)
    a = _write(tmp_path / 'a.hdf5', density=base_data)
    b = _write(tmp_path / 'b.hdf5', density=changed)

    difference = compare_results(a, b)

    assert difference.dataset == 'density'
    assert difference.index == (1, 1)
    assert difference.value_a == 400.0
    assert difference.value_b == changed[1, 1].item()
    assert difference.count == 1
    assert 'index (1, 1)' in str(difference)


def test_the_first_index_and_the_number_of_differences_are_reported(tmp_path, base_data):
    changed = base_data.copy()
    changed[0, 1] = 351.0
    changed[1, 2] = 918.0
    a = _write(tmp_path / 'a.hdf5', density=base_data)
    b = _write(tmp_path / 'b.hdf5', density=changed)

    difference = compare_results(a, b)

    assert difference.index == (0, 1)
    assert difference.count == 2


def test_a_change_of_nan_position_is_reported(tmp_path, base_data):
    changed = base_data.copy()
    changed[0, 2] = 917.0
    a = _write(tmp_path / 'a.hdf5', density=base_data)
    b = _write(tmp_path / 'b.hdf5', density=changed)

    difference = compare_results(a, b)

    assert difference.index == (0, 2)
    assert np.isnan(difference.value_a)
    assert difference.value_b == 917.0


def test_positive_and_negative_zero_are_equal(tmp_path):
    a = _write(tmp_path / 'a.hdf5', LWC=np.array([0.0]))
    b = _write(tmp_path / 'b.hdf5', LWC=np.array([-0.0]))
    assert compare_results(a, b) is None


def test_a_missing_dataset_is_reported_in_both_directions(tmp_path, base_data):
    a = _write(tmp_path / 'a.hdf5', density=base_data, runoff=base_data)
    b = _write(tmp_path / 'b.hdf5', density=base_data)

    assert compare_results(a, b).dataset == 'runoff'
    assert compare_results(a, b).reason == 'only in A'
    assert compare_results(b, a).reason == 'only in B'


def test_a_change_of_shape_is_reported(tmp_path, base_data):
    a = _write(tmp_path / 'a.hdf5', density=base_data)
    b = _write(tmp_path / 'b.hdf5', density=base_data[:, :2])

    difference = compare_results(a, b)

    assert difference.dataset == 'density'
    assert '(2, 3)' in difference.reason
    assert '(2, 2)' in difference.reason


def test_the_first_dataset_in_alphabetical_order_is_reported(tmp_path, base_data):
    a = _write(tmp_path / 'a.hdf5', age=base_data, density=base_data)
    b = _write(tmp_path / 'b.hdf5', age=base_data + 1, density=base_data + 1)
    assert compare_results(a, b).dataset == 'age'


def test_ignore_skips_a_dataset(tmp_path, base_data):
    a = _write(tmp_path / 'a.hdf5', density=base_data, runoff=base_data, refrozen=base_data)
    b = _write(tmp_path / 'b.hdf5', density=base_data, runoff=base_data + 1)

    assert compare_results(a, b, ignore=['runoff', 'refrozen']) is None
    assert compare_results(a, b, ignore=['refrozen']).dataset == 'runoff'


# A stub for ShortRun. Like the config validator of the model, it stops on an output
# that is not valid at its ref. It reads the file in the working folder, CFM_main/.
STUB_CFM_TESTING = '''
import json
from pathlib import Path
import h5py

class ShortRun:
    def __init__(self, folder, **settings):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True)
        self.settings = settings

    def run(self):
        {run_body}
        valid_outputs = json.loads(Path('example_df.json').read_text())['outputs_options']
        for output in self.settings['outputs']:
            if output not in valid_outputs:
                raise ValueError(f'invalid output: {{output}}')
        with h5py.File(self.folder / 'CFMresults.hdf5', 'w') as f:
            f['density'] = [{density}]
            f['age'] = [1.0]
'''


def _commit_stub(repo, density=350.0, run_body='pass', outputs=('density', 'runoff')):
    '''Commit a stub of the model to repo and return the commit hash.'''
    (repo / 'tests').mkdir(exist_ok=True)
    (repo / 'CFM_main').mkdir(exist_ok=True)
    (repo / 'tests' / 'cfm_testing.py').write_text(
        STUB_CFM_TESTING.format(density=density, run_body=run_body))
    outputs_json = ', '.join(f'"{output}"' for output in outputs)
    (repo / 'CFM_main' / 'example_df.json').write_text(f'{{"outputs_options": [{outputs_json}]}}')
    for command in (['add', '.'],
                    ['-c', 'user.name=test', '-c', 'user.email=test@example.com',
                     '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'stub']):
        subprocess.run(['git', '-C', str(repo), *command], check=True)
    return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def stub_repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '--quiet', str(repo)], check=True)
    return repo


def _worktree_count(repo):
    listing = subprocess.run(['git', '-C', str(repo), 'worktree', 'list'], check=True,
                             capture_output=True, text=True).stdout
    return len(listing.splitlines())


def test_check_passes_for_one_commit_against_itself(stub_repo, capsys):
    commit = _commit_stub(stub_repo)

    assert run_check(commit, commit, repo=stub_repo) == 0

    output = capsys.readouterr().out
    assert 'darcy: equal' in output
    assert 'bucket: equal' in output
    assert 'Result: pass' in output
    assert _worktree_count(stub_repo) == 1


def test_check_runs_the_code_of_each_ref(stub_repo, capsys):
    commit_a = _commit_stub(stub_repo, density=350.0)
    commit_b = _commit_stub(stub_repo, density=351.0)

    assert run_check(commit_a, commit_b, repo=stub_repo) == 1

    output = capsys.readouterr().out
    assert 'darcy: differs, dataset density' in output
    assert 'A = 350.0, B = 351.0' in output
    assert 'Result: DIFFERS' in output
    assert _worktree_count(stub_repo) == 1


def test_check_ignores_a_named_dataset(stub_repo, capsys):
    commit_a = _commit_stub(stub_repo, density=350.0)
    commit_b = _commit_stub(stub_repo, density=351.0)

    assert run_check(commit_a, commit_b, ignore=['density'], repo=stub_repo) == 0

    output = capsys.readouterr().out
    assert 'Ignored datasets: density' in output
    assert 'darcy: equal (1 datasets: age)' in output


@pytest.mark.parametrize('arguments, expected_year', [({}, 1980), ({'start_year': 1995}, 1995)])
def test_check_runs_the_cases_from_the_start_year(stub_repo, capsys, arguments, expected_year):
    # The stub stops the run if the year is not correct.
    commit = _commit_stub(stub_repo, run_body=f"assert self.settings['start_year'] == {expected_year}")

    assert run_check(commit, commit, repo=stub_repo, **arguments) == 0
    assert f'Start year: {expected_year}' in capsys.readouterr().out


def test_the_command_line_passes_the_start_year_to_the_check(monkeypatch):
    calls = []
    monkeypatch.setattr(compare_commits, 'run_check', lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr('sys.argv', ['compare_commits.py', 'A', 'B', '--start-year', '1995'])

    compare_commits.main()

    assert calls == [{'start_year': 1995}]


def test_check_turns_on_only_the_outputs_at_both_refs(stub_repo):
    commit_a = _commit_stub(stub_repo, outputs=('density', 'runoff'))
    commit_b = _commit_stub(stub_repo, outputs=('density', 'runoff', 'new_output'))
    # new_output stops the run at the ref that does not list it, in either order.
    assert run_check(commit_a, commit_b, repo=stub_repo) == 0
    assert run_check(commit_b, commit_a, repo=stub_repo) == 0


def test_check_reports_a_failed_run_and_removes_the_worktree(stub_repo, capsys):
    commit = _commit_stub(stub_repo, run_body="raise ValueError('model crash')")

    assert run_check(commit, commit, repo=stub_repo) == 2

    output = capsys.readouterr().out
    assert 'FAILED' in output
    assert 'model crash' in output
    assert _worktree_count(stub_repo) == 1


def test_check_reports_an_unknown_ref(stub_repo, capsys):
    commit = _commit_stub(stub_repo)
    assert run_check(commit, 'no-such-ref', repo=stub_repo) == 2
    assert 'FAILED' in capsys.readouterr().out
