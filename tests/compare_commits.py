#!/usr/bin/env python
'''
compare_commits.py
==================
Check that two commits give the same model results. Usage:

    python tests/compare_commits.py <ref A> <ref B> [--ignore <dataset>] [--start-year <year>]

For each ref, the script makes a temporary git worktree, runs each case with the
code of that ref, and removes the worktree. It then compares each dataset of the
two CFMresults.hdf5 files exactly: the shape, the NaN positions and each value.
There is no tolerance. 0.0 and -0.0 are the same value.

Both refs run on this machine because the results of some schemes differ between
machines. So a saved copy of the results cannot do this check.

The script reads commits only. It does not see uncommitted changes.

Exit code: 0 if the results are equal, 1 if a dataset differs, 2 if a ref is not
found or a run fails.
'''

import argparse
import json
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS_FILE = 'CFMresults.hdf5'
LIQUID_SCHEMES = ('darcy', 'bucket')

# The ShortRun settings of each case. All other keys come from example_df.json of
# the ref, so the check tests the default of that ref.
# output_bits and grid_outputs change only how the results are written. The float32
# and the interpolation to a grid of example_df.json can hide a small difference.
CASE_SETTINGS = {
    'years': 2,
    'start_year': 1980,  # The default of --start-year. darcy runs bucket before 1980.
    'DFfile': 'CFM_example_66.5_-46.25.pkl',  # DYE-2, which has melt.
    'output_bits': 'float64',
    'grid_outputs': False,
}

# Runs in a new process, in CFM_main/ of the worktree, so it imports the code of that
# ref. It is not a file in tests/ because the base ref of a PR does not have this script.
RUN_CASE = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from cfm_testing import ShortRun
ShortRun(sys.argv[2], **json.loads(sys.argv[3])).run()
'''


class RunFailed(Exception):
    '''A git command or a model run failed.'''


@dataclass
class Difference:
    '''
    The first dataset that differs. A change of values sets index, value_a, value_b
    and count. A missing dataset or a change of shape sets only the reason.
    '''
    dataset: str
    reason: str
    index: tuple = None
    value_a: float = None
    value_b: float = None
    count: int = None

    def __str__(self):
        text = f'dataset {self.dataset}: {self.reason}'
        if self.count is not None:
            text += f'. First at index {self.index}: A = {self.value_a!r}, B = {self.value_b!r}'
        return text


def compare_results(path_a, path_b, ignore=()):
    '''
    Compare two results files. The datasets are compared in alphabetical order.

    :param ignore: the names of datasets to skip
    :return: None if the files are equal, or a Difference for the first dataset that
        differs
    '''
    with h5py.File(path_a, 'r') as file_a, h5py.File(path_b, 'r') as file_b:
        for name in sorted((set(file_a) | set(file_b)) - set(ignore)):
            if name not in file_b:
                return Difference(name, 'only in A')
            if name not in file_a:
                return Difference(name, 'only in B')

            a, b = file_a[name][()], file_b[name][()]
            if a.shape != b.shape:
                return Difference(name, f'shape {a.shape} in A and {b.shape} in B')

            differs = (a != b) & ~(np.isnan(a) & np.isnan(b))  # NaN != NaN is true
            if differs.any():
                index = tuple(int(i) for i in np.argwhere(differs)[0])
                count = int(np.count_nonzero(differs))
                return Difference(name, f'{count} of {a.size} values differ', index,
                                  a[index].item(), b[index].item(), count)
    return None


def _git(repo, *args):
    '''Run a git command in repo and return its output.'''
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RunFailed(f'git {" ".join(args)}: {result.stderr.strip()}')
    return result.stdout.strip()


@contextmanager
def _temporary_worktree(repo, commit, path):
    '''Check out commit in a new worktree at path. Remove the worktree at the end.'''
    _git(repo, 'worktree', 'add', '--quiet', '--detach', str(path), commit)
    try:
        yield path
    finally:
        _git(repo, 'worktree', 'remove', '--force', str(path))


def _common_outputs(repo, commit_a, commit_b):
    '''Return the outputs that outputs_options of example_df.json lists at both commits.'''
    options_a, options_b = (json.loads(_git(repo, 'show', f'{commit}:CFM_main/example_df.json'))['outputs_options']
                            for commit in (commit_a, commit_b))
    return [output for output in options_a if output in options_b]


def _run_case(worktree, folder, liquid, outputs, start_year):
    '''Run one case with the code in worktree. Return the path of the results file.'''
    settings = dict(CASE_SETTINGS, liquid=liquid, outputs=outputs, start_year=start_year)
    result = subprocess.run(
        [sys.executable, '-c', RUN_CASE, str(worktree / 'tests'), str(folder), json.dumps(settings)],
        cwd=worktree / 'CFM_main', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        last_lines = '\n'.join(result.stdout.splitlines()[-20:])
        raise RunFailed(f'The {liquid} case failed at {worktree}:\n{last_lines}')
    return folder / RESULTS_FILE


def run_check(ref_a, ref_b, ignore=(), repo=REPO, start_year=CASE_SETTINGS['start_year']):
    '''
    Run each case at both refs, compare the results and print a report.

    :param ignore: the names of datasets to skip
    :param start_year: the first year of the cases
    :return: the exit code of the script
    '''
    start = time.perf_counter()
    try:
        commit_a, commit_b = (_git(repo, 'rev-parse', '--verify', f'{ref}^{{commit}}') for ref in (ref_a, ref_b))
        print(f'A: {ref_a} ({commit_a[:7]})')
        print(f'B: {ref_b} ({commit_b[:7]})')
        print(f'Start year: {start_year}')
        if ignore:
            print(f'Ignored datasets: {", ".join(ignore)}')

        outputs = _common_outputs(repo, commit_a, commit_b)
        with tempfile.TemporaryDirectory(prefix='cfm_compare_') as tmp:
            tmp = Path(tmp).resolve()
            results = {}
            for label, commit in (('A', commit_a), ('B', commit_b)):
                with _temporary_worktree(repo, commit, tmp / f'worktree_{label}') as worktree:
                    for liquid in LIQUID_SCHEMES:
                        results[label, liquid] = _run_case(worktree, tmp / label / liquid, liquid, outputs, start_year)

            equal = True
            for liquid in LIQUID_SCHEMES:
                difference = compare_results(results['A', liquid], results['B', liquid], ignore)
                if difference is None:
                    with h5py.File(results['A', liquid], 'r') as f:
                        compared = sorted(set(f) - set(ignore))
                    print(f'{liquid}: equal ({len(compared)} datasets: {", ".join(compared)})')
                else:
                    equal = False
                    print(f'{liquid}: differs, {difference}')
    except RunFailed as error:
        print(f'FAILED: {error}')
        return 2

    print(f'Result: {"pass" if equal else "DIFFERS"}. Time: {time.perf_counter() - start:.0f} s')
    return 0 if equal else 1


def main():
    parser = argparse.ArgumentParser(description='Check that two commits give the same model results.')
    parser.add_argument('ref_a', help='a git ref, normally the base of the PR')
    parser.add_argument('ref_b', help='a git ref, normally HEAD')
    parser.add_argument('--ignore', action='append', default=[], metavar='DATASET',
                        help='skip this dataset of CFMresults.hdf5 (can repeat)')
    parser.add_argument('--start-year', type=int, default=CASE_SETTINGS['start_year'], metavar='YEAR',
                        help='the first year of the cases, 1980 or later. A later year has more water and takes longer')
    args = parser.parse_args()
    return run_check(args.ref_a, args.ref_b, args.ignore, start_year=args.start_year)


if __name__ == '__main__':
    sys.exit(main())
