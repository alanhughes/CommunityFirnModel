'''
r1m_step_counters.py
====================
Counters and state checks for the resingledomain (R1M) meltwater scheme.

Experiment code. It records how the sub-step control of re_snowpack.py spends
its time, and it stops the run at the first non-finite value in the column.
It must not change the model results.

- new_call_counters() starts the counters of one resingledomain call.
- record_call() writes one row to r1m_step_counters.csv in resultsFolder and
  prints one summary line for each model year.
- check_column() and check_finite_arrays() raise NonFiniteStateError at the
  first non-finite value or dz <= 0. check_column() also raises if z does not
  increase, and it can write a state dump before it raises.
- record_geometry() writes one row for each time step to r1m_geometry.csv: the
  thinnest node, the smallest step in z, and the largest difference between z
  and the cumulative sum of dz.
'''

import os
import time

import numpy as np

from constants import S_PER_YEAR, RHO_I_MGM
import r1m_state_dumps

CSV_NAME = 'r1m_step_counters.csv'
GEOMETRY_CSV_NAME = 'r1m_geometry.csv'

CSV_COLUMNS = (
    'modeltime', 'step', 'wall_time_s', 'dt_s', 'melt_mwe', 'rain_mwe',
    'max_lwc_m', 'domain_nodes', 'domain_depth_m', 'max_bottom', 'windows',
    'accepted_substeps', 'newton_iterations', 'influx_halvings', 'backsteps',
    'backstep_iteration16', 'backstep_positive_head', 'backstep_head_jump',
    'backstep_effsat_above_1', 'overflow_dtheta_dh', 'overflow_theta_update',
    'nonfinite_head_solves', 'tdma_zero_pivot', 'nonfinite_accepted_substeps',
    'nonfinite_head_window_end', 'min_deltat_s', 'mean_deltat_s',
    'substeps_below_1s', 'substeps_below_1e-10s', 'guard_hits',
    'max_bottom_depth_m', 'min_dz_newton_m', 'min_cfm_dz_entry_m', 'min_cfm_dz_exit_m',
)

GEOMETRY_COLUMNS = (
    'modeltime', 'step', 'nodes', 'surface_dz_m', 'min_dz_m', 'min_dz_node',
    'min_dz_depth_m', 'min_diff_z_m', 'min_diff_z_node', 'max_z_mismatch_m',
    'nodes_dz_below_1mm',
)

PROGRESS_AFTER_S = 60.  # a call that takes longer prints a progress line every PROGRESS_EVERY windows
PROGRESS_EVERY = 10


class NonFiniteStateError(FloatingPointError):
    '''A state array has a non-finite value, or dz <= 0.'''


class OverflowCounter:
    '''
    Error callback for np.errstate(over='call', call=...). NumPy calls it once
    for each array operation that overflows, not once for each element.
    '''

    def __init__(self):
        self.count = 0

    def __call__(self, kind, flag):
        self.count += 1


def new_call_counters(self, iii):
    '''Return the counters of one resingledomain call, with its inputs.'''
    return {
        'start': time.perf_counter(),
        'modeltime': self.modeltime[iii],
        'step': iii,
        'dt_s': self.dt[iii],
        'melt_mwe': self.snowmeltSec[iii] * S_PER_YEAR * RHO_I_MGM,
        'rain_mwe': self.rainSec[iii] * S_PER_YEAR * RHO_I_MGM,
        'max_lwc_m': np.max(self.LWC),
        'domain_nodes': 0,
        'domain_depth_m': 0.,
        'max_bottom': 0,
        'windows': 0,
        'accepted_substeps': 0,
        'newton_iterations': 0,
        'influx_halvings': 0,
        'backsteps': 0,
        'backstep_iteration16': 0,
        'backstep_positive_head': 0,
        'backstep_head_jump': 0,
        'backstep_effsat_above_1': 0,
        'overflow_dtheta_dh': OverflowCounter(),
        'overflow_theta_update': OverflowCounter(),
        'nonfinite_head_solves': 0,
        'tdma_zero_pivot': 0,
        'nonfinite_accepted_substeps': 0,
        'nonfinite_head_window_end': 0,
        'min_deltat_s': np.inf,
        'sum_deltat_s': 0.,
        'substeps_below_1s': 0,
        'substeps_below_1e-10s': 0,
        'guard_hits': 0,
        'max_bottom_depth_m': 0.,
        'min_dz_newton_m': np.inf,
        'min_cfm_dz_entry_m': np.min(self.dz),
    }


def count_bottom(counters, dz, bottom):
    '''Record the deepest bottom and the thinnest layer of the Newton domain 0..bottom.'''
    counters['max_bottom'] = max(counters['max_bottom'], bottom)
    counters['max_bottom_depth_m'] = max(counters['max_bottom_depth_m'], np.sum(dz[0:(bottom+1)]))
    counters['min_dz_newton_m'] = min(counters['min_dz_newton_m'], np.min(dz[0:(bottom+1)]))


def count_backstep(counters, Miteration, Mhead, delta_Mhead2, MeffSat, bottom):
    '''
    Count one back step and each of the four tests that are true. More than one
    test can be true in the same back step.
    '''
    counters['backsteps'] += 1
    if Miteration == 16:
        counters['backstep_iteration16'] += 1
    if np.any(Mhead[0:(bottom+1)] > 0):
        counters['backstep_positive_head'] += 1
    if np.any(abs(delta_Mhead2[0:(bottom+1)]) > 1000):
        counters['backstep_head_jump'] += 1
    if np.max(MeffSat[0:(bottom+1)]) > 1.0:
        counters['backstep_effsat_above_1'] += 1


def count_accepted_substep(counters, Mdeltat, Mhead, Mtheta, Mlwc):
    '''Count one accepted sub-step and its length.'''
    counters['accepted_substeps'] += 1
    counters['min_deltat_s'] = min(counters['min_deltat_s'], Mdeltat)
    counters['sum_deltat_s'] += Mdeltat
    if Mdeltat < 1.:
        counters['substeps_below_1s'] += 1
    if Mdeltat < 1e-10:
        counters['substeps_below_1e-10s'] += 1
    if not (np.all(np.isfinite(Mhead)) and np.all(np.isfinite(Mtheta)) and np.all(np.isfinite(Mlwc))):
        counters['nonfinite_accepted_substeps'] += 1


def report_progress(counters, Mdeltat_new, bottom):
    '''In a long call, print a progress line every PROGRESS_EVERY windows.'''
    elapsed = time.perf_counter() - counters['start']
    if elapsed > PROGRESS_AFTER_S and counters['windows'] % PROGRESS_EVERY == 0:
        print(f"R1M progress: modeltime {counters['modeltime']:.4f}, window {counters['windows']}, "
              f"{elapsed:.0f} s, accepted {counters['accepted_substeps']}, "
              f"newton {counters['newton_iterations']}, backsteps {counters['backsteps']}, "
              f"min deltat {counters['min_deltat_s']:.3g} s, next deltat {Mdeltat_new:.3g} s, "
              f"bottom {bottom}", flush=True)


def _csv_row(counters):
    row = dict(counters)
    row['wall_time_s'] = time.perf_counter() - counters['start']
    row['overflow_dtheta_dh'] = counters['overflow_dtheta_dh'].count
    row['overflow_theta_update'] = counters['overflow_theta_update'].count
    if counters['accepted_substeps'] > 0:
        row['mean_deltat_s'] = counters['sum_deltat_s'] / counters['accepted_substeps']
    else:
        row['min_deltat_s'] = np.nan
        row['mean_deltat_s'] = np.nan
    return row


def _print_year_summary(year, totals):
    print(f"R1M year {year}: calls {totals['calls']}, wall {totals['wall_time_s']:.1f} s, "
          f"dry calls {totals['dry_calls']}, newton {totals['newton_iterations']}, "
          f"backsteps {totals['backsteps']}, min deltat {totals['min_deltat_s']:.3g} s, "
          f"overflows {totals['overflows']}, nonfinite heads {totals['nonfinite_head_solves']}, "
          f"guard hits {totals['guard_hits']}, max bottom {totals['max_bottom']}", flush=True)


def _new_year_totals():
    return {'calls': 0, 'wall_time_s': 0., 'dry_calls': 0, 'newton_iterations': 0,
            'backsteps': 0, 'min_deltat_s': np.inf, 'overflows': 0,
            'nonfinite_head_solves': 0, 'guard_hits': 0, 'max_bottom': 0}


def record_call(self, counters):
    '''
    Append the row of one call to r1m_step_counters.csv in resultsFolder. The
    first call of a model run writes the file again with a header. The file is
    closed after each row, so the rows stay on disk if the run crashes.
    '''
    row = _csv_row(counters)
    row['min_cfm_dz_exit_m'] = np.min(self.dz)
    path = os.path.join(self.c['resultsFolder'], CSV_NAME)
    first_call = not hasattr(self, 'r1m_year_totals')
    with open(path, 'w' if first_call else 'a') as f:
        if first_call:
            f.write(','.join(CSV_COLUMNS) + '\n')
        f.write(','.join(repr(float(row[k])) if isinstance(row[k], (float, np.floating)) else str(row[k])
                         for k in CSV_COLUMNS) + '\n')

    year = int(np.floor(row['modeltime']))
    if first_call:
        self.r1m_year = year
        self.r1m_year_totals = _new_year_totals()
    elif year != self.r1m_year:
        _print_year_summary(self.r1m_year, self.r1m_year_totals)
        self.r1m_year = year
        self.r1m_year_totals = _new_year_totals()
    totals = self.r1m_year_totals
    totals['calls'] += 1
    totals['wall_time_s'] += row['wall_time_s']
    if row['melt_mwe'] == 0. and row['rain_mwe'] == 0.:
        totals['dry_calls'] += 1
    totals['newton_iterations'] += row['newton_iterations']
    totals['backsteps'] += row['backsteps']
    totals['min_deltat_s'] = min(totals['min_deltat_s'], counters['min_deltat_s'])
    totals['overflows'] += row['overflow_dtheta_dh'] + row['overflow_theta_update']
    totals['nonfinite_head_solves'] += row['nonfinite_head_solves']
    totals['guard_hits'] += row['guard_hits']
    totals['max_bottom'] = max(totals['max_bottom'], row['max_bottom'])


def check_finite_arrays(where, modeltime, dz, **arrays):
    '''
    Raise NonFiniteStateError if dz or one of the arrays has a non-finite value,
    or if dz has a value <= 0. The message gives the place, the model time, the
    array, the first bad node and its value.
    '''
    for name, values in (('dz', dz),) + tuple(arrays.items()):
        bad = np.where(~np.isfinite(values))[0]
        if len(bad) > 0:
            raise NonFiniteStateError(
                f'{where}: modeltime {modeltime}, {name} non-finite at {len(bad)} of {len(values)} nodes, '
                f'first node {bad[0]} value {values[bad[0]]}')
    bad = np.where(dz <= 0.)[0]
    if len(bad) > 0:
        raise NonFiniteStateError(
            f'{where}: modeltime {modeltime}, dz <= 0 at {len(bad)} of {len(dz)} nodes, '
            f'first node {bad[0]} value {dz[bad[0]]}')


def column_geometry(dz, z):
    '''
    Return the thinnest node, the smallest step in z, and the largest difference
    between z and the cumulative sum of dz. The heat solve uses z, not dz, so a
    step of zero in z gives a 0/0 in its node weights.
    '''
    z_from_dz = np.concatenate(([0.], np.cumsum(dz)[:-1]))
    diff_z = np.diff(z)
    i_dz = int(np.argmin(dz))
    i_z = int(np.argmin(diff_z))
    return {'nodes': len(dz), 'surface_dz_m': dz[0], 'min_dz_m': dz[i_dz], 'min_dz_node': i_dz,
            'min_dz_depth_m': z[i_dz], 'min_diff_z_m': diff_z[i_z], 'min_diff_z_node': i_z,
            'max_z_mismatch_m': np.max(np.abs(z - z_from_dz)),
            'nodes_dz_below_1mm': int(np.sum(dz < 1e-3))}


def _geometry_text(dz, z):
    return ', '.join(f'{k} {v:.6g}' if isinstance(v, float) else f'{k} {v}'
                     for k, v in column_geometry(dz, z).items())


def check_column(self, iii, where, dump=None):
    '''
    Check z, dz, rho, Tz and LWC of the model column, and check that z
    increases. If a check fails and dump is a snapshot from
    r1m_state_dumps.snapshot_state(), write it as a 'heat_solve' dump first.
    '''
    try:
        check_finite_arrays(where, self.modeltime[iii], self.dz,
                            z=self.z, rho=self.rho, Tz=self.Tz, LWC=self.LWC)
        bad = np.where(np.diff(self.z) <= 0.)[0]
        if len(bad) > 0:
            raise NonFiniteStateError(
                f'{where}: modeltime {self.modeltime[iii]}, z does not increase at {len(bad)} of '
                f'{len(self.z)-1} steps, first node {bad[0]}, z {float(self.z[bad[0]])!r} then {float(self.z[bad[0]+1])!r}')
    except NonFiniteStateError as error:
        message = f'{error}; geometry: {_geometry_text(self.dz, self.z)}'
        if dump is not None:
            path = r1m_state_dumps.write_dump(self, dump, 'heat_solve', where=where, error=message)
            message = f'{message}; dump {path}'
        raise NonFiniteStateError(message) from None


def record_geometry(self, iii):
    '''
    Append one row to r1m_geometry.csv in resultsFolder. The first call of a
    model run writes the file again with a header.
    '''
    row = column_geometry(self.dz, self.z)
    row['modeltime'] = self.modeltime[iii]
    row['step'] = iii
    path = os.path.join(self.c['resultsFolder'], GEOMETRY_CSV_NAME)
    first_row = not hasattr(self, 'r1m_geometry_rows')
    with open(path, 'w' if first_row else 'a') as f:
        if first_row:
            f.write(','.join(GEOMETRY_COLUMNS) + '\n')
            self.r1m_geometry_rows = 0
        f.write(','.join(repr(float(row[k])) if isinstance(row[k], (float, np.floating)) else str(row[k])
                         for k in GEOMETRY_COLUMNS) + '\n')
    self.r1m_geometry_rows += 1


def print_last_year_summary(self):
    '''Print the summary line of the last model year, at the end of the run.'''
    if hasattr(self, 'r1m_year_totals'):
        _print_year_summary(self.r1m_year, self.r1m_year_totals)
