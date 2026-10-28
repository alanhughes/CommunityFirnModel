'''
replay_r1m_dump.py
==================
Run one call again from a dump of r1m_state_dumps.py. Experiment code for the
resingledomain (R1M) investigation.

Usage, from CFM_main:
    python replay_r1m_dump.py <dump.pkl> [output folder]

- A 'resingledomain' dump: run resingledomain on the saved state. The counter
  row goes to r1m_step_counters.csv in the output folder (default: a folder
  next to the dump). If the dump holds the outputs of the call in the run,
  check that the replay gives the same values.
- A 'heat_solve' dump: print the column geometry and the nodes where the node
  weights of the heat solve are not finite. Then run the heat solve of the step
  again and give the first node with non-finite Tz.
'''

import os
import pickle
import sys
import time

import numpy as np

from firn_density_nospin import FirnDensityNoSpin
from re_snowpack import resingledomain
from diffusion import heatDiff, refreezeDiff
from r1m_step_counters import column_geometry

RESINGLEDOMAIN_OUTPUTS = ('rho', 'age', 'dz', 'Tz', 'z', 'mass', 'dzn', 'LWC', 'PLWC_mem', 'r2',
                          'refrozen', 'Trunoff', 'dh_melt')


def load_model(dump, out_folder):
    '''Return a FirnDensityNoSpin with the saved state, without __init__.'''
    model = FirnDensityNoSpin.__new__(FirnDensityNoSpin)
    model.__dict__.update(dump['state'])
    model.c['resultsFolder'] = out_folder
    for name in ('r1m_year', 'r1m_year_totals', 'r1m_geometry_rows'):
        model.__dict__.pop(name, None)  # the counter files then start again with a header
    return model


def replay_resingledomain(dump, model, step):
    start = time.perf_counter()
    outputs = resingledomain(model, step)
    print(f'wall time {time.perf_counter() - start:.2f} s')
    if 'outputs' not in dump:
        print('the dump has no outputs of the run to compare')
        return
    differ = []
    for name, ran, replayed in zip(RESINGLEDOMAIN_OUTPUTS, dump['outputs'], outputs):
        if not np.array_equal(np.asarray(ran), np.asarray(replayed), equal_nan=True):
            differ.append(name)
    print('outputs equal to the run' if not differ else f'outputs differ from the run: {differ}')


def heat_solve_weights(z):
    '''Return the indices where the node weights f_u, f_d of the heat solve are not finite.'''
    z_edges = np.append(z, z[-1] + np.diff(z)[-1])  # as in diffusion.py
    Z_P = (z_edges[1:] + z_edges[:-1]) / 2
    Z_P_diff = np.diff(Z_P)
    deltaZ_u = np.append(Z_P_diff[0], Z_P_diff)
    deltaZ_d = np.append(Z_P_diff, Z_P_diff[-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        f_u = 1 - (Z_P - z_edges[:-1]) / deltaZ_u
        f_d = 1 - (z_edges[1:] - Z_P) / deltaZ_d
    return np.where(~np.isfinite(f_u))[0], np.where(~np.isfinite(f_d))[0]


def replay_heat_solve(dump, model, step):
    print(f"error in the run: {dump.get('error')}")
    for key, value in column_geometry(model.dz, model.z).items():
        print(f'  {key}: {value!r}')
    bad_u, bad_d = heat_solve_weights(model.z)
    print(f'non-finite f_u at {len(bad_u)} nodes {bad_u[:10]}, f_d at {len(bad_d)} nodes {bad_d[:10]}')
    for node in np.union1d(bad_u, bad_d)[:5]:
        lo, hi = max(node - 2, 0), min(node + 3, len(model.z))
        print(f'  around node {node}: z {model.z[lo:hi]!r}, dz {model.dz[lo:hi]!r}')

    if model.MELT and np.any(model.LWC > 0.):  # the branch order of time_evolve
        print(f'refreezeDiff, solver {model.melt_solver}')
        Tz = refreezeDiff(model, step, solver_name=model.melt_solver)[0]
    else:
        print('heatDiff')
        Tz = heatDiff(model, step)[0]
    bad = np.where(~np.isfinite(Tz))[0]
    print(f'Tz non-finite at {len(bad)} of {len(Tz)} nodes' + (f', first node {bad[0]}' if len(bad) else ''))


def main():
    path = sys.argv[1]
    with open(path, 'rb') as f:
        dump = pickle.load(f)
    out_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(path)[0] + '_replay'
    os.makedirs(out_folder, exist_ok=True)
    model = load_model(dump, out_folder)
    step = dump['replay_step']
    print(f"{dump['kind']} dump: modeltime {dump['modeltime']}, step {dump['step']}, "
          f"not saved: {dump['skipped']}")
    if dump['kind'] == 'resingledomain':
        replay_resingledomain(dump, model, step)
    elif dump['kind'] == 'heat_solve':
        replay_heat_solve(dump, model, step)


if __name__ == '__main__':
    main()
