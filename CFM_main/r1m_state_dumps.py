'''
r1m_state_dumps.py
==================
State dumps for the resingledomain (R1M) investigation.

Experiment code. A dump holds a copy of the model state before one call, so that
replay_r1m_dump.py can run that call again in a few seconds, without the full
run. It must not change the model results.

- snapshot_state() copies the state of step iii. Arrays over the nodes are
  copied in full. Arrays over the time steps keep only a window of
  2*TIME_WINDOW+1 steps around iii, and replay_step gives the index of iii in
  that window. The output writer, the forcing tables, and the lists that grow
  by one element each step are not copied, because a copy at each step of a
  long run would take O(n^2) time. resingledomain and the heat solve do not
  read them.
- write_dump() pickles a snapshot to r1m_dumps/ in resultsFolder.
- SLOW_CALL_S: a resingledomain call that takes longer than this is dumped. Set
  it with the environment variable CFM_R1M_DUMP_S (seconds, default 10).
'''

import copy
import os
import pickle

import numpy as np
import pandas as pd

DUMP_FOLDER = 'r1m_dumps'
SLOW_CALL_S = float(os.environ.get('CFM_R1M_DUMP_S', '10'))
TIME_WINDOW = 2  # time steps kept on each side of iii
SKIP = ('MOutputs', 'forcing_dict', 'SEBfluxes', 'dHAll', 'dHAllcorr')  # not read by resingledomain or the heat solve
MAX_ITEMS = 1000  # a dict, list or tuple with more items is not copied


def snapshot_state(self, iii):
    '''Return a copy of the state of step iii that a replay needs.'''
    nsteps = len(self.modeltime)
    first = max(iii - TIME_WINDOW, 0)
    last = min(iii + TIME_WINDOW + 1, nsteps)
    state = {}
    skipped = []
    for name, value in vars(self).items():
        if name in SKIP:
            skipped.append(name)
        elif isinstance(value, (np.ndarray, pd.Index)):
            if value.ndim >= 1 and value.shape[0] == nsteps and nsteps != len(self.dz):
                state[name] = value[first:last].copy()
            else:
                state[name] = value.copy()
        elif isinstance(value, (dict, list, tuple)) and len(value) <= MAX_ITEMS:
            state[name] = copy.deepcopy(value)
        elif value is None or isinstance(value, (bool, int, float, str, np.generic)):
            state[name] = value
        else:
            skipped.append(name)
    return {'state': state, 'step': iii, 'replay_step': iii - first,
            'modeltime': self.modeltime[iii], 'skipped': skipped}


def write_dump(self, snapshot, kind, **extra):
    '''
    Pickle a snapshot to r1m_dumps/<kind>_step<iii>.pkl in resultsFolder and
    return the path. kind names the call to replay ('resingledomain' or
    'heat_solve'). extra goes into the dump, for example the outputs of the call
    or the error message.
    '''
    folder = os.path.join(self.c['resultsFolder'], DUMP_FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{kind}_step{snapshot['step']}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(dict(snapshot, kind=kind, **extra), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"R1M dump: {kind}, modeltime {snapshot['modeltime']:.4f}, step {snapshot['step']}, {path}", flush=True)
    return path


def copy_outputs(outputs):
    '''Copy the return values of a call, so that later steps do not change them.'''
    return tuple(np.copy(v) if isinstance(v, np.ndarray) else copy.deepcopy(v) for v in outputs)
