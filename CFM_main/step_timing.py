'''
step_timing.py
==============
Wall time and CPU time of each time step of FirnDensityNoSpin.time_evolve.

The key "step_timing" (boolean, default false) switches it on. With the key off,
none of this code runs: each hook in time_evolve is one line behind "if timing:",
and the model has no attribute step_timing_stages.
The hooks only read the model state. They never assign to a model array, so the
results of a run are the same with the key on and off. This module imports only
standard modules and numpy, and no model module.

With the key on, a run gives:

- step_timing.csv in resultsFolder: one row for each time step. The file is a
  plain csv. The rows of one model year are written at the end of that year and
  the file is flushed, so a run that stops keeps the rows up to the last full
  year. An old step_timing.csv in resultsFolder is replaced at the start of the
  main run. The spin-up removes resultsFolder only if it holds nothing but
  .json and .hdf5 files, so an old file can stay until then.
- One line for each model year on the standard output, with its own timestamp.
- The dictionary firn.step_timing_stages on the model instance, for a program
  that runs the model (see STAGES).

The time step is split into five parts. A mark gives the end of a part:

- pre:  merging, densification, strain, surface energy balance, the melt factor of bare ice.
- melt: the melt scheme (the block "MELT"), or nothing on a dry step.
- mid:  firn air, isotopes, grain growth, accumulation, sublimation, stress.
- heat: heat diffusion or the refreezing solver.
- post: output storage, regridding, spin-up update, the budget arrays.

The columns of a row:

- step, modeltime, scheme. The scheme is "darcy", "bucket", "dry" (MELT is on, but
  no melt scheme ran in this step) or "nomelt" (MELT is off). It comes from
  model.melt_diagnostics when that dictionary belongs to this step. So with
  liquid "darcy" and a model time before bucket_until_year, the scheme is
  "bucket". The schemes "resingledomain" and "prefsnowpack" do not write
  melt_diagnostics, so their steps show as "dry".
- wall_s, cpu_s: the whole step. wall_<part>_s, cpu_<part>_s: the five parts,
  in seconds. The whole step is the sum of the parts. The times come from
  time.perf_counter() and time.process_time().
- nodes, wet_nodes_in: the number of nodes, and the number of nodes with LWC > 0,
  when the melt part starts.
- melt_mwe, rain_mwe: melt and rain of the step [m w.e.], when the melt part starts.
- wet_nodes_out: the number of nodes with LWC > 0 at the end of the step.
- substeps, root_solves, water_in_mwe, lwc_end_mwe, refrozen_mwe, runoff_mwe,
  budget_residual, n_wet_cold_layers, idle_substeps: from model.melt_diagnostics.
  Empty when the scheme does not give the value (root_solves and idle_substeps:
  the implicit flow step only; substeps: darcy only).

The small work of a mark (the counts of the nodes, the read of the melt) falls in the
part that follows the mark. The time of the csv rows is in no part: it is the
time between two steps, and it is not measured.
'''

import csv
import datetime
import os
import time

import numpy as np

CSV_NAME = 'step_timing.csv'

PARTS = ('pre', 'melt', 'mid', 'heat', 'post')

STAGES = ('loop_start', 'switch', 'loop_end', 'write_end')

DIAGNOSTIC_COLUMNS = (
    ('substeps', 'substeps'), ('root_solves', 'root_solves'), ('water_in_mwe', 'water_in'),
    ('lwc_end_mwe', 'lwc_end'), ('refrozen_mwe', 'refrozen'), ('runoff_mwe', 'runoff'),
)

CSV_COLUMNS = (
    ('step', 'modeltime', 'scheme', 'wall_s', 'cpu_s')
    + tuple(f'wall_{part}_s' for part in PARTS)
    + tuple(f'cpu_{part}_s' for part in PARTS)
    + ('nodes', 'wet_nodes_in', 'wet_nodes_out', 'melt_mwe', 'rain_mwe')
    + tuple(name for name, _ in DIAGNOSTIC_COLUMNS)
    + ('budget_residual', 'n_wet_cold_layers', 'idle_substeps')
)


def _now():
    return {'wall': time.perf_counter(), 'cpu': time.process_time()}


def _seconds(value):
    return '%.10f' % value


def _number(value):
    '''A value of melt_diagnostics as text, or an empty field.'''
    if value is None:
        return ''
    return '%.9g' % value


class StepTimer:
    '''
    Collects the time of each part of a time step and writes step_timing.csv.

    :param model: the FirnDensityNoSpin instance. The timer only reads it.
    :param mwe_per_rate: factor from a rate [m ice eq. per year] to a step amount [m w.e.]
        (seconds per year times the density of ice in Mg m-3). The caller gives it, because this
        module does not import the constants.

    The hooks in time_evolve, in the order of the step:

    - begin(iii): at the start of a step.
    - mark(part): at the end of the part with this name. mark('pre') also reads the nodes and the melt.
    - end_step(iii): at the end of the step. It ends 'post' and keeps the row.
    - finish(): after the loop. It writes the last year and closes the file.
    - mark_write_end(): after the output file is written.
    '''

    def __init__(self, model, mwe_per_rate):
        self.model = model
        self.mwe_per_rate = mwe_per_rate
        self.stages = {name: None for name in STAGES}
        self.file = open(os.path.join(model.c['resultsFolder'], CSV_NAME), 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_COLUMNS)
        self.year = None
        self.rows = []
        self.year_totals = None
        self.first_year_part = False
        self.n_steps = len(model.modeltime)
        self.bucket_until_year = getattr(model, 'bucket_until_year', None)
        self.step_years = (model.modeltime[1] - model.modeltime[0]) if self.n_steps > 1 else 0.
        self.parts_wall = {}
        self.parts_cpu = {}
        self.extra = {}
        model.step_timing_stages = self.stages
        self.stages['loop_start'] = _now()

    ### The hooks ###

    def begin(self, iii):
        '''Start a step. Write the previous model year if this step is in a new year.'''
        modeltime = self.model.modeltime[iii]
        year = int(np.floor(modeltime))
        if self.year is None:
            self._start_year(year, modeltime)
        elif year != self.year:
            self._write_year(part_year=self.first_year_part)
            self._start_year(year, modeltime)
        if self.stages['switch'] is None and self.bucket_until_year is not None \
                and modeltime >= self.bucket_until_year:
            self.stages['switch'] = _now()
        self.iii = iii
        self.t_prev = _now()

    def mark(self, part):
        '''End the part with this name.'''
        now = _now()
        self.parts_wall[part] = now['wall'] - self.t_prev['wall']
        self.parts_cpu[part] = now['cpu'] - self.t_prev['cpu']
        self.t_prev = now
        if part == 'pre':
            model = self.model
            self.extra['nodes'] = len(model.dz)
            self.extra['wet_nodes_in'] = int(np.count_nonzero(model.LWC > 0.))
            if model.MELT:
                self.extra['melt_mwe'] = model.snowmeltSec[self.iii] * self.mwe_per_rate
                self.extra['rain_mwe'] = model.rainSec[self.iii] * self.mwe_per_rate
            else:
                self.extra['melt_mwe'] = ''
                self.extra['rain_mwe'] = ''

    def end_step(self, iii):
        '''End the step: end the part post and keep the row.'''
        self.mark('post')
        model = self.model
        wall_parts = [self.parts_wall[part] for part in PARTS]
        cpu_parts = [self.parts_cpu[part] for part in PARTS]
        diagnostics = getattr(model, 'melt_diagnostics', None)
        if diagnostics is not None and diagnostics.get('step') == iii:
            scheme = diagnostics['scheme']
            checks = diagnostics.get('checks', {})
            cold = checks.get('cold', checks.get('cold_2', {}))
            values = [diagnostics.get(key) for _, key in DIAGNOSTIC_COLUMNS]
            residual = checks.get('budget', {}).get('residual')
            cold_layers = cold.get('n_wet_cold_layers')
            idle = diagnostics.get('idle_substeps')
        else:
            scheme = 'dry' if model.MELT else 'nomelt'
            values = [None] * len(DIAGNOSTIC_COLUMNS)
            residual = cold_layers = idle = None
        wall = sum(wall_parts)
        cpu = sum(cpu_parts)
        extra = self.extra
        melt_mwe = extra['melt_mwe']
        rain_mwe = extra['rain_mwe']
        row = [iii, '%.10g' % model.modeltime[iii], scheme, _seconds(wall), _seconds(cpu)]
        row += [_seconds(value) for value in wall_parts]
        row += [_seconds(value) for value in cpu_parts]
        row += [extra['nodes'], extra['wet_nodes_in'], int(np.count_nonzero(model.LWC > 0.)),
                _number(None if melt_mwe == '' else melt_mwe),
                _number(None if rain_mwe == '' else rain_mwe)]
        row += [_number(value) for value in values]
        row += [_number(residual), '' if cold_layers is None else int(cold_layers),
                '' if idle is None else int(idle)]
        self.rows.append(row)

        totals = self.year_totals
        totals['steps'] += 1
        totals['wall'] += wall
        totals['cpu'] += cpu
        totals['wall_melt'] += wall_parts[1]
        totals['wall_heat'] += wall_parts[3]
        totals['schemes'][scheme] = totals['schemes'].get(scheme, 0) + 1
        totals['substeps'] += values[0] or 0
        totals['root_solves'] += values[1] or 0
        if melt_mwe != '':
            totals['melt_mwe'] += melt_mwe

    def finish(self):
        '''Write the last model year and close the file.'''
        if self.year is not None:
            last = self.model.modeltime[self.n_steps - 1]
            part_year = last + self.step_years < self.year + 1 - 0.5 * self.step_years
            self._write_year(part_year=part_year or self.first_year_part)
        self.file.close()
        self.stages['loop_end'] = _now()

    def mark_write_end(self):
        '''Record the time after the output file is written.'''
        self.stages['write_end'] = _now()

    ### Private ###

    def _start_year(self, year, modeltime):
        if self.year is None:
            self.first_year_part = (modeltime - year) > 0.5 * self.step_years
        else:
            self.first_year_part = False
        self.year = year
        self.rows = []
        self.year_totals = {'steps': 0, 'wall': 0., 'cpu': 0., 'wall_melt': 0., 'wall_heat': 0.,
                            'schemes': {}, 'substeps': 0, 'root_solves': 0, 'melt_mwe': 0.}

    def _write_year(self, part_year):
        '''Write the rows of the model year to the file, flush it, and print the line of the year.'''
        self.writer.writerows(self.rows)
        self.file.flush()
        t = self.year_totals
        counts = ', '.join(f'{name} {t["schemes"].get(name, 0)}' for name in ('darcy', 'bucket', 'dry', 'nomelt')
                           if t['schemes'].get(name, 0) > 0 or name in ('darcy', 'bucket', 'dry'))
        stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        note = ' (part year)' if part_year else ''
        print(f'{stamp} step_timing year {self.year}{note}: wall {t["wall"]:.2f} s, cpu {t["cpu"]:.2f} s, '
              f'steps {t["steps"]} ({counts}), substeps {t["substeps"]}, root_solves {t["root_solves"]}, '
              f'melt {t["melt_mwe"]:.3f} m we, wall melt {t["wall_melt"]:.2f} s, '
              f'wall heat {t["wall_heat"]:.2f} s', flush=True)
        self.rows = []
