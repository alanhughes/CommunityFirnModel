'''
Tests for the per-node refreeze self.refrozen: its sum, its node order, and how it
reaches the output.
'''

import h5py
import numpy as np
import pytest

import melt
import sublim
from constants import S_PER_YEAR

N_NODES = 60
DENSITY = 400.0
NODE_THICKNESS = 0.1
MELT_THREE_NODES = 2.5 * DENSITY * NODE_THICKNESS / 917.0  # [m ie]: removes two nodes and part of a third


def _melt_step(synthetic_column, scheme, **overrides):
    '''Call one scheme once on a cold column. Return the column and the column total [m w.e.].'''
    settings = dict(n=N_NODES, rho_profile=DENSITY, T_profile=263.15, dz=NODE_THICKNESS, melt=MELT_THREE_NODES)
    settings.update(overrides)
    column = synthetic_column(**settings)
    scheme_outputs = getattr(melt, scheme)(column, 0)
    return column, scheme_outputs[10]


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_sum_is_column_total_with_melt_removal(synthetic_column, scheme):
    column, refreeze_total = _melt_step(synthetic_column, scheme)
    assert refreeze_total > 0.0
    assert column.refrozen.shape == column.dz.shape
    assert column.refrozen.sum() == pytest.approx(refreeze_total, rel=1e-12)
    assert column.refrozen.min() >= 0.0


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_refrozen_is_mass_gain_of_each_node(synthetic_column, scheme):
    # Below the melted nodes, each node starts with the same mass. So the mass
    # gain of a node shows its refreeze, and this tests the node order.
    column, _ = _melt_step(synthetic_column, scheme)
    mass_gain = column.mass - DENSITY * column.dz
    assert column.refrozen[1:] * 1000.0 == pytest.approx(mass_gain[1:], abs=1e-8)
    assert np.any(column.refrozen[1:] > 0.0)


@pytest.mark.scheme
def test_darcy_moves_refreeze_of_melted_nodes_to_top_node(synthetic_column):
    # The first sub-step refreezes water in the top nodes. The later sub-steps melt
    # those nodes. Their refreeze is then in the new top node, beyond the mass gain.
    column, refreeze_total = _melt_step(synthetic_column, 'darcyscheme')
    top_node_mass_gain = column.mass[0] - DENSITY * column.dz[0]
    assert column.refrozen[0] * 1000.0 > top_node_mass_gain + 1e-6
    assert column.refrozen[1:].sum() < refreeze_total


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_refrozen_replaces_the_value_of_the_last_step(synthetic_column, scheme):
    column = synthetic_column(n=N_NODES, rho_profile=DENSITY, T_profile=273.15, dz=NODE_THICKNESS, melt=0.01)
    column.refrozen = np.full(N_NODES, 1.0)
    getattr(melt, scheme)(column, 0)
    assert column.refrozen.sum() == 0.0  # a column at the melting point refreezes nothing


@pytest.mark.scheme
def test_darcy_with_only_rain_keeps_node_order(synthetic_column):
    column, refreeze_total = _melt_step(synthetic_column, 'darcyscheme', melt=0.0, rain=0.05)
    assert refreeze_total > 0.0
    assert column.refrozen.sum() == pytest.approx(refreeze_total, rel=1e-12)
    assert column.refrozen * 1000.0 == pytest.approx(column.mass - DENSITY * column.dz, abs=1e-8)


@pytest.mark.unit
def test_sublim_moves_refreeze_of_removed_nodes_to_top_node(synthetic_column):
    column = synthetic_column(n=N_NODES, rho_profile=DENSITY, dz=NODE_THICKNESS)
    column.MELT = True
    column.PLWC_mem = np.zeros(N_NODES)
    column.totwatersublim = 0.0
    column.Ts = np.array([263.15])
    sublimated_mass = 2.5 * DENSITY * NODE_THICKNESS  # removes two nodes and part of a third [kg m-2]
    column.sublimSec = np.array([-sublimated_mass / 917.0 / S_PER_YEAR])
    column.refrozen = np.arange(1.0, N_NODES + 1.0)
    before = column.refrozen.copy()
    sublim.sublim(column, 0)
    assert column.refrozen.sum() == pytest.approx(before.sum())
    assert column.refrozen[0] == before[0:3].sum()
    assert np.array_equal(column.refrozen[1:N_NODES - 3], before[3:N_NODES - 1])


def _results(run, *names):
    with h5py.File(run.results_path, 'r') as f:
        return tuple(f[name][()] for name in names)


@pytest.mark.conservation
@pytest.mark.parametrize('liquid', ['bucket', 'darcy'])
def test_output_sum_is_column_total_in_a_full_run(short_config, liquid):
    # A full run has dry steps, new surface nodes, and refreeze in the enthalpy solver.
    run = short_config(years=2, start_year=1995, liquid=liquid, outputs=['refrozen', 'meltoutputs'],
                       grid_outputs=False, output_bits='float64')
    run.run()
    refrozen, refreeze = _results(run, 'refrozen', 'refreeze')
    assert refreeze[1:, 1].sum() > 0.0
    assert np.any(refreeze[1:, 1] == 0.0)  # the dry steps are in the test
    assert refrozen[1:, 1:].sum(axis=1) == pytest.approx(refreeze[1:, 1], abs=1e-12)


@pytest.mark.conservation
def test_output_lines_up_with_depth(short_config, monkeypatch):
    # Replace the refreeze of bucket by the age of each node, as a marker. The enthalpy
    # solver adds at most the liquid water [m], and the marker is in seconds. So the
    # output shows the node order after the new surface node.
    import firn_density_nospin

    bucket = melt.bucket

    def marked_bucket(self, iii):
        outputs = bucket(self, iii)
        self.refrozen = self.age.copy()
        return outputs

    monkeypatch.setattr(firn_density_nospin, 'bucket', marked_bucket)
    run = short_config(years=2, start_year=1995, liquid='bucket', outputs=['refrozen', 'age', 'meltoutputs'],
                       grid_outputs=False, output_bits='float64')
    run.run()
    refrozen, age = _results(run, 'refrozen', 'age')
    wet_steps = np.nonzero(refrozen[1:, 1:].max(axis=1) > 1.0)[0] + 1
    assert wet_steps.size > 0
    # The age is written in years, and each node has aged by the same step since the
    # marker. So the difference is one value in each step. The top node has no marker.
    difference = refrozen[wet_steps, 2:] - age[wet_steps, 2:] * S_PER_YEAR
    assert np.ptp(difference, axis=1).max() < 1.0


@pytest.mark.unit
def test_gridded_output_conserves_the_amount():
    # An amount per node must not be interpolated between nodes.
    from ModelOutputs import ModelOutputs

    z = np.arange(0.0, 10.0, 0.5)
    amount = np.zeros_like(z)
    amount[6] = 0.3  # one node refreezes
    outputs = ModelOutputs({'grid_outputs': True, 'grid_output_res': 0.25, 'output_bits': 'float64'},
                           {'z': z, 'refrozen': amount}, 1, 0.0, len(z))
    outputs.updateMO({'z': z, 'refrozen': amount}, 1.0, 1)
    assert outputs.Mout_dict['refrozen'][0, 1:].sum() == pytest.approx(0.3)
    assert outputs.Mout_dict['refrozen'][1, 1:].sum() == pytest.approx(0.3)
