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
MELT_FORCING = 'CFM_example_66.5_-46.25.pkl'  # DYE-2, which has melt


def _melt_step(synthetic_column, scheme, **overrides):
    '''Call one scheme once on a cold column. Return the column and the column total [m w.e.].'''
    settings = dict(n=N_NODES, rho_profile=DENSITY, T_profile=263.15, dz=NODE_THICKNESS, melt=MELT_THREE_NODES)
    settings.update(overrides)
    column = synthetic_column(**settings)
    scheme_outputs = getattr(melt, scheme)(column, 0)
    return column, scheme_outputs[10]


@pytest.mark.scheme
def test_bucket_sum_is_column_total_with_melt_removal(synthetic_column):
    column, refreeze_total = _melt_step(synthetic_column, 'bucket')
    assert refreeze_total > 0.0
    assert column.refrozen.shape == column.dz.shape
    assert column.refrozen.sum() == pytest.approx(refreeze_total, rel=1e-12)
    assert column.refrozen.min() >= 0.0


@pytest.mark.scheme
def test_darcy_sum_leaves_out_the_refreeze_of_melted_nodes(synthetic_column):
    # The first sub-step refreezes water in the top nodes. The later sub-steps melt
    # those nodes. The column total counts this refreeze, and self.refrozen does not.
    column, refreeze_total = _melt_step(synthetic_column, 'darcyscheme')
    assert column.refrozen.shape == column.dz.shape
    assert column.refrozen.min() >= 0.0
    assert 0.0 < column.refrozen.sum() < refreeze_total


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
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_refrozen_replaces_the_value_of_the_last_step(synthetic_column, scheme):
    column = synthetic_column(n=N_NODES, rho_profile=DENSITY, T_profile=273.15, dz=NODE_THICKNESS, melt=0.01)
    column.refrozen = np.full(N_NODES, 1.0)
    getattr(melt, scheme)(column, 0)
    assert column.refrozen.sum() == 0.0  # a column at the melting point refreezes nothing


@pytest.mark.scheme
def test_bucket_adds_the_second_refreeze(synthetic_column):
    # Water under a lens has no room, so it ponds above the lens into cold nodes
    # after the first refreeze. Only the second refreeze can freeze it.
    rho, T, lwc = np.full(40, 400.0), np.full(40, 273.15), np.zeros(40)
    rho[10:13], rho[21:] = 900.0, 900.0
    T[:13], T[21:] = 263.15, 263.15
    lwc[13:21] = 0.08
    column = synthetic_column(n=40, rho_profile=rho, T_profile=T, dz=NODE_THICKNESS, LWC=lwc,
                              config=dict(Ponding=True, DirectRunoff=0.0, RunoffZuoOerlemans=False, DownToIce=False,
                                          ThickImp=0.1, RhoImp=830.0, ColeouLesaffre=True))
    refreeze_total = melt.bucket(column, 0)[10]
    assert refreeze_total > 0.0
    assert column.refrozen.sum() == pytest.approx(refreeze_total, rel=1e-12)
    assert np.all(column.refrozen[:5] == 0.0) and np.all(column.refrozen[5:10] > 0.0)


@pytest.mark.scheme
def test_darcy_keeps_the_refreeze_of_the_bottom_node(synthetic_column):
    # The bottom node refreezes in the first sub-step. The later sub-steps melt only
    # part of the top node, so the melt removal must keep the bottom value.
    lwc = np.zeros(N_NODES)
    lwc[-1] = 1e-3
    column, refreeze_total = _melt_step(synthetic_column, 'darcyscheme', melt=1e-3, LWC=lwc)
    assert column.refrozen[-1] > 0.0
    assert column.refrozen.sum() == pytest.approx(refreeze_total, rel=1e-12)


@pytest.mark.scheme
def test_darcy_with_only_rain_keeps_node_order(synthetic_column):
    column, refreeze_total = _melt_step(synthetic_column, 'darcyscheme', melt=0.0, rain=0.05)
    assert refreeze_total > 0.0
    assert column.refrozen.sum() == pytest.approx(refreeze_total, rel=1e-12)
    assert column.refrozen * 1000.0 == pytest.approx(column.mass - DENSITY * column.dz, abs=1e-8)


@pytest.mark.unit
def test_sublim_shifts_the_refreeze_and_keeps_the_bottom_node(synthetic_column):
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
    # Two nodes go. The partly sublimated node keeps its value, and the two new nodes have none.
    assert np.array_equal(column.refrozen[:N_NODES - 3], before[2:N_NODES - 1])
    assert column.refrozen[N_NODES - 3] == before[-1]
    assert np.all(column.refrozen[N_NODES - 2:] == 0.0)


def _results(run, *names):
    with h5py.File(run.results_path, 'r') as f:
        return tuple(f[name][()] for name in names)


@pytest.mark.conservation
def test_output_sum_is_column_total_in_a_full_run(short_config):
    # A full run has dry steps, new surface nodes, and refreeze in the enthalpy solver.
    # test_refrozen_output.py tests the darcy sum, which leaves out the melted nodes.
    run = short_config(years=2, start_year=1980, liquid='bucket', DFfile=MELT_FORCING, outputs=['refrozen', 'meltoutputs'],
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
    run = short_config(years=2, start_year=1980, liquid='bucket', DFfile=MELT_FORCING, outputs=['refrozen', 'age', 'meltoutputs'],
                       grid_outputs=False, output_bits='float64')
    run.run()
    refrozen, age = _results(run, 'refrozen', 'age')
    wet_steps = np.nonzero(refrozen[1:, 1:].max(axis=1) > 1.0)[0] + 1
    assert wet_steps.size > 0
    # The age is written in years, and each node has aged by the same step since the
    # marker. So the difference is one value in each step. The top node has no marker.
    difference = refrozen[wet_steps, 2:] - age[wet_steps, 2:] * S_PER_YEAR
    assert np.ptp(difference, axis=1).max() < 1.0


def _gridded(amount, varname='refrozen', z=np.arange(0.0, 10.0, 0.5), res=0.25):
    '''Return the grid depths and the first and second output rows of one variable on the grid.'''
    from ModelOutputs import ModelOutputs

    state = {'z': z, varname: amount}
    outputs = ModelOutputs({'grid_outputs': True, 'grid_output_res': res, 'output_bits': 'float64'}, state, 1, 0.0, len(z))
    outputs.updateMO(state, 1.0, 1)
    rows = outputs.Mout_dict[varname]
    return z[0] + np.arange(rows.shape[1] - 1) * res, rows[0, 1:], rows[1, 1:]


def _one_node(node, n_nodes=20):
    amount = np.zeros(n_nodes)
    amount[node] = 0.3
    return amount


@pytest.mark.unit
@pytest.mark.parametrize('varname', ['refrozen', 'LWC'])
@pytest.mark.parametrize('node, depths', [(0, [0.0, 0.25]), (6, [3.0, 3.25]), (18, [9.0, 9.25])])
def test_gridded_output_places_the_amount_in_its_node(varname, node, depths):
    # Each node is 0.5 m thick and each grid interval is 0.25 m, so a node fills two
    # intervals. The value at a grid depth is the amount from that depth to the next.
    grid, first_row, later_row = _gridded(_one_node(node), varname)
    assert grid[first_row > 0.0] == pytest.approx(depths)
    assert first_row[first_row > 0.0] == pytest.approx([0.15, 0.15])
    np.testing.assert_array_equal(later_row, first_row)


@pytest.mark.unit
def test_gridded_output_conserves_the_amount_with_a_coarse_grid():
    # A grid interval holds parts of several nodes. The last interval holds all that is below its top.
    amount = np.arange(1.0, 21.0)
    _, row, _ = _gridded(amount, res=0.8)
    assert row[0] == pytest.approx(amount[0] + 0.6 * amount[1])
    assert np.nansum(row) == pytest.approx(amount.sum())


@pytest.mark.unit
def test_gridded_output_stops_at_the_bottom_of_the_column():
    # The grid is made again at each write, so a shorter column has NaN below its depth.
    z = np.arange(0.0, 10.0, 0.5)
    from ModelOutputs import ModelOutputs

    outputs = ModelOutputs({'grid_outputs': True, 'grid_output_res': 0.25, 'output_bits': 'float64'},
                           {'z': z, 'refrozen': np.zeros(20)}, 1, 0.0, 20)
    short_z = np.arange(0.0, 5.0, 0.5)
    amount = np.full(10, 0.1)
    outputs.updateMO({'z': short_z, 'refrozen': amount}, 1.0, 1)
    row = outputs.Mout_dict['refrozen'][1, 1:]
    grid = np.arange(row.size) * 0.25
    assert np.nansum(row) == pytest.approx(1.0)
    assert np.all(np.isnan(row[grid > 4.5]))
