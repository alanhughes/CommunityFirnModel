'''
Single calls of sublim.sublim on a synthetic column.
'''

import numpy as np
import pytest

import sublim
from constants import S_PER_YEAR


@pytest.mark.scheme
@pytest.mark.parametrize('sublimation_ie, nodes_added', [(0.001, 1), (0.0273, 2), (0.055, 3)])
def test_sublimation_keeps_the_bottom_water(synthetic_column, sublimation_ie, nodes_added):
    # Each node holds 20 kg m-2. A sublimation of 1 mm ie leaves the top node partly
    # in place, 27.3 mm ie removes one whole node, and 55 mm ie removes two. The
    # scheme adds one node at the bottom for each node that it takes from the top.
    # The water of the old bottom node stays in the column, at the index of that
    # node, and the new nodes are dry. The water of the old bottom node is not
    # counted as sublimated.
    n = 60
    lwc = np.zeros(n)
    lwc[-1] = 1.0e-4
    column = synthetic_column(n=n, rho_profile=400.0, T_profile=273.15, dz=0.05, LWC=lwc)
    column.PLWC_mem = np.zeros(n)
    column.refrozen = np.zeros(n)
    column.Ts = np.array([273.15])
    column.totwatersublim = 0.0
    column.sublimSec = np.array([-sublimation_ie / S_PER_YEAR])
    outputs = sublim.sublim(column, 0)
    lwc_after, totwatersublim = outputs[8], outputs[10]
    expected = np.zeros(n)
    expected[n - nodes_added] = 1.0e-4
    np.testing.assert_allclose(lwc_after, expected, atol=1e-15)
    assert totwatersublim == pytest.approx(0.0, abs=1e-15)


@pytest.mark.scheme
def test_sublimated_water_is_the_water_of_the_sublimated_nodes(synthetic_column):
    # 27.3 mm ie is 25.0 kg m-2. It removes the top node (20 kg m-2 of firn and
    # 0.2 kg m-2 of water) and 4.8 kg m-2 of the second node, water first. So the
    # water of the top two nodes sublimates, and the water of the bottom node does not.
    n = 60
    lwc = np.zeros(n)
    lwc[:2] = 2.0e-4
    lwc[-1] = 1.0e-4
    column = synthetic_column(n=n, rho_profile=400.0, T_profile=273.15, dz=0.05, LWC=lwc)
    column.PLWC_mem = np.zeros(n)
    column.refrozen = np.zeros(n)
    column.Ts = np.array([273.15])
    column.totwatersublim = 0.0
    column.sublimSec = np.array([-0.0273 / S_PER_YEAR])
    outputs = sublim.sublim(column, 0)
    lwc_after, totwatersublim = outputs[8], outputs[10]
    expected = np.zeros(n)
    expected[n - 2] = 1.0e-4
    np.testing.assert_allclose(lwc_after, expected, atol=1e-15)
    assert totwatersublim == pytest.approx(4.0e-4, abs=1e-15)
