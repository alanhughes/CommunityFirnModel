'''
The per-node arrays after a melt that removes nodes, and the warning for the
schemes that ignore keep_firnthickness in their melt removal.
'''

import numpy as np
import pytest

import melt
from cfm_testing import example_config
from config_validation import validate_config
from constants import RHO_I

NODES = 60
DENSITY = 400.0
THICKNESS = 0.1
NODE_MASS = DENSITY * THICKNESS  # [kg m-2]
STEP = 600.0  # [s] darcyscheme starts with 60 s sub-steps, so a sub-step removes whole nodes.


def _melted_column(synthetic_column, scheme, melted_nodes):
    '''Run one scheme on a column at the melting point. Give each node a different label.'''
    column = synthetic_column(n=NODES, rho_profile=DENSITY, T_profile=273.15, dz=THICKNESS,
                              melt=melted_nodes * NODE_MASS / RHO_I, dt=STEP)
    column.bdot_mean = 0.1 + np.arange(NODES, dtype=float)
    column.Dcon = 1000.0 + np.arange(NODES, dtype=float)
    labels = column.bdot_mean.copy(), column.Dcon.copy()
    getattr(melt, scheme)(column, 0)
    return column, labels


def _shifted(values, removed):
    '''The array after the top nodes are removed. Copies of the bottom value fill the base.'''
    return np.concatenate((values[removed:], np.full(removed, values[-1])))


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_melt_shifts_bdot_mean_and_dcon(synthetic_column, scheme):
    # 11 whole nodes melt, and the 12th melts in part, so 11 nodes leave the column.
    column, (bdot_mean, dcon) = _melted_column(synthetic_column, scheme, melted_nodes=11.5)
    np.testing.assert_array_equal(column.bdot_mean, _shifted(bdot_mean, 11))
    np.testing.assert_array_equal(column.Dcon, _shifted(dcon, 11))


@pytest.mark.scheme
@pytest.mark.parametrize('scheme', ['bucket', 'darcyscheme'])
def test_melt_within_the_top_node_keeps_bdot_mean_and_dcon(synthetic_column, scheme):
    column, (bdot_mean, dcon) = _melted_column(synthetic_column, scheme, melted_nodes=0.5)
    np.testing.assert_array_equal(column.bdot_mean, bdot_mean)
    np.testing.assert_array_equal(column.Dcon, dcon)


def _warnings_for(capsys, **overrides):
    config = example_config()
    config.update(overrides)
    validate_config(config)
    return [line for line in capsys.readouterr().out.splitlines()
            if 'WARNING' in line and 'keep_firnthickness' in line]


@pytest.mark.unit
@pytest.mark.parametrize('liquid', ['darcy', 'resingledomain', 'prefsnowpack'])
def test_warning_for_a_scheme_that_ignores_keep_firnthickness(capsys, liquid):
    warnings = _warnings_for(capsys, liquid=liquid, keep_firnthickness=True)
    assert len(warnings) == 1
    assert f"'{liquid}'" in warnings[0]


@pytest.mark.unit
@pytest.mark.parametrize('overrides', [
    {'liquid': 'bucket', 'keep_firnthickness': True},
    {'liquid': 'darcy', 'keep_firnthickness': False},
    {'liquid': 'resingledomain', 'keep_firnthickness': False},
    {'liquid': 'prefsnowpack', 'keep_firnthickness': False},
])
def test_no_warning_if_the_scheme_follows_the_key_or_the_key_is_false(capsys, overrides):
    assert _warnings_for(capsys, **overrides) == []


@pytest.mark.unit
def test_no_warning_if_the_key_is_absent(capsys):
    config = example_config()
    config['liquid'] = 'darcy'
    del config['keep_firnthickness']
    validate_config(config)
    assert 'keep_firnthickness' not in capsys.readouterr().out
