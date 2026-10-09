'''
Short model runs with config combinations that have crashed before.
'''

import pytest


@pytest.mark.conservation
def test_melt_without_sublimation_completes(short_config):
    # SUBLIM false skips the sublimation block, which is the only code that
    # set subLWCvol to a scalar. See UWGlaciology/CommunityFirnModel#19.
    run = short_config(years=2, MELT=True, SUBLIM=False, melt_strict=True)
    run.run()
    assert run.results_path.exists()
