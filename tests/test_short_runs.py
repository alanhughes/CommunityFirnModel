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


@pytest.mark.conservation
def test_darcy_completes(short_config):
    # Before model year 1980 the darcy branch falls back to bucket. bucket
    # returns 13 values (the last is dh_melt) and the fallback unpacked 12.
    run = short_config(years=2, liquid='darcy', melt_strict=True)
    run.run()
    assert run.results_path.exists()


@pytest.mark.conservation
def test_darcy_with_early_output_completes(short_config):
    # The first steps are dry. With output from the first step, update_dH
    # read self.dh_melt before any branch had set it.
    # See UWGlaciology/CommunityFirnModel#21.
    run = short_config(years=2, liquid='darcy', melt_strict=True, TWriteStart=0.0)
    run.run()
    assert run.results_path.exists()


@pytest.mark.conservation
@pytest.mark.parametrize('liquid', ['prefsnowpack', 'resingledomain'])
@pytest.mark.parametrize('TWriteStart', [None, 0.0], ids=['output_1980', 'output_first_step'])
def test_snowpack_scheme_completes(short_config, liquid, TWriteStart):
    # __init__ set refreeze, runoff, meltvol and rainvol to 1-element arrays.
    # The dry cases of these branches do not overwrite them, so NumPy 2.5
    # raised at refreezing2check[iii] = self.refreeze. Both schemes also used
    # the per-step array self.dt as one step length. The run starts at model
    # year 1434 and reaches the scheme itself (not the bucket fallback) after
    # 1980. With TWriteStart 0, update_dH reads dh_melt from the first step.
    overrides = {} if TWriteStart is None else {'TWriteStart': TWriteStart}
    run = short_config(years=2, liquid=liquid, **overrides)
    run.run()
    assert run.results_path.exists()
