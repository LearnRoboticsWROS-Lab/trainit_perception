"""Regression on a REAL frame from the Isaac cell (captured 2026-08-28, proof 1 of D-014).

The synthetic tests prove the geometry; this one proves the thresholds survive a real
render -- the theory default s=[120,255] found NOTHING here, because under the dome
light the red cube renders at S = 62-72. Expected values are what proof 1 measured and
verified against the cube's known pose (error 1.0 mm XY, 0.5 mm Z in base_link).
"""
import os

import numpy as np
import pytest

from trainit_perception.detectors import ColorMaskDetector, depth_to_metres

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, 'fixtures', 'isaac_cell_red_cube.npz')


@pytest.fixture(scope='module')
def frame():
    f = np.load(FIX, allow_pickle=False)
    return f['rgb'], depth_to_metres(f['depth'], str(f['depth_encoding'])), f['K']


def test_default_params_find_the_cube_on_a_real_frame(frame):
    rgb, depth_m, K = frame
    dets = ColorMaskDetector({'class_id': 'cube'}).detect(rgb, depth_m, K)
    assert len(dets) == 1, 'exactly one red object in the cell'
    d = dets[0]
    assert d.pixel == pytest.approx((446.4, 251.0), abs=1.5)
    assert d.position == pytest.approx((0.0252, 0.0123, 0.4754), abs=0.002)
    assert d.size[0] == pytest.approx(0.020, abs=0.002)
    assert d.score > 0.9


def test_the_old_saturation_default_would_have_missed_it(frame):
    """Documents WHY the default moved: keep this failing mode visible."""
    rgb, depth_m, K = frame
    assert ColorMaskDetector({'s': [120, 255]}).detect(rgb, depth_m, K) == []
