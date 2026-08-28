"""Unit tests on a SYNTHETIC frame built from the cell's VERIFIED intrinsics.

fx = fy = 424, cx = 424, cy = 240 were measured on the live camera_info; a 2 cm cube
seen from the new camera height (optical distance 0.473 m) is ~18 px. The tests check
that the detector recovers those metres back from the pixels — the whole point of the
back-projection.
"""
import numpy as np
import pytest

from trainit_perception.detectors import (ColorMaskDetector, depth_at, depth_to_metres,
                                          make_detector)

W, H = 848, 480
K = np.array([[424.0, 0, 424.0], [0, 424.0, 240.0], [0, 0, 1.0]])
Z = 0.473
CUBE_PX = 18


def frame(cube_at=(500, 300), cube_px=CUBE_PX, colour=(140, 13, 15), z=Z, extra=None):
    """Grey table, one dark-red square, flat depth plane. Returns (rgb, depth_m)."""
    rgb = np.full((H, W, 3), (160, 163, 170), np.uint8)
    depth = np.full((H, W), z, np.float32)
    u, v = cube_at
    rgb[v:v + cube_px, u:u + cube_px] = colour
    for (eu, ev, epx, ecol) in (extra or []):
        rgb[ev:ev + epx, eu:eu + epx] = ecol
    return rgb, depth


def test_finds_the_cube_and_recovers_its_size():
    rgb, depth = frame()
    dets = ColorMaskDetector({'class_id': 'cube'}).detect(rgb, depth, K)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 'cube'
    # centroid of an 18 px square starting at (500, 300): 500 + 8.5, 300 + 8.5
    assert d.pixel == pytest.approx((508.5, 308.5), abs=0.6)
    # back-projection with the real intrinsics
    exp_x = (508.5 - 424.0) * Z / 424.0
    exp_y = (308.5 - 240.0) * Z / 424.0
    assert d.position == pytest.approx((exp_x, exp_y, Z), abs=1e-3)
    # 18 px at 0.473 m through f = 424 is 0.0201 m: the detector gets the 2 cm cube back
    assert d.size[0] == pytest.approx(0.020, abs=0.002)
    assert d.size[1] == pytest.approx(0.020, abs=0.002)
    assert d.score > 0.9                     # a filled square is solid


def test_nothing_red_means_nothing_found():
    rgb, depth = frame(colour=(160, 163, 170))          # the "cube" is table-coloured
    assert ColorMaskDetector().detect(rgb, depth, K) == []


def test_hue_wrap_catches_both_sides_of_red():
    """Red straddles hue 0. A range written lo > hi must catch a pinkish red (H≈175)
    AND an orange-red (H≈5)."""
    det = ColorMaskDetector({'h': [170, 10]})
    for colour in ((150, 10, 30), (150, 30, 10)):        # H ≈ 171 and H ≈ 9
        rgb, depth = frame(colour=colour)
        assert len(det.detect(rgb, depth, K)) == 1, colour


def test_largest_blob_wins_with_max_objects_one():
    rgb, depth = frame(extra=[(200, 100, 40, (140, 13, 15))])   # a bigger red square too
    dets = ColorMaskDetector({'max_objects': 1}).detect(rgb, depth, K)
    assert len(dets) == 1 and dets[0].area_px > CUBE_PX * CUBE_PX
    both = ColorMaskDetector({'max_objects': 2}).detect(rgb, depth, K)
    assert len(both) == 2 and both[0].area_px >= both[1].area_px


def test_min_area_rejects_speckle():
    rgb, depth = frame(cube_px=4)                        # 16 px² of red
    assert ColorMaskDetector({'min_area_px': 60}).detect(rgb, depth, K) == []


def test_depth_hole_at_centroid_is_rescued_by_the_window():
    rgb, depth = frame()
    depth[300:318, 500:518] = 0.0                       # the whole cube is a hole...
    depth[309, 509] = 0.0
    depth[306:312, 506:512] = Z                          # ...except a small valid patch
    d = ColorMaskDetector().detect(rgb, depth, K)
    assert len(d) == 1 and d[0].position[2] == pytest.approx(Z)


def test_16uc1_millimetres_give_the_same_metres():
    rgb, depth32 = frame()
    depth16 = (depth32 * 1000.0).astype(np.uint16)
    a = ColorMaskDetector().detect(rgb, depth_to_metres(depth32, '32FC1'), K)[0]
    b = ColorMaskDetector().detect(rgb, depth_to_metres(depth16, '16UC1'), K)[0]
    assert a.position == pytest.approx(b.position, abs=1e-3)


def test_depth_at_ignores_invalid_values():
    d = np.full((10, 10), np.nan, np.float32)
    d[4:7, 4:7] = 0.5
    d[5, 5] = np.inf
    assert depth_at(d, 5, 5, window=5) == pytest.approx(0.5)
    assert np.isnan(depth_at(np.zeros((10, 10), np.float32), 5, 5))


def test_registry_and_defaults_are_the_profile():
    det = make_detector('color_mask', {'class_id': 'cube'})
    assert det.params['class_id'] == 'cube' and det.params['h'] == [170, 10]
    with pytest.raises(KeyError):
        make_detector('no_such_method')


def test_message_builder():
    from trainit_perception.msgs import to_detection3d_array
    from builtin_interfaces.msg import Time
    rgb, depth = frame()
    dets = ColorMaskDetector({'class_id': 'cube'}).detect(rgb, depth, K)
    msg = to_detection3d_array(dets, 'camera_color_optical_frame', Time(sec=1, nanosec=2))
    assert msg.header.frame_id == 'camera_color_optical_frame'
    assert len(msg.detections) == 1
    r = msg.detections[0].results[0]
    assert r.hypothesis.class_id == 'cube'
    assert r.pose.pose.position.z == pytest.approx(Z)
    assert r.pose.pose.orientation.w == 1.0
    assert msg.detections[0].bbox.size.x == pytest.approx(0.020, abs=0.002)
