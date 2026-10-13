"""The publisher: BEST FIRST, the fitted normal as the orientation, and the terms carried
(ADR-0017 Decision 5). Before this, `to_detection3d_array` had no ordering, hard-coded an identity
orientation and DROPPED the `Detection.extra` dict it already carried.
"""
import math
import os
import sys

import numpy as np
import pytest
from builtin_interfaces.msg import Time

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import R_FACE, synth_disc                # noqa: E402

from trainit_perception.detectors import ColorMaskDetector, Detection  # noqa: E402
from trainit_perception.msgs import (quat_from_z, to_detection3d_array,  # noqa: E402
                                     to_quality_json, to_quality_payload)
from trainit_perception.ranking import (PROVIDES_FIT3D, PROVIDES_MASK_AREA,  # noqa: E402
                                        PROVIDES_SCORE, Ranker, make_face, plane_ransac)

W, H = 848, 480
K = np.array([[424.0, 0, 424.0], [0, 424.0, 240.0], [0, 0, 1.0]])
Z = 0.473
STAMP = Time(sec=7, nanosec=11)


def two_blobs():
    """Two red blobs at the same depth: the LARGER one is the less solid one.

    A 20x20 filled square (area 400, solidity 1,00) and a 22x22 square with an 8x8 hole punched
    out of it (area 420, solidity 0,87). The detector's own order is largest blob first, so it
    returns the holed one first -- and any rule with `score` in it puts the solid one first.
    """
    rgb = np.full((H, W, 3), (160, 163, 170), np.uint8)
    depth = np.full((H, W), Z, np.float32)
    rgb[100:120, 100:120] = (140, 13, 15)                     # A: 20x20 filled  -> 400 px
    rgb[300:322, 300:322] = (140, 13, 15)                     # B: 22x22 ...
    rgb[305:313, 305:313] = (160, 163, 170)                   # less an 8x8 hole -> 420 px
    return rgb, depth


def test_the_detector_returns_largest_FIRST_and_the_RANKING_disagrees():
    rgb, depth = two_blobs()
    params = {'max_objects': 2, 'morph_kernel': 0, 'candidate_rule': 'detector_order'}
    raw = ColorMaskDetector(params).detect(rgb, depth, K)
    assert [d.area_px for d in raw] == [420, 400], 'the detector sorts by blob area'
    # detector_order publishes exactly that, and says so
    assert [d.area_px for d in ColorMaskDetector(params).detect_ranked(rgb, depth, K)] == \
        [420, 400]
    # the ranking puts the SOLID one first, and that is the whole point of a rank
    ranked = ColorMaskDetector(dict(params, candidate_rule='score')).detect_ranked(rgb, depth, K)
    assert [d.area_px for d in ranked] == [400, 420]
    assert ranked[0].score > ranked[1].score


def test_visible_fraction_ranks_a_colour_mask_and_the_terms_are_PUBLISHED():
    rgb, depth = two_blobs()
    r = 0.0178                               # a face whose disc is ~800 px at this depth
    det = ColorMaskDetector({'max_objects': 2, 'morph_kernel': 0, 'expected_radius_m': r,
                             'candidate_rule': 'score_x_visible'})
    dets = det.detect_ranked(rgb, depth, K)
    expected = math.pi * (424.0 * r / Z) ** 2
    assert [d.area_px for d in dets] == [400, 420]
    a, b = dets[0].extra, dets[1].extra
    assert a['visible_fraction'] == pytest.approx(400 / expected, rel=1e-6)
    assert b['visible_fraction'] == pytest.approx(420 / expected, rel=1e-6)
    assert a['score_x_visible'] > b['score_x_visible']
    # every rule this detector can compute goes out, so a tuner can see the alternatives
    assert {'score', 'box_overlap', 'visible_fraction', 'score_x_visible',
            'score_x_nonoverlap'} <= set(a)
    assert 'detector_order' not in a, 'a rule with no factors is a constant, not a term'
    assert 'pose_quality' not in a, 'a colour mask runs no 3D fit and must not pretend otherwise'
    # the terms are NON-INCREASING under the configured rule, as published
    vals = [d.extra['score_x_visible'] for d in dets]
    assert vals == sorted(vals, reverse=True)


def l_shape_and_a_square():
    """An L-shaped blob and a solid square sitting in its concave corner, 9 px clear of it.

    The square's BOX is entirely inside the L's box while the two are not connected -- which is
    `box_overlap`'s asymmetry in a picture: all of the square is involved with the L (1,00) and
    only 24 % of the L with the square.
    """
    rgb = np.full((H, W, 3), (160, 163, 170), np.uint8)
    depth = np.full((H, W), Z, np.float32)
    rgb[100:141, 100:111] = (140, 13, 15)                     # the L's vertical bar
    rgb[130:141, 100:141] = (140, 13, 15)                     # the L's horizontal bar
    rgb[100:125, 120:136] = (140, 13, 15)                     # the separate square
    return rgb, depth


def test_box_overlap_is_published_for_a_detector_with_no_depth_at_all():
    """Boxes only: no mask area, no face, no depth, and a rank all the same (ADR-0014)."""
    rgb, depth = l_shape_and_a_square()
    params = {'max_objects': 3, 'morph_kernel': 0}
    dets = ColorMaskDetector(dict(params, candidate_rule='score_x_nonoverlap')).detect_ranked(
        rgb, depth, K)
    assert [d.area_px for d in dets] == [781, 400]
    by_area = {d.area_px: d.extra for d in dets}
    assert by_area[400]['box_overlap'] == pytest.approx(1.0), 'all of the square is under the L'
    assert by_area[781]['box_overlap'] == pytest.approx(400 / 1681, abs=0.01)
    vals = [d.extra['score_x_nonoverlap'] for d in dets]
    assert vals == sorted(vals, reverse=True)
    # and the two rules DISAGREE: by solidity the filled square wins, by overlap it is last
    by_score = ColorMaskDetector(dict(params, candidate_rule='score')).detect_ranked(
        rgb, depth, K)
    assert [d.area_px for d in by_score] == [400, 781]


def test_the_array_goes_out_IN_THE_ORDER_IT_IS_GIVEN_and_nothing_re_sorts_it():
    """The permutation is applied once, by the ranker. A serialiser that re-ranked would have to
    know what `visible_fraction` means, which is the boundary the contract exists to keep."""
    dets = [Detection('cube', 0.1, (0, 0, 1.0)), Detection('cube', 0.9, (0, 0, 1.0))]
    msg = to_detection3d_array(dets, 'camera_color_optical_frame', STAMP)
    assert [d.results[0].hypothesis.score for d in msg.detections] == [0.1, 0.9]
    assert [d.id for d in msg.detections] == ['cube_0', 'cube_1']


def test_the_FITTED_NORMAL_becomes_the_detection_ORIENTATION():
    """A tool asking to align with the detected object gets its axis along the surface normal."""
    tilt = math.radians(30)
    n = (math.sin(tilt), 0.0, -math.cos(tilt))
    msg = to_detection3d_array([Detection('cap', 0.9, (0.01, 0.0, 0.6), normal=n)], 'optical',
                               STAMP)
    q = msg.detections[0].results[0].pose.pose.orientation
    # rotate +Z by the quaternion and check it lands on the normal
    x, y, z, w = q.x, q.y, q.z, q.w
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    assert (zx, zy, zz) == pytest.approx(n, abs=1e-6)
    assert msg.detections[0].bbox.center.orientation == q


def test_NO_normal_keeps_the_identity_quaternion():
    """What a centroid-only method honestly has to say. The pre-ADR-0017 behaviour, preserved."""
    msg = to_detection3d_array([Detection('cube', 0.9, (0, 0, 0.5))], 'optical', STAMP)
    q = msg.detections[0].results[0].pose.pose.orientation
    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)


def test_quat_from_z_on_the_degenerate_directions():
    for n, expect in (((0, 0, 1), (0.0, 0.0, 0.0, 1.0)), ((0, 0, -1), (1.0, 0.0, 0.0, 0.0)),
                      ((0, 0, 0), (0.0, 0.0, 0.0, 1.0))):
        q = quat_from_z(n)
        assert (q.x, q.y, q.z, q.w) == pytest.approx(expect)
    q = quat_from_z((0, 0, -2))                   # not normalised: still the 180 deg about X
    assert (q.x, q.w) == pytest.approx((1.0, 0.0))


def test_the_SIDECAR_carries_the_terms_in_the_SAME_order_paired_by_STAMP():
    rgb, depth = two_blobs()
    det = ColorMaskDetector({'max_objects': 2, 'morph_kernel': 0, 'candidate_rule': 'score'})
    dets = det.detect_ranked(rgb, depth, K)
    arr = to_detection3d_array(dets, 'optical', STAMP)
    payload = to_quality_payload(dets, 'optical', STAMP)
    assert payload['stamp'] == {'sec': 7, 'nanosec': 11}
    assert payload['frame_id'] == 'optical' == arr.header.frame_id
    assert payload['stamp']['sec'] == arr.header.stamp.sec
    assert len(payload['detections']) == len(arr.detections) == 2
    # detections[i] there IS detections[i] here: one permutation was applied to both
    for row, d3 in zip(payload['detections'], arr.detections):
        assert row['class_id'] == d3.results[0].hypothesis.class_id
        assert row['score'] == pytest.approx(d3.results[0].hypothesis.score)
    assert payload['detections'][0]['score'] >= payload['detections'][1]['score']
    assert 'score' in to_quality_json(dets, 'optical', STAMP)


def test_a_NON_FINITE_term_goes_out_as_null_and_not_as_invalid_JSON():
    """`obstruction_mm` is NaN for "the depth could not answer" -- a different answer from 0
    ("clear") -- and the literal NaN `json.dumps` emits is not valid JSON."""
    import json
    d = Detection('cap', 0.9, (0, 0, 0.6))
    d.extra.update({'obstruction_mm': float('nan'), 'clearance': 0.0})
    text = to_quality_json([d], 'optical', STAMP)
    assert 'NaN' not in text
    row = json.loads(text)['detections'][0]
    assert row['obstruction_mm'] is None and row['clearance'] == 0.0


# --- the refusal, through the ranker a fitting detector would use ---------------------------

class _FittingDetector(ColorMaskDetector):
    """A stand-in for a detector that runs a 3D fit: the colour mask with a plane fit bolted on,
    so the refusal and the depth terms can be exercised without a learned model."""

    method = 'test_color_mask_3d'
    PROVIDES = frozenset({PROVIDES_SCORE, PROVIDES_MASK_AREA, PROVIDES_FIT3D})

    @classmethod
    def defaults(cls):
        return {**ColorMaskDetector.defaults(), **cls.ranking_defaults()}


def test_a_pose_quality_of_EXACTLY_ZERO_IS_DROPPED_whatever_the_consumer_asked_for():
    """The bin-wall false positive: a mask that fitted a 78 mm face where 41,9 mm was configured.
    MEASURED to score exactly 0.000 while the poorest real fit scored 0.085."""
    face = make_face('disc', {'expected_radius_m': R_FACE})
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1), radius=R_FACE)
    wall_mask, wall_depth, _, _ = synth_disc((0.10, 0.0, 1.000), (0, 0, -1), radius=0.039)
    good = plane_ransac(mask, depth, K, face=face)
    wall = plane_ransac(wall_mask, wall_depth, K, face=face)
    assert good.ok and wall.ok
    dets = [Detection('cap', 0.9, wall.point, normal=wall.normal, area_px=int(wall_mask.sum()),
                      fit=wall),
            Detection('cap', 0.5, good.point, normal=good.normal, area_px=int(mask.sum()),
                      fit=good)]
    ranker = Ranker.from_params({'candidate_rule': 'candidate_x_visible',
                                 'expected_radius_m': R_FACE},
                                provides=_FittingDetector.PROVIDES)
    out = ranker.apply(dets, K)
    assert len(out) == 1, 'the bin wall is refused, not merely ranked last'
    assert out[0].score == 0.5
    assert out[0].extra['pose_quality'] > 0.0
    assert out[0].extra['radius_ratio'] == pytest.approx(1.0, abs=0.1)
    assert out[0].extra['half_extent_a_mm'] == pytest.approx(R_FACE * 1e3, abs=2.0)


def test_min_pose_quality_is_a_FURTHER_threshold_and_never_a_replacement():
    face = make_face('disc', {'expected_radius_m': R_FACE})
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1), noise_mm=1.5, seed=4)
    f = plane_ransac(mask, depth, K, face=face)
    det = Detection('cap', 0.9, f.point, normal=f.normal, area_px=int(mask.sum()), fit=f)
    base = {'candidate_rule': 'candidate_2d3d', 'expected_radius_m': R_FACE}
    pq = Ranker.from_params(base, provides=_FittingDetector.PROVIDES).terms([det], K)[0]
    assert 0.0 < pq['pose_quality'] < 1.0
    assert len(Ranker.from_params(base, provides=_FittingDetector.PROVIDES).apply([det], K)) == 1
    strict = dict(base, min_pose_quality=pq['pose_quality'] + 0.01)
    assert Ranker.from_params(strict, provides=_FittingDetector.PROVIDES).apply([det], K) == []


def test_the_clearance_terms_ride_on_the_sidecar_when_a_TOOL_is_configured():
    face = make_face('disc', {'expected_radius_m': R_FACE})
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    m2, d2, _, _ = synth_disc((0.012, 0.0, 0.594), (0, 0, -1), seed=2)
    scene = np.where(np.isfinite(d2), d2, depth)
    f = plane_ransac(mask, depth, K, face=face)
    det = Detection('cap', 0.9, f.point, normal=f.normal, area_px=int(mask.sum()), fit=f)
    ranker = Ranker.from_params({'candidate_rule': 'candidate_x_clearance',
                                 'expected_radius_m': R_FACE, 'tool_radius_m': 0.010,
                                 'tool_reach_mm': 8.0}, provides=_FittingDetector.PROVIDES)
    out = ranker.apply([det], K, depth_m=scene)
    assert out[0].extra['obstruction_mm'] == pytest.approx(6.0, abs=0.8)
    assert out[0].extra['clearance'] < 0.4
    assert 'candidate_x_clearance' in out[0].extra
