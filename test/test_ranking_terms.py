"""The candidate TERMS, against cases whose answer is arithmetic (ADR-0014, ADR-0017 Decision 4).

They are kept apart because they measure different things, and the tests say what each one can and
cannot see -- which is the whole reason for having more than one.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K, R_FACE, synth_disc            # noqa: E402

from trainit_perception.ranking import (GraspFit, PoseQualityGates,  # noqa: E402
                                        approach_clearance, box_overlap, make_face,
                                        plane_ransac, pose_quality, visible_fraction)

FACE = make_face('disc', {'expected_radius_m': R_FACE})
FULL_AT_600 = FACE.expected_area_px(0.600, K)


# --- box_overlap: boxes only, so it works with any detector at all ---------------------------

def test_a_box_alone_is_not_overlapped():
    assert box_overlap([(0, 0, 10, 10)]).tolist() == [0.0]
    assert box_overlap([]).tolist() == []


def test_overlap_is_a_fraction_of_THIS_box_not_of_the_union():
    """A small box inside a big one is fully involved; the big one is barely. IoU hides that."""
    big, small = (0, 0, 100, 100), (10, 10, 30, 30)
    ov = box_overlap([big, small])
    assert ov[1] == pytest.approx(1.0)              # all of the small box is under the big one
    assert ov[0] == pytest.approx(400 / 10000)      # only 4 % of the big one
    iou = 400 / (10000 + 400 - 400)
    assert abs(ov[1] - iou) > 0.9, 'IoU would have hidden exactly the case that matters'


def test_half_over_half_is_a_half_and_a_mere_touch_is_nothing():
    assert box_overlap([(0, 0, 10, 10), (5, 0, 15, 10)]).tolist() == pytest.approx([0.5, 0.5])
    assert box_overlap([(0, 0, 10, 10), (10, 0, 20, 10)]).tolist() == pytest.approx([0.0, 0.0])


def test_the_WORST_neighbour_is_the_one_reported():
    ov = box_overlap([(0, 0, 10, 10), (5, 0, 15, 10), (2.5, 0, 12.5, 10)])
    assert ov[0] == pytest.approx(0.75)             # the third covers more than the second


# --- visible_fraction: the mask against the area the FACE should show ------------------------

def test_visible_fraction_is_the_mask_against_the_disc_the_face_should_show():
    assert visible_fraction(FULL_AT_600, FULL_AT_600) == pytest.approx(1.0)
    assert visible_fraction(FULL_AT_600 / 2, FULL_AT_600) == pytest.approx(0.5)
    assert visible_fraction(0.0, FULL_AT_600) == 0.0
    assert visible_fraction(10 * FULL_AT_600, FULL_AT_600) == 1.0        # clipped, never above 1


def test_an_UNKNOWN_face_area_reads_1_AND_IS_WHY_THE_RULE_IS_REFUSED():
    """The term cannot be computed, so it answers 1.0 -- which is exactly why a rule that needs it
    must be refused at start-up rather than silently multiplied by one (ADR-0017 Decision 3).
    VERIFIED on the pre-ADR-0017 code: candidate_x_visible and candidate_2d3d both read 0,7000."""
    assert visible_fraction(123.0, None) == 1.0
    assert make_face('disc', {}).expected_area_px(0.6, K) is None


def test_a_tilted_face_READS_LOW_and_that_is_the_validated_behaviour():
    """A tilted disc covers fewer pixels and this term says so. Dividing it back out is wrong.

    Correcting by cos(tilt) is the textbook way to isolate occlusion from foreshortening, and
    measured against the twin's own labels it destroys the number: the fit's tilt in a bin runs to
    a median of 33 degrees and a p90 of 63, cos(63 deg) is 0,45, and the term saturated at 1,00 on
    two thirds of detections including objects the truth said were 60 % covered. Without the
    correction the same 224 detections correlate with the truth at +0,94.
    """
    for tilt in (0, 20, 40, 60):
        ellipse = FULL_AT_600 * math.cos(math.radians(tilt))
        got = visible_fraction(ellipse, FULL_AT_600, tilt)
        assert got == pytest.approx(math.cos(math.radians(tilt)), abs=0.02)
    # and the argument is unused, so passing it cannot change the answer
    assert visible_fraction(FULL_AT_600 / 2, FULL_AT_600, 0.0) == \
        visible_fraction(FULL_AT_600 / 2, FULL_AT_600, 55.0)


def test_it_scales_with_depth_because_a_farther_face_covers_fewer_pixels():
    for z in (0.5, 0.6, 0.8):
        r_px = 424.0 * R_FACE / z
        assert visible_fraction(math.pi * r_px * r_px,
                                FACE.expected_area_px(z, K)) == pytest.approx(1.0)


def test_THE_CASE_THE_TWO_MEASURES_DISAGREE_ON():
    """Two objects side by side: their BOXES overlap, their FACES are both entirely visible.

    This is why "reject anything whose box overlaps" is not the same rule as "reject anything that
    is covered", and why both numbers are published instead of one. Measured correlation between
    `1 - box_overlap` and `visible_fraction` over a real bin: only +0,54.
    """
    r_px = 424.0 * R_FACE / 0.6
    d = int(round(2 * r_px)) - 6                     # centres just under a diameter apart
    boxes = [(0, 0, 2 * r_px, 2 * r_px), (d, 0, d + 2 * r_px, 2 * r_px)]
    ov = box_overlap(boxes)
    assert ov[0] > 0.1, 'the boxes really do overlap'
    assert visible_fraction(FULL_AT_600, FULL_AT_600) == pytest.approx(1.0), 'nothing is hidden'


# --- clearance: a probability, and a NaN is not a zero --------------------------------------

def test_clearance_is_a_probability_and_saturates():
    assert approach_clearance(0.0, reach_mm=8.0) == 1.0
    assert approach_clearance(4.0, reach_mm=8.0) == pytest.approx(0.5)
    assert approach_clearance(8.0, reach_mm=8.0) == 0.0
    assert approach_clearance(80.0, reach_mm=8.0) == 0.0          # clipped, never negative
    assert approach_clearance(-3.0, reach_mm=8.0) == 1.0          # clipped the other way too


def test_CANNOT_TELL_is_0_and_not_1_and_an_unset_window_switches_the_term_OFF():
    assert approach_clearance(float('nan'), reach_mm=8.0) == 0.0
    assert approach_clearance(0.0, reach_mm=0.0) == 1.0
    with pytest.raises(TypeError):
        approach_clearance(0.0)              # the grip window is the TOOL's, never a default


# --- pose_quality: four gates, and exactly 0.000 is a REFUSAL --------------------------------

def test_pose_quality_is_1_on_a_perfect_fit_and_every_gate_can_pull_it_down():
    perfect = GraspFit((0, 0, 0.6), (0, 0, -1), (R_FACE, R_FACE),
                       dict(depth_coverage=1.0, inlier_ratio=1.0, rms_mm=0.0, radius_ratio=1.0))
    assert pose_quality(perfect) == pytest.approx(1.0)
    for key, value in (('depth_coverage', 0.5), ('inlier_ratio', 0.8),
                       ('rms_mm', 2.0), ('radius_ratio', 1.2)):
        q = dict(perfect.quality, **{key: value})
        worse = GraspFit(perfect.point, perfect.normal, perfect.half_extents_m, q)
        assert pose_quality(worse) < 1.0, f'{key} is not gating'


def test_EXACTLY_ZERO_IS_A_REFUSAL_NOT_A_LOW_SCORE():
    """MEASURED on the live cell (2026-10-01): one mask over the bin's wall edge fitted a 78 mm
    face 0,4 m outside the bin and scored exactly 0.000, while the POOREST fit on a real object
    scored 0.085. The gates separate the two with room to spare, so a detector must not offer the
    first as a grasp whatever a consumer asked for."""
    # the bin wall: a 78 mm "face" where a 41,9 mm one was configured -> radius_ratio 1.86
    wall = GraspFit((0.0, 0.0, 1.0), (0, 0, -1), (0.039, 0.039),
                    dict(depth_coverage=1.0, inlier_ratio=1.0, rms_mm=0.0,
                         radius_ratio=0.039 / R_FACE))
    assert pose_quality(wall) == 0.0, 'a refusal, and it has to be exactly zero'
    # a poor but real object: every gate bites a little and nothing clips flat
    poor = GraspFit((0.0, 0.0, 0.6), (0, 0, -1), (R_FACE, R_FACE),
                    dict(depth_coverage=0.55, inlier_ratio=0.75, rms_mm=2.6, radius_ratio=1.15))
    assert 0.0 < pose_quality(poor) < 0.3
    # and a fit that failed outright is 0 without looking at anything
    assert pose_quality(GraspFit((0, 0, 0), (0, 0, -1), ok=False, why='no depth')) == 0.0


def test_the_four_gates_are_SERIALISABLE_CONFIGURATION_not_function_defaults():
    """ADR-0017 Decision 4: a shipped detector whose refusal threshold is a Python default cannot
    be tuned in the Setup Assistant or reproduced from a bundle."""
    assert PoseQualityGates().as_params() == {
        'pose_quality_min_coverage': 0.4, 'pose_quality_min_inliers': 0.6,
        'pose_quality_max_rms_mm': 3.0, 'pose_quality_radius_tol': 0.25}
    loose = PoseQualityGates.from_params({'pose_quality_radius_tol': 2.0})
    assert loose.radius_tol == 2.0 and loose.min_coverage == 0.4      # the rest keep the measured
    wall = GraspFit((0, 0, 1.0), (0, 0, -1), (0.039, 0.039),
                    dict(depth_coverage=1.0, inlier_ratio=1.0, rms_mm=0.0,
                         radius_ratio=0.039 / R_FACE))
    assert pose_quality(wall) == 0.0
    assert pose_quality(wall, gates=loose) > 0.0, 'the gate is configuration, and it moved'
    # the four defaults round-trip through a flat params dict, which is what a bundle stores
    assert PoseQualityGates.from_params(PoseQualityGates().as_params()) == PoseQualityGates()


def test_a_ZERO_tolerance_is_a_STRICT_gate_and_not_a_DISABLED_one():
    """The value a reader reaches for to make a gate strict must not silently remove it."""
    g = PoseQualityGates(max_rms_mm=0.0, radius_tol=0.0)
    exact = GraspFit((0, 0, 0.6), (0, 0, -1), (R_FACE, R_FACE),
                     dict(depth_coverage=1.0, inlier_ratio=1.0, rms_mm=0.0, radius_ratio=1.0))
    off = GraspFit((0, 0, 0.6), (0, 0, -1), (R_FACE, R_FACE),
                   dict(depth_coverage=1.0, inlier_ratio=1.0, rms_mm=0.1, radius_ratio=1.0))
    assert pose_quality(exact, gates=g) == 1.0
    assert pose_quality(off, gates=g) == 0.0


def test_the_gate_FOLLOWS_THE_FACE_rather_than_assuming_a_radius():
    """`extent_keys` is the face's own declaration, so a rectangular face gates on its two."""
    q = dict(depth_coverage=1.0, inlier_ratio=1.0, rms_mm=0.0,
             extent_a_ratio=1.5, extent_b_ratio=1.0)
    f = GraspFit((0, 0, 0.6), (0, 0, -1), (0.03, 0.02), q)
    assert pose_quality(f) == pytest.approx(1.0), 'nothing named radius_ratio is there to gate on'
    assert pose_quality(f, extent_keys=('extent_a_ratio', 'extent_b_ratio')) == 0.0


def test_quality_falls_when_the_mask_leaks_onto_a_neighbour():
    """The failure that matters: a mask that swallowed a second object. RANSAC locks onto ONE
    surface and throws the other away, so the fitted extent stays right and the point is still on
    a real object -- the giveaway is `inlier_ratio`, and pose_quality must be reading it."""
    m1, d1, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    m2, d2, _, _ = synth_disc((0.035, 0.0, 0.625), (0, 0, -1), seed=1)
    mask = m1 | m2
    depth = np.where(np.isfinite(d1), d1, d2)
    good = plane_ransac(m1, d1, K, face=FACE)
    bad = plane_ransac(mask, depth, K, face=FACE)
    assert pose_quality(bad) < pose_quality(good)
    assert good.quality['inlier_ratio'] > 0.95
    assert bad.quality['inlier_ratio'] < 0.80        # half the mask is not on the fitted surface
    assert abs(bad.quality['radius_ratio'] - 1.0) < 0.2   # the surface it DID find is a real one
