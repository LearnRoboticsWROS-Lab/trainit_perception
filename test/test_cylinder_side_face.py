"""The CYLINDER-SIDE face: the one a bottle picked from the side needs, and the HALF it refuses.

The area it can answer honestly is the silhouette, a rectangle of `2r` by `h` projected -- and
that is NOT the same thing as a `rect` face, even though the two formulas coincide. These tests
measure the three differences, and the third is the reason this face ships with NO extent terms:

  1. the invariance: a cylinder's silhouette survives the roll about its own axis, the one
     rotation a bottle in a crate is guaranteed to have;
  2. the depth: a cylinder spans `r` in depth, so the formula is exact only as `r/z -> 0`, and
     against the depth the fit reports it errs a few per cent LOW -- the safe direction;
  3. WHAT A PLANE FIT CAN MEASURE ON IT: not the radius. A `rect` face configured with the
     honest numbers scores a perfectly good bottle EXACTLY 0.000, which is a refusal.

A half-done face that silently mis-measures is worse than none, so the transverse extent is not
reported under any name until a cylinder FIT exists to measure it.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K, synth_cylinder                  # noqa: E402

from trainit_perception.ranking import (CylinderSideFace, expected_area_px,  # noqa: E402
                                        face_schema, make_face, p90_extent, plane_ransac,
                                        pose_quality, visible_fraction)

R, H = 0.0325, 0.060               # a 65 mm bottle, gripped over a 60 mm band of its body
F = 424.0
CYL = {'expected_radius_m': R, 'expected_height_m': H}
INLIER_MM = 2.0                    # `plane_ransac`'s shipped tolerance


def cyl():
    return make_face('cylinder_side', CYL)


# --- what the face DECLARES -------------------------------------------------------------------

def test_the_face_declares_NO_EXTENT_TERMS_and_that_is_the_declaration():
    assert face_schema('cylinder_side') == {
        'name': 'cylinder_side',
        'params': ('expected_radius_m', 'expected_height_m'),
        'extent_term_keys': (),
        'enables_terms': ('visible_fraction',)}
    assert isinstance(cyl(), CylinderSideFace)
    assert cyl().extent_terms((0.02, 0.03)) == {}
    # whatever offsets it is handed: the plane it would measure is not the face
    rng = np.random.default_rng(1)
    assert cyl().half_extents(rng.random((500, 2)) * 0.05, p90_extent) == (0.0, 0.0)


def test_BOTH_sizes_are_needed_and_MILLIMETRES_are_refused():
    assert cyl().has_known_area is True
    for partial in ({'expected_radius_m': R}, {'expected_height_m': H}, {}):
        face = make_face('cylinder_side', partial)
        assert face.has_known_area is False
        assert face.expected_area_px(0.6, K) is None
    with pytest.raises(ValueError) as e:
        make_face('cylinder_side', {'expected_radius_m': 32.5, 'expected_height_m': 60.0})
    assert 'MILLIMETRES' in str(e.value)


# --- the area it CAN answer: the silhouette ----------------------------------------------------

def test_the_expected_area_is_the_SILHOUETTE_2r_BY_h_and_not_the_curved_surface():
    face = cyl()
    for z in (0.5, 0.6, 0.8):
        assert face.expected_area_px(z, K) == pytest.approx((2 * R * F / z) * (H * F / z))
    assert face.expected_area_px(1.2, K) == pytest.approx(face.expected_area_px(0.6, K) / 4.0)
    # `2r`, not the unrolled `pi*r`: the camera sees the OUTLINE, which is as wide as the
    # diameter, so the expected area is 2/pi of the curved surface a developed cylinder has
    unrolled = (math.pi * R * F / 0.6) * (H * F / 0.6)
    assert face.expected_area_px(0.6, K) == pytest.approx(unrolled * 2.0 / math.pi)


def test_IT_MATCHES_THE_RASTER_AGAINST_THE_DEPTH_THE_FIT_REPORTS_and_never_saturates():
    """MEASURED on an analytic ray/cylinder raster: a FULLY VISIBLE silhouette reads 0,958 to
    0,975 of the formula across `r/z` from 0,008 to 0,054 -- a few per cent LOW, which is the
    safe direction. Fed the cylinder's AXIS depth instead it would read up to 1,07 and CLIP,
    which is the saturation ADR-0017 Decision 6 exists to catch."""
    for (r, h, z) in ((0.0325, 0.060, 0.600), (0.0110, 0.040, 0.600),
                      (0.0325, 0.060, 0.800), (0.0050, 0.030, 0.600)):
        face = make_face('cylinder_side', {'expected_radius_m': r, 'expected_height_m': h})
        mask, depth, _c, _a = synth_cylinder((0.0, 0.0, z), (1, 0, 0), r, h)
        fit = plane_ransac(mask, depth, K, face=face)
        got = visible_fraction(int(mask.sum()), expected_area_px(face, fit, K))
        assert 0.93 <= got < 1.0, f'r={r} h={h} z={z} read {got}'
        # and fed the AXIS depth it would clip -- the failure this convention avoids
        assert int(mask.sum()) / face.expected_area_px(z, K) > got


def test_THE_INVARIANCE_THAT_MAKES_THIS_FACE_WORTH_HAVING_the_roll_about_its_own_axis():
    """A bottle in a crate has one rotation you cannot control: the roll about its own axis. The
    silhouette is `2r` wide seen from ANY direction perpendicular to that axis, so the area term
    does not move -- which a rectangular face could not claim."""
    face = cyl()
    areas = []
    for axis in ((1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -2, 0)):
        mask, depth, _c, _a = synth_cylinder((0.0, 0.0, 0.600), axis, R, H)
        fit = plane_ransac(mask, depth, K, face=face)
        areas.append(visible_fraction(int(mask.sum()), expected_area_px(face, fit, K)))
    assert max(areas) - min(areas) < 0.03, f'the silhouette moved: {areas}'


def test_half_the_silhouette_covered_reads_a_HALF():
    face = cyl()
    mask, depth, _c, _a = synth_cylinder((0.0, 0.0, 0.600), (1, 0, 0), R, H)
    fit = plane_ransac(mask, depth, K, face=face)
    full = visible_fraction(int(mask.sum()), expected_area_px(face, fit, K))
    half = visible_fraction(0.5 * int(mask.sum()), expected_area_px(face, fit, K))
    assert half == pytest.approx(0.5 * full, rel=1e-6)


# --- the half it REFUSES, and the measurement that settles it ---------------------------------

def test_A_PLANE_FIT_MEASURES_THE_RANSAC_BAND_AND_NOT_THE_CYLINDER():
    """THE REASON `extent_term_keys` IS EMPTY.

    RANSAC keeps the band where the curved surface stays inside its tolerance, whose half-width
    is about `sqrt(2 * tol * r)` -- 11,40 mm for a 32,5 mm bottle at the shipped 2 mm. MEASURED:
    the fit recovers (26,86, 12,29) mm of half-extent on a 60 mm band of a 32,5 mm cylinder, so
    the transverse "radius" reads 0,38 of the truth. That number is a property of the TOLERANCE,
    not of the face, and publishing it as the face's extent is the silent mis-measurement.
    """
    band_mm = math.sqrt(2 * INLIER_MM * 1e-3 * R) * 1e3
    assert band_mm == pytest.approx(11.40, abs=0.05)
    mask, depth, _c, _a = synth_cylinder((0.0, 0.0, 0.600), (1, 0, 0), R, H)
    # a face that DID report its two in-plane half-extents, to see what it would have said
    rect_probe = make_face('rect', {'expected_half_a_m': R, 'expected_half_b_m': 0.5 * H})
    fit = plane_ransac(mask, depth, K, face=rect_probe)
    wide, narrow = sorted((x * 1e3 for x in fit.half_extents_m), reverse=True)
    assert narrow == pytest.approx(12.3, abs=1.5), 'the band, not the radius'
    assert narrow / (R * 1e3) == pytest.approx(0.38, abs=0.06)
    assert wide == pytest.approx(26.9, abs=1.5), 'the axial half-extent, which IS measurable'
    # ...and the cylinder face claims neither of them
    assert plane_ransac(mask, depth, K, face=cyl()).half_extents_m == (0.0, 0.0)


def test_A_RECT_FACE_ON_A_CYLINDER_REFUSES_A_PERFECTLY_GOOD_BOTTLE():
    """So `cylinder_side` is not `rect` with different numbers: configured with the HONEST
    dimensions (32,5 x 30 mm half-extents), a rect face scores this bottle EXACTLY 0.000 -- which
    `Ranker.keep` drops outright, whatever a consumer asked for."""
    mask, depth, _c, _a = synth_cylinder((0.0, 0.0, 0.600), (1, 0, 0), R, H)
    rect_face = make_face('rect', {'expected_half_a_m': R, 'expected_half_b_m': 0.5 * H})
    rect_fit = plane_ransac(mask, depth, K, face=rect_face)
    assert pose_quality(rect_fit, extent_keys=rect_face.extent_term_keys) == 0.0
    # the cylinder face gates on no extent at all, so the candidate survives to be ranked
    face = cyl()
    fit = plane_ransac(mask, depth, K, face=face)
    pq = pose_quality(fit, extent_keys=face.extent_term_keys)
    assert pq > 0.0


def test_THE_FIT_STILL_SAYS_IT_IS_FITTING_A_PLANE_TO_SOMETHING_THAT_IS_NOT_ONE():
    """Nothing is hidden by dropping the extent gate: `inlier_ratio` reads 0,45 and
    `pose_quality` 0,18 on that same bottle, against 0,95+ and ~1,0 on a flat face. The fit is
    telling you, correctly, what a cylinder does to a plane -- which is why the missing half is a
    `cylinder_ransac` METHOD and not a different face."""
    face = cyl()
    mask, depth, _c, _a = synth_cylinder((0.0, 0.0, 0.600), (1, 0, 0), R, H)
    fit = plane_ransac(mask, depth, K, face=face)
    assert fit.ok
    assert fit.quality['inlier_ratio'] == pytest.approx(0.45, abs=0.06)
    assert pose_quality(fit, extent_keys=face.extent_term_keys) == pytest.approx(0.18, abs=0.05)
    assert 'extent_a_ratio' not in fit.quality and 'radius_ratio' not in fit.quality
    # a thin cylinder is nearly a plane over its band, and the fit says THAT too
    thin = make_face('cylinder_side', {'expected_radius_m': 0.011, 'expected_height_m': 0.040})
    m2, d2, _c2, _a2 = synth_cylinder((0.0, 0.0, 0.600), (1, 0, 0), 0.011, 0.040)
    f2 = plane_ransac(m2, d2, K, face=thin)
    assert f2.quality['inlier_ratio'] > fit.quality['inlier_ratio']
