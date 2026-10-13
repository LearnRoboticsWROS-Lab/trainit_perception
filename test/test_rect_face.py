"""The RECTANGLE face: the second entry in the registry, and the proof the plug is a plug.

ADR-0018 measured this face as the test of the extension point: written against the ABC and
registered, it needs ZERO changes to `fit3d`, `terms`, `ranker` or `rules`. These tests check both
halves of that claim -- that the arithmetic is right, and that nothing outside `faces.py` had to
learn about it.

Every number asserted here is read off an analytic raster (`synth_rect`), so an error is in the
face or in the fit and nowhere else.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K, synth_rect                      # noqa: E402

from trainit_perception.ranking import (RectFace, centroid, expected_area_px,  # noqa: E402
                                        extent_terms, face_schema, make_face, mad_extent,
                                        p90_extent, plane_ransac, pose_quality, visible_fraction)

A, B = 0.030, 0.020                  # a 60 x 40 mm face: a label panel, a carton's end
F = 424.0
RECT = {'expected_half_a_m': A, 'expected_half_b_m': B}


def rect():
    return make_face('rect', RECT)


# --- what the face DECLARES -----------------------------------------------------------------

def test_the_face_declares_everything_a_configurator_needs():
    assert face_schema('rect') == {
        'name': 'rect',
        'params': ('expected_half_a_m', 'expected_half_b_m'),
        'extent_term_keys': ('extent_a_ratio', 'extent_b_ratio'),
        'enables_terms': ('visible_fraction',)}
    assert isinstance(rect(), RectFace)


def test_BOTH_half_extents_are_needed_because_one_side_is_not_a_size():
    assert rect().has_known_area is True
    for partial in ({'expected_half_a_m': A}, {'expected_half_b_m': B}, {}):
        face = make_face('rect', partial)
        assert face.has_known_area is False, 'half a rectangle is not a known size'
        assert face.expected_area_px(0.6, K) is None
        assert face.extent_terms((A, B)) == {}
        # still DECLARED, so a rule that needs the term is refused rather than silently degenerate
        assert face.extent_term_keys == ('extent_a_ratio', 'extent_b_ratio')


def test_a_SIZE_IN_MILLIMETRES_is_refused_instead_of_silently_divided_by():
    """ADR-0018 Decision 4: the unit error does not raise on its own. The survey's own run
    reported a 51 m^2 face and a 4042 mm radius without a word."""
    with pytest.raises(ValueError) as e:
        make_face('rect', {'expected_half_a_m': 30.0, 'expected_half_b_m': 20.0})   # mm as metres
    assert 'MILLIMETRES' in str(e.value) and 'expected_half_a_m' in str(e.value)
    with pytest.raises(ValueError):
        make_face('rect', {'expected_half_a_m': A, 'expected_half_b_m': 1e-9})
    with pytest.raises(ValueError):
        make_face('rect', {'expected_half_a_m': A, 'expected_half_b_m': -B})


# --- expected_area_px: the projected rectangle ------------------------------------------------

def test_the_expected_area_is_the_PROJECTED_RECTANGLE_and_falls_with_the_square_of_depth():
    face = rect()
    for z in (0.5, 0.6, 0.8):
        assert face.expected_area_px(z, K) == pytest.approx((2 * A * F / z) * (2 * B * F / z))
    assert face.expected_area_px(1.2, K) == pytest.approx(face.expected_area_px(0.6, K) / 4.0)
    # and it is NOT a disc: the same half-extents as a disc radius cover a different area
    assert face.expected_area_px(0.6, K) != pytest.approx(
        make_face('disc', {'expected_radius_m': A}).expected_area_px(0.6, K))


def test_IT_MATCHES_A_SYNTHETIC_MASK_and_does_not_depend_on_the_ROLL_about_its_normal():
    """MEASURED on the raster, 60 x 40 mm at 600 mm: 1,000 of the analytic area at 30 deg of
    roll, 1,002 at 73 deg. The axis-aligned case reads 1,04 because every pixel along an edge
    falls in or out together -- a quantisation of the raster, not of the formula."""
    face = rect()
    want = face.expected_area_px(0.600, K)
    for roll, tol in ((0.0, 0.05), (30.0, 0.01), (45.0, 0.05), (73.0, 0.01)):
        mask, _depth, _c, _n, _axes = synth_rect((0.0, 0.0, 0.600), (0, 0, -1), A, B, roll)
        assert int(mask.sum()) == pytest.approx(want, rel=tol), f'roll {roll}'


def test_visible_fraction_uses_it_as_the_DENOMINATOR_with_no_change_to_the_term():
    mask, depth, _c, _n, _axes = synth_rect((0.0, 0.0, 0.600), (0, 0, -1), A, B, 30.0)
    fit = plane_ransac(mask, depth, K, face=rect())
    assert visible_fraction(int(mask.sum()), expected_area_px(rect(), fit, K)) == \
        pytest.approx(1.0, abs=0.02)
    # half the panel covered reads a half, which is the whole point of the denominator
    assert visible_fraction(0.5 * int(mask.sum()),
                            expected_area_px(rect(), fit, K)) == pytest.approx(0.5, abs=0.02)


# --- half_extents: PER AXIS, which is the whole difference from a disc -----------------------

def test_the_two_axes_are_measured_SEPARATELY_and_the_p90_calibration_is_0_90():
    """A uniform fill's 90th percentile is 0,9 of its half-width on EACH axis, against 0,9487 of
    a DISC's radius -- the estimator is calibrated on the shape, so the shape owns the choice."""
    face = rect()
    rng = np.random.default_rng(5)
    pts = np.stack([rng.uniform(-A, A, 60000), rng.uniform(-B, B, 60000)], 1)
    p90 = face.half_extents(pts, p90_extent)
    mad = face.half_extents(pts, mad_extent)
    assert p90[0] == pytest.approx(0.9005 * A, rel=0.02)
    assert p90[1] == pytest.approx(0.9005 * B, rel=0.02)
    assert mad[0] == pytest.approx(0.7408 * A, rel=0.02)
    assert p90[0] / p90[1] == pytest.approx(A / B, rel=0.03), 'the two axes are KEPT APART'
    assert face.half_extents(np.zeros((0, 2)), p90_extent) == (0.0, 0.0)
    # a disc, by contrast, collapses the very same points to one number twice
    disc = make_face('disc', {'expected_radius_m': A})
    assert disc.half_extents(pts, p90_extent)[0] == disc.half_extents(pts, p90_extent)[1]


def test_A_AND_B_ARE_INTERCHANGEABLE_because_a_plane_fits_AXES_CARRY_NO_LABELS():
    """MEASURED, and this is why the comparison sorts: for a face configured 30 x 20 mm,
    `centroid` handed back (14,69, 23,08) mm -- the IMAGE axes, in the other order. Trusting the
    fit's axis order would be a silent mis-measurement on exactly the hardest candidates."""
    swapped = make_face('rect', {'expected_half_a_m': B, 'expected_half_b_m': A})
    assert rect().extent_terms((A, B)) == swapped.extent_terms((A, B))
    assert rect().extent_terms((A, B)) == rect().extent_terms((B, A))
    assert rect().extent_terms((A, B))['extent_a_ratio'] == pytest.approx(1.0)
    assert rect().extent_terms((A, B))['extent_b_ratio'] == pytest.approx(1.0)
    # the measured case: the fit reports the short axis first, and the ratios still come out right
    mask, depth, _c, _n, _axes = synth_rect((0.0, 0.0, 0.600), (0, 0, -1), A, B, 0.0)
    fc = centroid(mask, depth, K, face=rect())
    assert fc.half_extents_m[0] < fc.half_extents_m[1], 'the fit really did swap them over'
    assert fc.quality['extent_a_ratio'] > fc.quality['extent_b_ratio'] * 0.8


# --- extent_terms: TWO gates, and `pose_quality` reads them with no change -------------------

def test_the_fit_carries_BOTH_ratios_and_pose_quality_gates_on_BOTH():
    """ZERO changes to `fit3d`, `terms` or `ranker`: the face's keys simply appear in the fit's
    quality dict and `pose_quality(extent_keys=face.extent_term_keys)` gates on them."""
    face = rect()
    mask, depth, _c, _n, _axes = synth_rect((0.0, 0.0, 0.600), (0, 0, -1), A, B, 30.0)
    fit = plane_ransac(mask, depth, K, face=face)
    assert set(fit.quality) >= {'extent_a_ratio', 'extent_b_ratio'}
    assert 'radius_ratio' not in fit.quality, 'the gate follows the FACE, not the fit'
    # MEASURED: p90 reads 0,90 of each true half-extent, so a CORRECT rectangle reads ~0,90
    assert fit.quality['extent_a_ratio'] == pytest.approx(0.90, abs=0.03)
    assert fit.quality['extent_b_ratio'] == pytest.approx(0.90, abs=0.03)
    pq = pose_quality(fit, extent_keys=face.extent_term_keys)
    assert 0.5 < pq < 0.85, 'inside the gate, and paying for the estimator bias'
    assert extent_terms(face, fit) == pytest.approx(
        {'extent_a_ratio': fit.quality['extent_a_ratio'],
         'extent_b_ratio': fit.quality['extent_b_ratio']})


def test_A_SQUARE_FITTED_WHERE_A_LONG_PANEL_WAS_CONFIGURED_IS_CAUGHT_PER_AXIS():
    """WHAT A SINGLE AVERAGED RADIUS CANNOT SEE, and the reason there are two terms.

    A 40 x 40 mm surface where a 60 x 40 mm face was configured -- a mask that caught two thirds
    of the panel -- has very nearly the RIGHT MEAN extent and the WRONG SHAPE. MEASURED: the rect
    face reads 0,667 on the long axis and 1,001 on the short one and halves `pose_quality` to
    0,334, while a disc face of the mean radius reads a single `radius_ratio` of 0,913 and scores
    0,825 -- it almost gets away with it.

    Note on the 1,001: a SQUARE's two in-plane variances are equal, so the SVD's axes are
    degenerate and land near the diagonals, where the 90th percentile is about 0,97 of the
    half-width rather than 0,90. The axes being arbitrary on a square is precisely why
    `extent_terms` sorts rather than trusting their order.
    """
    face = rect()
    mask, depth, _c, _n, _axes = synth_rect((0.0, 0.0, 0.600), (0, 0, -1), B, B, 0.0)
    fit = plane_ransac(mask, depth, K, face=face)
    assert fit.quality['extent_a_ratio'] == pytest.approx(0.667, abs=0.02)
    assert fit.quality['extent_b_ratio'] == pytest.approx(1.00, abs=0.02)
    assert pose_quality(fit, extent_keys=face.extent_term_keys) == pytest.approx(0.334, abs=0.02)
    # a DISC face averages the two and almost gets away with it; the rect face does not
    disc = make_face('disc', {'expected_radius_m': 0.5 * (A + B)})
    disc_fit = plane_ransac(mask, depth, K, face=disc)
    assert disc_fit.quality['radius_ratio'] == pytest.approx(0.913, abs=0.02)
    assert pose_quality(disc_fit, extent_keys=disc.extent_term_keys) > \
        2 * pose_quality(fit, extent_keys=face.extent_term_keys)


def test_a_TILTED_rectangle_reads_LOW_and_nothing_divides_the_cosine_back_out():
    """The same measured behaviour the disc has (ADR-0018 Decision 5): the area term reports what
    the camera gets, and the tilt is the `edge_on` TERM instead of a correction folded in here."""
    face = rect()
    square_on = face.expected_area_px(0.600, K)
    for tilt in (20, 40, 60):
        nx, nz = math.sin(math.radians(tilt)), -math.cos(math.radians(tilt))
        mask, _d, _c, _n, _axes = synth_rect((0.0, 0.0, 0.600), (nx, 0.0, nz), A, B, 0.0)
        got = visible_fraction(int(mask.sum()), square_on)
        assert got == pytest.approx(math.cos(math.radians(tilt)), abs=0.05), f'tilt {tilt}'
