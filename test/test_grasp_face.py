"""The GRASP FACE plug: the four lines that used to carry a disc assumption (ADR-0017 Decision 2).

Each test names the line it stands for, because the point of the plug is that there are exactly
four of them and a new face has to answer exactly those.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K, R_FACE, synth_disc            # noqa: E402

from trainit_perception.ranking import (DiscFace, GraspFace, available_faces,  # noqa: E402
                                        expected_area_px, extent_terms, face_params, make_face,
                                        mad_extent, p90_extent, plane_ransac, register_face)


def test_the_registry_has_the_shipped_faces_and_refuses_a_name_it_does_not_know():
    assert available_faces() == ['cylinder_side', 'disc', 'rect']
    assert isinstance(make_face('disc', {'expected_radius_m': R_FACE}), DiscFace)
    assert face_params('disc') == ('expected_radius_m',)
    with pytest.raises(KeyError) as e:
        make_face('hexagon', {})
    assert 'disc' in str(e.value), 'the message has to say what IS available'


def test_LINE_1_the_expected_area_is_the_disc_the_face_should_show():
    """visible_fraction's denominator: pi * (f * r / z)^2."""
    face = make_face('disc', {'expected_radius_m': R_FACE})
    for z in (0.5, 0.6, 0.8):
        r_px = 424.0 * R_FACE / z
        assert face.expected_area_px(z, K) == pytest.approx(math.pi * r_px * r_px)
    # it HALVES with the square of the depth, which is what makes the term scale-free
    assert face.expected_area_px(1.2, K) == pytest.approx(face.expected_area_px(0.6, K) / 4.0)


def test_LINE_1_a_face_WITHOUT_a_size_answers_NONE_instead_of_guessing():
    """The whole of ADR-0017 Decision 3 rests on this: no size -> no term -> a refusable rule."""
    face = make_face('disc', {})
    assert face.has_known_area is False
    assert face.expected_area_px(0.6, K) is None
    assert face.extent_terms((0.02, 0.02)) == {}
    assert face.extent_term_keys == ('radius_ratio',)       # still DECLARED, just not produced


def test_LINE_2_and_3_the_face_collapses_the_offsets_RADIALLY_for_both_estimators():
    """centroid's MAD radial spread and plane_ransac's hypot-collapsed extent, in one place.

    The arithmetic the two estimators are calibrated against, for a UNIFORMLY FILLED disc of
    radius R: the median radius is R/sqrt(2) = 0,7071 R, so the MAD estimate is 1,4826 x that =
    1,048 R, and the 90th percentile is R x sqrt(0,9) = 0,9487 R. Both land within 5 % of R,
    which is why either can be compared with a configured face radius -- and they are NOT the
    same number, which is why WHICH one to use stays the fit method's decision.
    """
    face = make_face('disc', {'expected_radius_m': R_FACE})
    rng = np.random.default_rng(3)
    pts = rng.random((40000, 2)) * 0.040 - 0.020
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 0.020]              # a filled disc of 20 mm
    mad = face.half_extents(pts, mad_extent)
    p90 = face.half_extents(pts, p90_extent)
    assert mad[0] == pytest.approx(1.0484 * 0.020, rel=0.02)
    assert p90[0] == pytest.approx(0.9487 * 0.020, rel=0.02)
    assert mad[0] == mad[1] and p90[0] == p90[1], \
        'a disc has ONE extent, so the two half-extents come back equal'
    assert abs(mad[0] - p90[0]) > 1e-4, 'two different estimators, as the two fit methods intend'
    # a ring isolates the collapse itself: every radius is exactly 20 mm, so p90 reads 20 mm
    ang = np.linspace(0, 2 * math.pi, 512, endpoint=False)
    ring = np.stack([0.020 * np.cos(ang), 0.020 * np.sin(ang)], 1)
    assert face.half_extents(ring, p90_extent)[0] == pytest.approx(0.020, abs=1e-4)
    # and an empty surface is (0, 0) rather than an exception
    assert face.half_extents(np.zeros((0, 2)), p90_extent) == (0.0, 0.0)


def test_LINE_4_the_extent_terms_are_what_pose_quality_GATES_on():
    face = make_face('disc', {'expected_radius_m': R_FACE})
    assert face.extent_terms((R_FACE, R_FACE))['radius_ratio'] == pytest.approx(1.0)
    assert face.extent_terms((2 * R_FACE, 2 * R_FACE))['radius_ratio'] == pytest.approx(2.0)
    assert set(face.extent_terms((R_FACE, R_FACE))) == set(face.extent_term_keys)


def test_THE_FIELD_the_fit_now_reports_TWO_half_extents():
    """GraspFit.radius_m was one scalar where the plane fit computed two half-extents and threw
    one away. Changed before there are consumers, which is the only cheap moment."""
    mask, depth, c, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    f = plane_ransac(mask, depth, K, face=make_face('disc', {'expected_radius_m': R_FACE}))
    assert not hasattr(f, 'radius_m'), 'the scalar must be GONE, not shadowed'
    a, b = f.half_extents_m
    assert a == pytest.approx(R_FACE, abs=0.002) and b == pytest.approx(R_FACE, abs=0.002)
    assert f.quality['radius_ratio'] == pytest.approx(1.0, abs=0.1)
    # the plug's two module-level entry points, in the shape ADR-0017 Decision 2 names them
    face = make_face('disc', {'expected_radius_m': R_FACE})
    assert expected_area_px(face, f, K) == pytest.approx(face.expected_area_px(0.600, K), rel=1e-3)
    assert extent_terms(face, f) == pytest.approx(f.quality['radius_ratio'] and
                                                  {'radius_ratio': f.quality['radius_ratio']})


def test_a_face_FROM_OUTSIDE_the_package_plugs_in():
    """The escape hatch the registry pattern exists for. A rectangular face measures its two axes
    separately and gates on two terms, and NOTHING in the fit changes."""
    class RectFace(GraspFace):
        name = 'test_rect'
        PARAMS = ('expected_half_a_m', 'expected_half_b_m')
        extent_term_keys = ('extent_a_ratio', 'extent_b_ratio')

        def __init__(self, a=0.0, b=0.0):
            self.a, self.b = float(a), float(b)

        @classmethod
        def from_params(cls, params):
            return cls(params.get('expected_half_a_m', 0.0), params.get('expected_half_b_m', 0.0))

        @property
        def has_known_area(self):
            return self.a > 0 and self.b > 0

        def half_extents(self, offsets, estimator):
            offsets = np.asarray(offsets, float)
            if offsets.size == 0:
                return (0.0, 0.0)
            return (estimator(offsets[:, 0]), estimator(offsets[:, 1]))

        def expected_area_px(self, z_m, K):
            if not self.has_known_area:
                return None
            f = 0.5 * (K[0, 0] + K[1, 1])
            return (2 * self.a * f / z_m) * (2 * self.b * f / z_m)

        def extent_terms(self, half_extents_m):
            if not self.has_known_area:
                return {}
            return {'extent_a_ratio': half_extents_m[0] / self.a,
                    'extent_b_ratio': half_extents_m[1] / self.b}

    try:
        register_face(RectFace)
        assert 'test_rect' in available_faces()
        mask, depth, c, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
        f = plane_ransac(mask, depth, K,
                         face=make_face('test_rect', {'expected_half_a_m': R_FACE,
                                                      'expected_half_b_m': R_FACE}))
        # a disc measured as a rectangle: both axes read the p90 of a projected radius, so both
        # ratios come out under 1 -- and there are TWO of them, which is the point
        assert set(f.quality) >= {'extent_a_ratio', 'extent_b_ratio'}
        assert 'radius_ratio' not in f.quality, 'the gate follows the FACE, not the fit'
    finally:
        from trainit_perception.ranking.faces import FACES
        FACES.pop('test_rect', None)


def test_a_FACE_parameter_passed_to_the_FIT_is_refused_not_ignored():
    """The old signature took `expected_radius_m` itself; leaving it acceptable-and-ignored would
    reopen the exact silent degeneration Decision 3 closes."""
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    with pytest.raises(TypeError) as e:
        plane_ransac(mask, depth, K, expected_radius_m=R_FACE)
    assert 'expected_radius_m' in str(e.value) and 'face' in str(e.value)
