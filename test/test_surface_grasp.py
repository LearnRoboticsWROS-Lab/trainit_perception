"""SurfaceGrasp (vNext M4-A): deterministic geometric fixtures whose answer is known exactly.

Every row of the M4 GO's M4-A list is a test here (or in test_surface_grasp_msgs.py / the TMR gtests): a valid
surface, deterministic multi-candidate ranking, tie-breaking, the normal hemisphere (both signs), approach
normalisation, the full orientation, the cone boundary, EEF / site compatibility, standoff bounds, visibility and
clearance, invalid / NaN geometry, no candidate, stable ids / terms / codes, lens distortion, repeatability.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from surface_grasp_scenes import BASE, Disc, distorted, pinhole, regions, render, tilted, unit  # noqa: E402

from trainit_perception.grasp import (CHECK_ORDER, CameraModel, CameraModelError, ConfigError,  # noqa: E402
                                      NORMAL_POLICY, PROVIDER_ID, Region, SurfaceGraspConfig, diagnostics,
                                      minimal_rotation, orient_toward_observer, propose, quat_xyzw)
from trainit_perception.grasp import surface_grasp as SG  # noqa: E402

R_FACE = 0.021


def cfg(**kw):
    p = dict(BASE)
    p.update(kw)
    return SurfaceGraspConfig.from_params(p)


def run(discs, camera=None, **kw):
    cam = camera or pinhole()
    depth, masks = render(discs, cam)
    return propose(regions(discs, masks), depth, cam, cfg(**kw)), depth, masks


def R_of(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def deg(a, b):
    a, b = unit(a), unit(b)
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), a @ b))


# --- 1. one obvious valid surface -------------------------------------------------------------------------------

def test_a_flat_disc_square_to_the_camera_gives_one_exact_candidate():
    disc = Disc((0.01, -0.02, 0.5), (0, 0, -1), R_FACE)
    res, _, _ = run([disc])
    assert not res.rejections and len(res.candidates) == 1
    c = res.candidates[0]
    assert np.linalg.norm(np.array(c.point) - disc.centre) < 1e-3
    assert deg(c.normal, (0, 0, -1)) < 0.05
    assert deg(c.approach, (0, 0, 1)) < 0.05                      # INTO the surface, away from the lens
    assert c.eef == 'main' and c.site == 'suction' and c.standoff_m == 0.05
    assert c.rank == 0 and 0 < c.score <= 1 and 0 < c.confidence <= 1
    assert c.id == 'part@' + f'{c.region_pixel[0]},{c.region_pixel[1]}'
    assert np.allclose(c.pregrasp, np.array(c.point) - 0.05 * np.array(c.approach))
    assert set(c.not_evaluated) >= {'reachability', 'collision'}


def test_on_a_curved_face_the_contact_is_on_the_measured_surface_not_inside_it():
    # a 25 mm sphere seen from 0.5 m: the plane through the RANSAC band lies INSIDE the cap; the tool must be sent
    # to the surface (the D0.7 Replicator evidence measured 2.4 mm inside before the contact was moved onto it)
    c, r = np.array([0.02, 0.01, 0.5]), 0.025
    depth, masks = render([], spheres=[(c, r)])
    reg = [Region('ball', 0.9, masks[0])]
    res = propose(reg, depth, pinhole(), cfg(contact_radius_m=0.006, pose_quality_min_inliers=0.3))
    cand = res.candidates[0]
    assert abs(np.linalg.norm(np.array(cand.point) - c) - r) < 5e-4          # on the sphere (2.8 mm before)
    assert deg(cand.normal, np.array(cand.point) - c) < 5.0                # about radial


# --- 2. normal hemisphere: the view ray decides, never n_z, and both input signs give the same answer --------------

@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_the_normal_faces_the_observer_whatever_sign_the_plane_fit_returns(sign):
    p = np.array([0.1, 0.0, 0.5])
    n = unit((0.2, 0.0, -1.0)) * sign
    out = orient_toward_observer(p, n)
    assert out @ (-p) > 0
    assert np.allclose(out, unit((0.2, 0.0, -1.0)))


def test_off_axis_a_view_ray_hemisphere_differs_from_the_n_z_rule_and_the_view_ray_wins():
    # A face far right, turned toward the camera: its observer-facing normal has n_z > 0. The old n_z < 0 rule
    # (fit3d._toward_camera) would point it AWAY from the camera, and the approach would come out of the surface.
    p = np.array([0.2, 0.0, 0.5])
    n_true = unit((-0.95, 0.0, 0.05))                  # faces the camera: n . (-p) > 0
    assert n_true @ (-p) > 0 and n_true[2] > 0
    disc = Disc(tuple(p), tuple(n_true), 0.03)
    cam = CameraModel.pinhole(np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]), 320, 240, 'cam_optical')
    depth, masks = render([disc], cam, background_z=None)
    assert masks[0].sum() > 100
    res = propose(regions([disc], masks), depth, cam,
                  cfg(approach_cone_axis=[1.0, 0.0, 0.0], reference_x_axis=[0.0, 0.0, -1.0], approach_cone_deg=30.0,
                      max_view_angle_deg=85.0))
    assert res.candidates, [r.why for r in res.rejections]
    c = res.candidates[0]
    assert np.array(c.normal) @ (-np.array(c.point)) > 0          # toward the observer
    assert deg(c.normal, n_true) < 0.5
    assert deg(c.approach, -n_true) < 0.5                         # into the surface


def test_the_normal_policy_is_recorded_in_every_candidate():
    res, _, _ = run([Disc((0, 0, 0.5), (0, 0, -1), R_FACE)])
    md = dict(res.candidates[0].metadata())
    assert md['normal_policy'] == NORMAL_POLICY == 'toward_observer'
    assert md['not_evaluated'].split(',') == sorted(res.candidates[0].not_evaluated)


# --- 3. approach normalisation and the full orientation ---------------------------------------------------------

@pytest.mark.parametrize('tilt', [0.0, 10.0, 25.0, 40.0])
def test_the_approach_is_unit_and_the_orientation_carries_it_on_its_plus_z(tilt):
    res, _, _ = run([Disc((0.0, 0.0, 0.5), tilted(tilt), R_FACE)])
    c = res.candidates[0]
    a = np.array(c.approach)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-12
    q = np.array(c.orientation_xyzw)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12 and q[3] >= 0
    R = R_of(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[:, 2], a, atol=1e-9)
    assert deg(a, -np.array(tilted(tilt))) < 0.1


def test_the_orientation_is_the_reference_tilted_minimally_the_m3_6_1_rule():
    c = run([Disc((0.0, 0.0, 0.5), tilted(30.0), R_FACE)])[0].candidates[0]
    R = R_of(c.orientation_xyzw)
    M = minimal_rotation(np.array([0, 0, 1.0]), np.array(c.approach))
    assert np.allclose(R, M @ np.eye(3), atol=1e-9)              # R_ref = identity (default axes)
    # square on: exactly the reference -- no arbitrary yaw
    c0 = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0].candidates[0]
    assert np.allclose(R_of(c0.orientation_xyzw), np.eye(3), atol=1e-6)


def test_the_reference_x_axis_sets_the_rotation_about_the_approach_and_nothing_else():
    disc = Disc((0.0, 0.0, 0.5), tilted(20.0), R_FACE)
    a = run([disc])[0].candidates[0]
    b = run([disc], reference_x_axis=[0.0, 1.0, 0.0])[0].candidates[0]
    Ra, Rb = R_of(a.orientation_xyzw), R_of(b.orientation_xyzw)
    assert np.allclose(Ra[:, 2], Rb[:, 2], atol=1e-12)            # same approach
    assert abs(deg(Ra[:, 0], Rb[:, 0]) - 90.0) < 1e-6            # yaw differs by the reference's 90 deg


def test_quat_xyzw_round_trips_and_has_one_sign():
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = unit(rng.normal(size=4))
        R = R_of(q)
        out = np.array(quat_xyzw(R))
        assert out[3] >= 0
        assert np.allclose(R_of(out), R, atol=1e-12)


# --- 4. the approach cone: inclusive boundary, never clamped -------------------------------------------------------

def test_the_cone_boundary_is_inclusive_and_one_nanoradian_past_it_is_rejected():
    disc = Disc((0.0, 0.0, 0.5), tilted(30.0), R_FACE)
    measured = run([disc])[0].candidates[0].approach_angle_deg
    at = run([disc], approach_cone_deg=measured)[0]
    assert len(at.candidates) == 1                                   # = cone: valid
    past = run([disc], approach_cone_deg=measured - math.degrees(1e-8))[0]
    assert not past.candidates and past.rejections[0].code == SG.APPROACH_OUTSIDE_CONE
    assert 'never clamped' in past.rejections[0].why


def test_a_candidate_outside_the_cone_is_rejected_not_rotated_into_it():
    res = run([Disc((0.0, 0.0, 0.5), tilted(50.0), R_FACE)], approach_cone_deg=40.0)[0]
    assert not res.candidates
    assert [r.code for r in res.rejections] == [SG.APPROACH_OUTSIDE_CONE]


def test_the_cone_axis_is_honoured_in_the_observation_frame():
    disc = Disc((0.0, 0.0, 0.5), tilted(35.0), R_FACE)
    axis = list(-np.array(tilted(35.0)))
    assert run([disc], approach_cone_deg=5.0, approach_cone_axis=axis)[0].candidates
    assert not run([disc], approach_cone_deg=5.0)[0].candidates


def test_a_grazing_surface_is_rejected_before_its_normal_is_used():
    res = run([Disc((0.0, 0.0, 0.5), tilted(84.0), 0.04)], max_view_angle_deg=80.0, approach_cone_deg=89.0)[0]
    assert not res.candidates
    assert res.rejections[0].code in (SG.NORMAL_GRAZING, SG.INSUFFICIENT_DEPTH, SG.FIT_FAILED)


# --- 5. ranking: deterministic, inspectable, ties by id -----------------------------------------------------------

def test_three_candidates_are_ranked_by_the_product_of_their_terms():
    discs = [Disc((-0.08, 0.0, 0.5), tilted(30.0), R_FACE), Disc((0.0, 0.0, 0.5), tilted(0.0), R_FACE),
             Disc((0.08, 0.0, 0.5), tilted(15.0), R_FACE)]
    res = run(discs)[0]
    assert [c.rank for c in res.candidates] == [0, 1, 2]
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)
    for c in res.candidates:
        assert c.score == pytest.approx(float(np.prod(list(c.terms.values()))), rel=1e-12)
        assert list(c.terms) == [t for t in SG.TERMS if t in c.terms]
    # square-on first, then 15, then 30 deg: alignment decides here
    assert [round(c.approach_angle_deg) for c in res.candidates] == [0, 15, 30]


def test_equal_scores_are_ordered_by_id():
    # mirror images about the optical axis: the same terms, different ids
    discs = [Disc((0.06, 0.0, 0.5), (0, 0, -1), R_FACE), Disc((-0.06, 0.0, 0.5), (0, 0, -1), R_FACE)]
    res = run(discs)[0]
    a, b = res.candidates
    assert a.score == pytest.approx(b.score, rel=1e-9)
    if a.score == b.score:
        assert a.id < b.id
    ids = sorted(c.id for c in res.candidates)
    res2 = propose(list(reversed(regions(discs, render(discs)[1]))), render(discs)[0], pinhole(), cfg())
    assert [c.id for c in res2.candidates] == [c.id for c in res.candidates]       # input order does not matter
    assert sorted(c.id for c in res2.candidates) == ids


def test_ties_break_by_id_exactly():
    a = SG.Candidate('b@1,1', 'x', (0, 0, .5), (0, 0, -1), (0, 0, 1), (0, 0, 0, 1), .05, 'main', 'suction', .5, .5,
                     {}, (), 0, 0, (1, 1), (.02, .02))
    b = SG.Candidate('a@1,1', 'x', (0, 0, .5), (0, 0, -1), (0, 0, 1), (0, 0, 0, 1), .05, 'main', 'suction', .5, .5,
                     {}, (), 0, 0, (1, 1), (.02, .02))
    out = sorted([a, b], key=lambda c: (-c.score, c.id))
    assert [c.id for c in out] == ['a@1,1', 'b@1,1']


def test_max_candidates_reports_the_rest_as_rejected_with_their_rank():
    discs = [Disc((x, 0.0, 0.5), (0, 0, -1), R_FACE) for x in (-0.1, 0.0, 0.1)]
    res = run(discs, max_candidates=2)[0]
    assert len(res.candidates) == 2
    assert [r.code for r in res.rejections] == [SG.BEYOND_MAX_CANDIDATES]
    assert 'ranked 2' in res.rejections[0].why


# --- 6. EEF / site compatibility ---------------------------------------------------------------------------------

def test_a_face_smaller_than_the_site_contact_is_rejected_for_that_site():
    res = run([Disc((0.0, 0.0, 0.3), (0, 0, -1), 0.006)], contact_radius_m=0.008)[0]
    assert [r.code for r in res.rejections] == [SG.SURFACE_TOO_SMALL]
    assert 'main/suction' in res.rejections[0].why
    assert run([Disc((0.0, 0.0, 0.3), (0, 0, -1), 0.012)], contact_radius_m=0.008)[0].candidates


def test_a_narrow_band_is_too_small_even_when_its_long_axis_is_not():
    # the side of a cap seen edge-on: a band 7 mm wide and 42 mm long. A disc face's radial extent reads it as
    # ~20 mm; the seal needs the NARROWER axis (found by M4-B on randomized capsules)
    disc = Disc((0.0, 0.0, 0.3), (0, 0, -1), 0.021)
    depth, masks = render([disc])
    v = np.nonzero(masks[0])[0]
    band = masks[0].copy()
    half_px = int(round(0.0035 * 300 / 0.3))              # 3.5 mm at 0.3 m, f = 300 px
    vc = int(round(v.mean()))
    band[:vc - half_px] = False
    band[vc + half_px + 1:] = False
    res = propose([Region('part', 0.9, band)], depth, pinhole(), cfg(contact_radius_m=0.009, min_points=20))
    assert [r.code for r in res.rejections] == [SG.SURFACE_TOO_SMALL]
    assert 'narrower principal axis' in res.rejections[0].why
    assert res.rejections[0].terms['inscribed_mm'] < 5.0                  # 4 px = 4 mm at 0.3 m
    whole = propose([Region('part', 0.9, masks[0])], depth, pinhole(), cfg(contact_radius_m=0.009))
    assert whole.candidates and whole.candidates[0].terms['seal'] > 0.3


@pytest.mark.parametrize('bad, word', [({'eef': ''}, 'eef is required'), ({'site': ''}, 'site is required'),
                                       ({'grasp_mode': 'parallel'}, 'grasp_mode'),
                                       ({'contact_radius_m': 0.0}, 'contact_radius_m'),
                                       ({'contact_radius_m': float('nan')}, 'finite')])
def test_an_unstated_or_incompatible_end_effector_is_refused_at_start_up(bad, word):
    with pytest.raises(ConfigError, match=word):
        cfg(**bad)


def test_every_candidate_names_the_configured_eef_and_site_never_a_default():
    c = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)], eef='tool2', site='cup_b')[0].candidates[0]
    assert (c.eef, c.site) == ('tool2', 'cup_b')


# --- 7. standoff -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('s, ok', [(0.001, True), (0.5, True), (0.05, True), (0.0009, False), (0.51, False),
                                   (0.0, False), (-0.05, False), (float('nan'), False), (float('inf'), False)])
def test_the_standoff_must_lie_in_its_band_and_is_never_clamped(s, ok):
    if ok:
        assert cfg(standoff_m=s).standoff_m == s
    else:
        with pytest.raises(ConfigError):
            cfg(standoff_m=s)


def test_the_standoff_band_itself_is_validated():
    with pytest.raises(ConfigError, match='standoff_max_m'):
        cfg(standoff_min_m=0.1, standoff_max_m=0.05, standoff_m=0.07)
    with pytest.raises(ConfigError, match='standoff_min_m'):
        cfg(standoff_min_m=0.0)
    assert cfg(standoff_min_m=0.02, standoff_max_m=0.02, standoff_m=0.02).standoff_m == 0.02


# --- 8. configuration refusals ------------------------------------------------------------------------------------

@pytest.mark.parametrize('bad', [{'approach_cone_deg': 0.0}, {'approach_cone_deg': 90.0},
                                 {'approach_cone_deg': -5.0}, {'approach_cone_deg': float('nan')},
                                 {'approach_cone_axis': [0, 0, 0]}, {'approach_cone_axis': [1, 2]},
                                 {'reference_x_axis': [0, 0, 2]}, {'max_view_angle_deg': 0.0},
                                 {'not_a_parameter': 1}, {'tool_radius_m': 0.01},
                                 {'min_clearance': 0.5}, {'min_visible_fraction': 0.5},
                                 {'max_candidates': 0}, {'min_points': 2}, {'z_min_m': 2.0, 'z_max_m': 1.0}])
def test_a_configuration_the_provider_cannot_honour_is_refused(bad):
    with pytest.raises(ConfigError):
        cfg(**bad)


def test_the_cone_has_no_default():
    p = dict(BASE)
    del p['approach_cone_deg']
    with pytest.raises(ConfigError, match='approach_cone_deg'):
        SurfaceGraspConfig.from_params(p)


# --- 9. visibility and clearance -----------------------------------------------------------------------------------

def test_a_half_covered_face_reads_half_visible_and_a_threshold_rejects_it():
    back = Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE, class_id='back')
    front = Disc((0.021, 0.0, 0.45), (0, 0, -1), 0.0212, class_id='front')
    res = run([back, front], expected_radius_m=R_FACE)[0]
    vis = {c.class_id: c.terms['visibility'] for c in res.candidates}
    assert 0.3 < vis['back'] < 0.7 and vis['front'] > 0.9
    res2 = run([back, front], expected_radius_m=R_FACE, min_visible_fraction=0.8)[0]
    assert {r.id.split('@')[0]: r.code for r in res2.rejections} == {'back': SG.VISIBILITY_LOW}


def test_visibility_is_not_evaluated_without_a_face_size_and_says_so():
    c = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0].candidates[0]
    assert 'visibility' not in c.terms and 'visibility' in c.not_evaluated


def test_an_obstruction_in_the_tool_path_lowers_clearance_and_a_threshold_rejects_it():
    face = Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)
    bar = Disc((0.004, 0.0, 0.47), (0, 0, -1), 0.004, class_id='bar')         # 30 mm above, in the cup's path
    clear = run([face], tool_radius_m=0.010, tool_reach_mm=8.0)[0].candidates[0]
    assert clear.terms['clearance'] == 1.0
    res = run([face, bar], tool_radius_m=0.010, tool_reach_mm=8.0, min_clearance=0.5)[0]
    codes = {r.id.split('@')[0]: r.code for r in res.rejections}
    assert codes.get('part') == SG.CLEARANCE_LOW


def test_a_clearance_the_depth_cannot_measure_is_a_rejection_not_a_pass(monkeypatch):
    monkeypatch.setattr(SG, 'approach_obstruction', lambda *a, **k: float('nan'))
    res = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)], tool_radius_m=0.01, tool_reach_mm=8.0)[0]
    assert [r.code for r in res.rejections] == [SG.CLEARANCE_UNKNOWN]


def test_nonoverlap_is_evaluated_only_against_other_boxes():
    one = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0].candidates[0]
    assert 'nonoverlap' in one.not_evaluated
    two = run([Disc((-0.06, 0.0, 0.5), (0, 0, -1), R_FACE), Disc((0.06, 0.0, 0.5), (0, 0, -1), R_FACE)])[0]
    assert all(c.terms['nonoverlap'] == 1.0 for c in two.candidates)


# --- 10. invalid geometry, no candidate --------------------------------------------------------------------------

def test_no_region_is_the_valid_no_candidate_answer():
    depth, _ = render([])
    res = propose([], depth, pinhole(), cfg())
    assert res.candidates == [] and res.rejections == [] and res.n_regions == 0


def test_an_empty_mask_nan_depth_and_noise_are_rejected_with_their_own_code():
    disc = Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)
    depth, masks = render([disc])
    empty = Region('e', 0.9, np.zeros_like(masks[0]))
    nan_depth = depth.copy()
    nan_depth[masks[0]] = np.nan
    rng = np.random.default_rng(0)
    noisy = depth.copy()
    noisy[masks[0]] = 0.4 + rng.uniform(0.0, 0.2, int(masks[0].sum()))
    assert propose([empty], depth, pinhole(), cfg()).rejections[0].code == SG.REGION_EMPTY
    assert propose(regions([disc], masks), nan_depth, pinhole(), cfg()).rejections[0].code == SG.INSUFFICIENT_DEPTH
    noise = propose(regions([disc], masks), noisy, pinhole(), cfg())
    assert not noise.candidates                                   # whatever plane RANSAC proposes in noise
    assert noise.rejections[0].code in (SG.FIT_FAILED, SG.NORMAL_GRAZING, SG.APPROACH_OUTSIDE_CONE,
                                        SG.POSE_QUALITY_REFUSED)


def test_a_non_finite_fit_never_becomes_a_candidate(monkeypatch):
    real = SG.plane_ransac_points

    def broken(*a, **k):
        f = real(*a, **k)
        f.normal = (float('nan'), 0.0, -1.0)
        return f
    monkeypatch.setattr(SG, 'plane_ransac_points', broken)
    res = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0]
    assert [r.code for r in res.rejections] == [SG.GEOMETRY_INVALID] and not res.candidates


def test_a_zero_length_normal_never_becomes_a_candidate(monkeypatch):
    real = SG.plane_ransac_points

    def broken(*a, **k):
        f = real(*a, **k)
        f.normal = (0.0, 0.0, 0.0)
        return f
    monkeypatch.setattr(SG, 'plane_ransac_points', broken)
    assert run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0].rejections[0].code == SG.GEOMETRY_INVALID


def test_a_face_far_bigger_than_the_configured_one_is_refused_by_the_extent_gate():
    res = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), 0.06)], expected_radius_m=0.021)[0]
    assert res.rejections[0].code == SG.POSE_QUALITY_REFUSED


def test_min_pose_quality_rejects_below_it():
    c = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0].candidates[0]
    res = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)], min_pose_quality=min(1.0, c.confidence + 1e-6))[0]
    if c.confidence < 1.0:
        assert res.rejections[0].code == SG.POSE_QUALITY_LOW


def test_the_codes_are_a_closed_ordered_list():
    # 0.8.0 (M4.5d, ADR-0031 amendment): FACE_FIT_FAILED, right after the plane's own checks
    assert len(set(CHECK_ORDER)) == len(CHECK_ORDER) == 16
    assert CHECK_ORDER.index('FACE_FIT_FAILED') == CHECK_ORDER.index('GEOMETRY_INVALID') + 1
    assert all(c.isupper() for c in CHECK_ORDER)


# --- 11. determinism ---------------------------------------------------------------------------------------------

def _strip(res):
    return ([(c.id, c.rank, c.score, c.confidence, c.terms, c.point, c.normal, c.approach, c.orientation_xyzw,
              c.not_evaluated) for c in res.candidates],
            [(r.id, r.code, r.why, r.terms) for r in res.rejections])


def test_repeated_identical_input_gives_identical_ids_terms_scores_order_and_rejections():
    discs = [Disc((-0.08, 0.01, 0.5), tilted(20.0), R_FACE), Disc((0.0, 0.0, 0.48), tilted(5.0, 'x'), R_FACE),
             Disc((0.08, -0.01, 0.52), tilted(50.0), R_FACE), Disc((0.0, 0.08, 0.5), (0, 0, -1), 0.004)]
    first = _strip(run(discs, expected_radius_m=R_FACE)[0])
    for _ in range(3):
        assert _strip(run(discs, expected_radius_m=R_FACE)[0]) == first
    codes = sorted(r[1] for r in first[1])
    assert SG.APPROACH_OUTSIDE_CONE in codes


def test_diagnostics_are_strict_json_with_null_for_non_finite():
    import json
    res = run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE), Disc((0.08, 0.0, 0.5), tilted(60), R_FACE)])[0]
    res.timing_ms['total'] = float('nan')
    d = diagnostics(res, 'cam_optical', 7, 11, pinhole())
    text = json.dumps(d, allow_nan=False)
    back = json.loads(text)
    assert back['provider'] == PROVIDER_ID and back['stamp'] == {'sec': 7, 'nanosec': 11}
    assert back['timing_ms']['total'] is None
    assert back['summary']['valid'] == 1 and back['summary']['rejected'] == 1
    assert back['rejected'][0]['code'] == SG.APPROACH_OUTSIDE_CONE


# --- 12. the camera model: distortion handled, never ignored -----------------------------------------------------

def test_a_distorted_lens_is_undistorted_and_ignoring_it_would_be_millimetres_wrong():
    cam = distorted()
    disc = Disc((0.15, 0.1, 0.5), (0, 0, -1), R_FACE)              # off-axis, where distortion bites
    depth, masks = render([disc], cam)
    good = propose(regions([disc], masks), depth, cam, cfg()).candidates[0]
    wrong = propose(regions([disc], masks), depth, pinhole(), cfg()).candidates[0]
    e_good = np.linalg.norm(np.array(good.point) - disc.centre)
    e_wrong = np.linalg.norm(np.array(wrong.point) - disc.centre)
    assert e_good < 5e-4
    assert e_wrong > 1e-3                                            # the debt this closes, measured


def test_the_camera_model_refuses_what_it_cannot_undistort():
    with pytest.raises(CameraModelError, match='not supported'):
        CameraModel.from_fields(k=[300, 0, 160, 0, 300, 120, 0, 0, 1], d=[0.1, 0, 0, 0], distortion_model='equidistant',
                                width=320, height=240, frame_id='c')
    ok = CameraModel.from_fields(k=[300, 0, 160, 0, 300, 120, 0, 0, 1], d=[0, 0, 0, 0], distortion_model='equidistant',
                                 width=320, height=240, frame_id='c')
    assert not ok.distorted
    for bad in ({'frame_id': ''}, {'k': [0] * 9}, {'width': 0}, {'k': [300, 0, 160, 0, 300, 120, 0, 0, float('nan')]}):
        kw = dict(k=[300, 0, 160, 0, 300, 120, 0, 0, 1], d=[], distortion_model='plumb_bob', width=320, height=240,
                  frame_id='c')
        kw.update(bad)
        with pytest.raises(CameraModelError):
            CameraModel.from_fields(**kw)


def test_the_camera_model_compares_by_value_and_projects_what_it_back_projects():
    a, b = distorted(), distorted()
    assert a.same_as(b) and not a.same_as(pinhole())
    rays = a.rays()
    v, u = 37, 291
    P = np.array([[rays[v, u, 0] * 0.6, rays[v, u, 1] * 0.6, 0.6]])
    assert np.allclose(a.project(P)[0], (u, v), atol=1e-3)


def test_a_depth_image_off_the_camera_grid_is_refused():
    with pytest.raises(ValueError, match='registered depth'):
        propose([], np.zeros((10, 10), np.float32), pinhole(), cfg())


# --- 13. regions from a detector ---------------------------------------------------------------------------------

def test_color_mask_detections_carry_their_masks_and_become_regions():
    from trainit_perception.detectors import make_detector
    rgb = np.zeros((60, 80, 3), np.uint8)
    rgb[10:30, 10:30] = (255, 0, 0)
    rgb[35:55, 40:70] = (255, 0, 0)
    depth = np.full((60, 80), 0.5, np.float32)
    det = make_detector('color_mask', {'class_id': 'red', 'max_objects': 5, 'morph_kernel': 0})
    dets = det.detect(rgb, depth, np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]]))
    regs = SG.regions_from_detections(dets)
    assert [int(r.mask.sum()) for r in regs] == [600, 400]
    assert all(r.class_id == 'red' for r in regs)


def test_a_detection_without_a_mask_is_refused_as_a_region():
    from trainit_perception.detectors.base import Detection
    with pytest.raises(ValueError, match='no mask'):
        SG.regions_from_detections([Detection('x', 0.5, (0, 0, 1))])


# --- review M4: support, bounds, boundaries, determinism on noisy data --------------------------------------------

def _ring(inner_depth):
    """A 16 mm disc at 0.5 m whose inner 10 mm radius shows `inner_depth` (a recess, or 0 = a hole); the region is
    the whole 16 mm disc (a segmenter does not know there is a hole)."""
    disc = Disc((0.0, 0.0, 0.5), (0, 0, -1), 0.016)
    depth, masks = render([disc])
    rays = pinhole().rays()
    r = np.hypot(rays[..., 0] * 0.5, rays[..., 1] * 0.5)
    inner = masks[0] & (r < 0.010)
    depth = depth.copy()
    depth[inner] = inner_depth
    return depth, masks[0]


def test_a_recess_never_pulls_the_contact_into_the_object():
    # a bottle mouth: the bottom 150 mm deeper inside the rim (review M4: the unbounded snap moved the contact there)
    depth, mask = _ring(0.65)
    res = propose([Region('mouth', 0.9, mask)], depth, pinhole(), cfg(contact_radius_m=0.009, min_points=20))
    assert not res.candidates
    assert res.rejections[0].code == SG.CONTACT_UNSUPPORTED
    assert abs(res.rejections[0].terms['contact_offset_mm']) < 1.0          # the snap ignored the bottom


def test_a_ring_around_a_hole_does_not_support_the_cup():
    depth, mask = _ring(0.0)
    res = propose([Region('ring', 0.9, mask)], depth, pinhole(), cfg(contact_radius_m=0.009, min_points=20))
    assert [r.code for r in res.rejections] == [SG.CONTACT_UNSUPPORTED]


def test_an_occluded_contact_is_not_supported_without_a_clearance_term():
    face = Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)
    bar = Disc((0.002, 0.0, 0.47), (0, 0, -1), 0.010, class_id='bar')       # lying on the contact, seen in front
    res = run([face, bar])[0]
    codes = {r.id.split('@')[0]: r.code for r in res.rejections}
    assert codes.get('part') == SG.CONTACT_UNSUPPORTED


def _noisy_scene(seed=7):
    disc = Disc((0.01, 0.0, 0.5), tilted(25.0), R_FACE)
    depth, masks = render([disc])
    rng = np.random.default_rng(seed)
    noisy = depth.copy()
    m = masks[0]
    noisy[m] += rng.normal(0, 0.001, int(m.sum())).astype(np.float32)
    idx = np.flatnonzero(m)
    out = rng.choice(idx, size=len(idx) // 20, replace=False)
    noisy.flat[out] += rng.uniform(-0.03, 0.03, len(out)).astype(np.float32)
    return disc, noisy, masks


def test_noisy_depth_is_still_deterministic_and_the_seed_only_moves_the_contact_sub_millimetre():
    disc, depth, masks = _noisy_scene()
    regs = regions([disc], masks)
    first = _strip(propose(regs, depth, pinhole(), cfg(min_pose_quality=0.0)))
    for _ in range(4):
        assert _strip(propose(regs, depth, pinhole(), cfg(min_pose_quality=0.0))) == first
    a = propose(regs, depth, pinhole(), cfg()).candidates[0]
    b = propose(regs, depth, pinhole(), cfg(seed=12345)).candidates[0]
    assert np.linalg.norm(np.array(a.point) - np.array(b.point)) < 1e-3
    assert deg(a.normal, b.normal) < 1.0
    assert np.linalg.norm(np.array(a.point) - disc.centre) < 2e-3


def test_the_grazing_boundary_is_exclusive_of_the_limit():
    disc = Disc((0.0, 0.0, 0.5), tilted(40.0), R_FACE)
    view = run([disc])[0].candidates[0].view_angle_deg
    assert run([disc], max_view_angle_deg=view)[0].rejections[0].code == SG.NORMAL_GRAZING      # >= refuses
    assert run([disc], max_view_angle_deg=view + 1e-6)[0].candidates


def test_the_seal_boundary_is_inclusive():
    disc = Disc((0.0, 0.0, 0.4), (0, 0, -1), R_FACE)
    r_in = run([disc])[0].candidates[0]
    inscribed = (1.0 - r_in.terms['seal']) ** -1 * 0.008                     # r_in from seal = (r_in - c) / r_in
    ok = run([disc], contact_radius_m=inscribed * (1 - 1e-9))[0]
    assert ok.candidates
    no = run([disc], contact_radius_m=inscribed * (1 + 1e-6))[0]
    assert no.rejections[0].code == SG.SURFACE_TOO_SMALL


def test_min_pose_quality_is_inclusive_and_refuses_just_above():
    disc = Disc((0.0, 0.0, 0.5), tilted(10.0), R_FACE)
    q = run([disc])[0].candidates[0].confidence
    assert run([disc], min_pose_quality=q)[0].candidates
    assert run([disc], min_pose_quality=min(1.0, q + 1e-9))[0].rejections[0].code == SG.POSE_QUALITY_LOW


def test_exactly_min_points_valid_pixels_is_enough():
    disc = Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)
    n = int(render([disc])[1][0].sum())
    assert run([disc], min_points=n)[0].rejections == [] or run([disc], min_points=n)[0].candidates
    assert run([disc], min_points=n + 1)[0].rejections[0].code == SG.INSUFFICIENT_DEPTH


def test_duplicate_region_ids_are_suffixed_deterministically():
    disc = Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)
    depth, masks = render([disc])
    regs = [Region('part', 0.9, masks[0]), Region('part', 0.8, masks[0])]
    ids = sorted(c.id for c in propose(regs, depth, pinhole(), cfg()).candidates)
    base = ids[0]
    assert ids == [base, base + '#1']
    assert sorted(c.id for c in propose(regs, depth, pinhole(), cfg()).candidates) == ids


def test_the_detection_score_enters_the_ranking_and_a_nan_score_is_zero():
    discs = [Disc((-0.06, 0.0, 0.5), (0, 0, -1), R_FACE, score=0.5), Disc((0.06, 0.0, 0.5), (0, 0, -1), R_FACE,
                                                                               score=0.9)]
    res = run(discs)[0]
    assert [c.terms['detection'] for c in res.candidates] == [0.9, 0.5]
    depth, masks = render(discs)
    regs = [Region('part', float('nan'), masks[0])]
    assert propose(regs, depth, pinhole(), cfg()).candidates[0].terms['detection'] == 0.0


def test_the_score_terms_of_a_fixed_scene_are_stable_values():
    c = run([Disc((0.0, 0.0, 0.5), tilted(20.0), R_FACE)], expected_radius_m=R_FACE)[0].candidates[0]
    assert c.terms['detection'] == pytest.approx(0.9)
    assert c.terms['alignment'] == pytest.approx(1 - 20.0 / 45.0, abs=2e-3)   # linear in the angle
    assert c.score == pytest.approx(np.prod(list(c.terms.values())), rel=1e-12)


def test_a_fit_behind_the_camera_is_invalid(monkeypatch):
    real = SG.plane_ransac_points

    def behind(*a, **k):
        f = real(*a, **k)
        f.point = (f.point[0], f.point[1], -0.5)
        return f
    monkeypatch.setattr(SG, 'plane_ransac_points', behind)
    assert run([Disc((0.0, 0.0, 0.5), (0, 0, -1), R_FACE)])[0].rejections[0].code == SG.GEOMETRY_INVALID


@pytest.mark.parametrize('flip', [False, True])
def test_whatever_sign_the_plane_fit_returns_propose_gives_the_same_candidate(monkeypatch, flip):
    real = SG.plane_ransac_points

    def signed(*a, **k):
        f = real(*a, **k)
        if flip:
            f.normal = tuple(-x for x in f.normal)
        return f
    monkeypatch.setattr(SG, 'plane_ransac_points', signed)
    c = run([Disc((0.05, 0.0, 0.5), tilted(15.0), R_FACE)])[0].candidates[0]
    assert np.array(c.normal) @ (-np.array(c.point)) > 0
    assert deg(c.approach, -np.array(tilted(15.0))) < 0.1


def test_the_snap_ignores_spikes_near_the_axis():
    c0, r = np.array([0.0, 0.0, 0.5]), 0.03
    depth, masks = render([], spheres=[(c0, r)])
    rays = pinhole().rays()
    rr = np.hypot(rays[..., 0], rays[..., 1]) * 0.47
    spikes = masks[0] & (rr < 0.004) & ((np.indices(depth.shape).sum(0) % 7) == 0)
    depth = depth.copy()
    depth[spikes] -= 0.006                                      # a few 6 mm spikes toward the camera
    res = propose([Region('ball', 0.9, masks[0])], depth, pinhole(),
                  cfg(contact_radius_m=0.006, pose_quality_min_inliers=0.2))
    c = res.candidates[0]
    assert abs(np.linalg.norm(np.array(c.point) - c0) - r) < 6e-4             # the median, not the mean
