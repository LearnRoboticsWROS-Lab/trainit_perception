"""Stage 3D against geometry whose answer is known exactly.

A synthetic disc at a chosen distance and tilt has one correct centre and one correct normal, so
an error here is in the fit and nowhere else -- no renderer, no model, no mask, no physics.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K, R_FACE, synth_disc            # noqa: E402

from trainit_perception.ranking import (GIVES_ORIENTATION, METHODS,  # noqa: E402
                                        approach_clearance, approach_obstruction, centroid, fit,
                                        make_face, plane_nearest, plane_ransac, pose_quality)

FACE = make_face('disc', {'expected_radius_m': R_FACE})


def test_plane_recovers_a_flat_disc_facing_the_camera():
    mask, depth, c, n = synth_disc((0.01, -0.02, 0.600), (0, 0, -1))
    f = plane_ransac(mask, depth, K, face=FACE)
    assert f.ok
    assert np.linalg.norm(np.array(f.point) - c) < 1e-3                # within a millimetre
    assert math.degrees(math.acos(abs(np.array(f.normal) @ n))) < 1.0  # within a degree
    assert f.half_extents_m[0] == pytest.approx(R_FACE, abs=0.002)
    assert f.quality['inlier_ratio'] > 0.99


@pytest.mark.parametrize('tilt_deg', [0, 15, 30, 45, 60])
def test_plane_recovers_the_normal_at_every_tilt(tilt_deg):
    t = math.radians(tilt_deg)
    n = (math.sin(t), 0.0, -math.cos(t))
    mask, depth, c, n = synth_disc((0.0, 0.0, 0.600), n)
    f = plane_ransac(mask, depth, K, face=FACE)
    assert f.ok
    err = math.degrees(math.acos(min(1.0, abs(np.array(f.normal) @ n))))
    assert err < 1.5, f'{tilt_deg} deg tilt -> normal off by {err:.2f} deg'
    assert np.linalg.norm(np.array(f.point) - c) < 2e-3
    assert f.tilt_deg == pytest.approx(tilt_deg, abs=1.5)


def test_the_normal_always_points_back_at_the_camera():
    """A plane fit has no opinion about which of its two normals is meant; a tool does."""
    for sign in (-1.0, 1.0):
        mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0.3 * sign, 0.2, sign))
        f = plane_ransac(mask, depth, K, face=FACE)
        assert f.normal[2] < 0.0


def test_depth_noise_degrades_the_fit_but_not_the_centre():
    mask, depth, c, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    _, noisy_depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1), noise_mm=1.0, seed=5)
    clean = plane_ransac(mask, depth, K, face=FACE)
    noisy = plane_ransac(mask, noisy_depth, K, face=FACE)
    assert noisy.ok
    assert np.linalg.norm(np.array(noisy.point) - c) < 2e-3          # centre survives noise
    assert noisy.quality['rms_mm'] > clean.quality['rms_mm']         # and the fit admits it


def test_a_mask_with_no_depth_FAILS_LOUDLY_instead_of_guessing():
    mask, _, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    depth = np.full(mask.shape, np.nan, np.float32)
    f = plane_ransac(mask, depth, K, face=FACE)
    assert not f.ok and 'valid depth' in f.why
    assert pose_quality(f) == 0.0
    assert f.half_extents_m == (0.0, 0.0)


def test_centroid_claims_no_orientation_and_still_measures_the_extent():
    mask, depth, c, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    f = centroid(mask, depth, K, face=FACE)
    assert f.ok and f.normal == (0.0, 0.0, -1.0)
    assert GIVES_ORIENTATION['centroid'] is False
    assert GIVES_ORIENTATION['plane_ransac'] is True
    # the MAD radial spread, through the face: within 6 % of the real radius on a filled disc
    assert f.half_extents_m[0] == pytest.approx(R_FACE, rel=0.06)
    assert f.quality['radius_ratio'] == pytest.approx(1.0, abs=0.06)


def test_a_fit_WITHOUT_a_face_measures_the_extent_and_CLAIMS_NOTHING_about_it():
    """No face configured = the size is unknown, so there is no ratio and no gate -- and the
    rule that would have needed one is refused elsewhere, loudly (ADR-0017 Decision 3)."""
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    f = plane_ransac(mask, depth, K)
    assert f.ok and f.half_extents_m[0] > 0.0
    assert 'radius_ratio' not in f.quality
    assert pose_quality(f) == pytest.approx(1.0), 'three gates, not four, and it says so'


def test_every_registered_method_runs_and_is_declared():
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    for name in METHODS:
        f = fit(name, mask, depth, K, face=FACE)
        assert f.ok, name
        assert name in GIVES_ORIENTATION, f'{name} does not declare whether it gives orientation'
    with pytest.raises(KeyError):
        fit('no_such_method', mask, depth, K)


def test_plane_nearest_finds_the_apex_of_a_DOME():
    """The method exists for a dome, where the nearest point is what a cup reaches first."""
    h, w = 224, 256
    v, u = np.mgrid[0:h, 0:w]
    z0, rad = 0.600, R_FACE
    x = (u - K[0, 2]) * z0 / K[0, 0]
    y = (v - K[1, 2]) * z0 / K[1, 1]
    rr = np.hypot(x, y)
    mask = rr <= rad
    depth = np.full((h, w), np.nan, np.float32)
    # a spherical cap of 15 mm rise, apex on the optical axis
    depth[mask] = (z0 - 0.015 * np.sqrt(np.clip(1.0 - (rr[mask] / rad) ** 2, 0, 1))).astype(
        np.float32)
    f = plane_nearest(mask, depth, K, face=FACE)
    assert f.ok
    assert abs(f.point[2] - (z0 - 0.015)) < 2e-3, 'the apex, not the rim'
    assert math.hypot(f.point[0], f.point[1]) < 0.25 * R_FACE


def test_plane_nearest_lands_on_the_RIM_of_a_flat_face_and_plane_ransac_does_not():
    """The trap, pinned deliberately: on a FLAT face the nearest point is on the rim, and a cup
    centred there overhangs the face and cannot seal. A flat face wants plane_ransac."""
    mask, depth, c, _ = synth_disc((0.0, 0.0, 0.600), (math.sin(math.radians(25)), 0.0,
                                                       -math.cos(math.radians(25))))
    near = plane_nearest(mask, depth, K, face=FACE)
    cen = plane_ransac(mask, depth, K, face=FACE)
    assert near.ok and cen.ok
    assert np.linalg.norm(np.array(near.point)[:2] - c[:2]) > 0.5 * R_FACE
    assert np.linalg.norm(np.array(cen.point) - c) < 2e-3


# ---- the approach: is there room for the tool? -----------------------------------------------

def test_a_clear_disc_obstructs_nothing():
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.60), (0.0, 0.0, -1.0))
    f = plane_ransac(mask, depth, K, face=FACE)
    assert approach_obstruction(f, depth, K, tool_radius_m=0.010) == 0.0
    assert approach_clearance(0.0, reach_mm=8.0) == 1.0


@pytest.mark.parametrize('rise_mm', [3.0, 6.0, 12.0])
def test_a_neighbour_standing_over_the_grasp_point_is_measured_to_the_millimetre(rise_mm):
    mask, depth, c, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    m2, d2, _, _ = synth_disc((0.012, 0.0, 0.600 - rise_mm * 1e-3), (0, 0, -1), seed=2)
    scene = np.where(np.isfinite(d2), d2, depth)
    f = plane_ransac(mask, depth, K, face=FACE)
    obs = approach_obstruction(f, scene, K, tool_radius_m=0.010)
    assert abs(obs - rise_mm) < 0.6, f'{rise_mm} mm neighbour measured as {obs:.2f} mm'
    assert approach_clearance(obs, reach_mm=8.0) == pytest.approx(
        max(0.0, 1.0 - rise_mm / 8.0), abs=0.08)


def test_the_surface_itself_is_never_its_own_obstruction():
    for tilt in (0, 20, 40, 60):
        t = math.radians(tilt)
        mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (math.sin(t), 0.0, -math.cos(t)))
        f = plane_ransac(mask, depth, K, face=FACE)
        obs = approach_obstruction(f, depth, K, tool_radius_m=0.010)
        assert obs < 1.5, f'{tilt} deg disc called itself an obstruction of {obs:.2f} mm'


def test_CLEAR_and_CANNOT_TELL_are_DIFFERENT_answers():
    """0 is clear, NaN is "the depth could not answer", and the clearance maps them to 1 and 0.
    Every bail-out returning 0 would mean the least information gave the most confident answer."""
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    f = plane_ransac(mask, depth, K, face=FACE)
    assert approach_obstruction(f, depth, K, tool_radius_m=0.0) == 0.0   # switched off, not unknown
    dead = plane_ransac(mask, np.full(mask.shape, np.nan, np.float32), K, face=FACE)
    assert not dead.ok
    assert math.isnan(approach_obstruction(dead, depth, K, tool_radius_m=0.010))
    assert approach_clearance(float('nan'), reach_mm=8.0) == 0.0         # and NOT 1.0
    blind = approach_obstruction(f, np.full(depth.shape, np.nan, np.float32), K,
                                 tool_radius_m=0.010)
    assert math.isnan(blind)


def test_depth_noise_alone_does_not_fabricate_an_obstruction():
    """A fixed 2 mm bar turns a millimetre of sensor noise into millimetres of obstruction that
    are not there. The bar is `noise_k` times the scatter THIS fit measured on THIS surface."""
    for seed in (1, 2, 3):
        mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1), noise_mm=1.0, seed=seed)
        f = plane_ransac(mask, depth, K, face=FACE)
        obs = approach_obstruction(f, depth, K, tool_radius_m=0.010)
        assert obs == 0.0, f'seed {seed}: noise alone reported {obs:.2f} mm of obstruction'


def test_a_real_obstruction_still_shows_through_the_same_noise():
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1), noise_mm=1.0, seed=1)
    m2, d2, _, _ = synth_disc((0.012, 0.0, 0.594), (0, 0, -1), noise_mm=1.0, seed=9)
    scene = np.where(np.isfinite(d2), d2, depth)
    f = plane_ransac(mask, depth, K, face=FACE)
    obs = approach_obstruction(f, scene, K, tool_radius_m=0.010)
    assert 4.0 < obs < 8.5, f'a 6 mm neighbour under 1 mm of noise read as {obs:.2f} mm'
    assert approach_clearance(obs, reach_mm=8.0) < 0.4


def test_a_TALL_obstruction_is_not_reported_as_clear():
    """No upper bound on the height, deliberately: filtering one would make the function fail OPEN
    and report a tall obstruction as clear."""
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    for rise_m in (0.030, 0.080, 0.200):
        m2, d2, _, _ = synth_disc((0.008, 0.0, 0.600 - rise_m), (0, 0, -1), seed=3)
        scene = np.where(np.isfinite(d2), d2, depth)
        f = plane_ransac(mask, depth, K, face=FACE)
        obs = approach_obstruction(f, scene, K, tool_radius_m=0.010)
        assert obs > 0.9 * rise_m * 1e3, f'{rise_m*1e3:.0f} mm read as {obs:.1f} mm'
        assert approach_clearance(obs, reach_mm=8.0) == 0.0


def test_a_neighbour_OUTSIDE_the_tool_radius_is_not_in_the_way():
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    m2, d2, _, _ = synth_disc((0.045, 0.0, 0.590), (0, 0, -1), seed=4)
    scene = np.where(np.isfinite(d2), d2, depth)
    f = plane_ransac(mask, depth, K, face=FACE)
    assert approach_obstruction(f, scene, K, tool_radius_m=0.010) == 0.0
    wide = approach_obstruction(f, scene, K, tool_radius_m=0.050)
    assert wide > 8.0, f'a 50 mm tool should meet the neighbour, measured {wide:.1f}'


def test_a_depth_image_with_a_trailing_axis_is_accepted():
    mask, depth, _, _ = synth_disc((0.0, 0.0, 0.600), (0, 0, -1))
    f = plane_ransac(mask, depth, K, face=FACE)
    assert approach_obstruction(f, depth[:, :, None], K, tool_radius_m=0.010) == 0.0
