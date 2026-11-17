"""SurfaceGrasp `centre_estimator: known_shape_fit` (TP 0.8.0, vNext M4.5d): the centre of a KNOWN disc from partial
depth evidence. Exact ray-cast scenes (discs, annuli, open cylinders, a background plane) whose true centre is known,
so every error is the estimator's. The owner's M4.5d §8 list, one test (or more) per row:

complete face; partial occlusion; asymmetric visible region; cavity / rim occlusion matching the capsule; noisy depth;
tilted disc; off-axis camera; lens distortion; insufficient evidence; wrong expected diameter; multiple nearby objects;
malformed / partial mask -- plus the configuration refusals, determinism and the diagnostics.
"""
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from surface_grasp_scenes import BASE, distorted, pinhole, unit  # noqa: E402

from trainit_perception.grasp import ConfigError, Region, SurfaceGraspConfig, propose  # noqa: E402
from trainit_perception.grasp import surface_grasp as SG  # noqa: E402
from trainit_perception.grasp.face_centre import (BY_CONSENSUS, BY_EDGES, KnownDiscParams,  # noqa: E402
                                                  arc_coverage_deg, fit_circle_known_radius, known_disc_centre)

R_IN = 0.0200           # a thin cup-shaped part: its cavity radius
R_OUT = 0.0203          # its outer radius (a 0.3 mm wall)


# =================================================================================================================
# an exact ray caster with the shapes a cup needs
# =================================================================================================================

@dataclass
class Disc:
    centre: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    radius: float
    r_in: float = 0.0                       # > 0: an annulus


@dataclass
class Tube:
    """An open cylinder: axis through `base` along unit `axis`, from height h0 to h1 above `base`."""
    base: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    radius: float
    h0: float
    h1: float


def cast(shapes: Sequence[object], camera, background_z: Optional[float] = 1.0):
    """-> (depth, owner index per pixel, -1 = background). The nearest hit wins; both roots of a tube count (a
    cup's inner wall is the far root)."""
    rays = camera.rays()
    d = np.concatenate([rays, np.ones(rays.shape[:2] + (1,))], -1)
    best = np.full(rays.shape[:2], np.inf)
    owner = np.full(rays.shape[:2], -1)
    for i, s in enumerate(shapes):
        if isinstance(s, Disc):
            n, c = unit(s.normal), np.asarray(s.centre, float)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (c @ n) / (d @ n)
            X = d * t[..., None]
            r = np.linalg.norm(X - c, axis=-1)
            hit = (r <= s.radius) & (r >= s.r_in) & np.isfinite(t) & (t > 0) & (t < best)
            best[hit], owner[hit] = t[hit], i
        else:
            a, b = unit(s.axis), np.asarray(s.base, float)
            # |(t d - b) - ((t d - b).a) a|^2 = r^2
            dp = d - (d @ a)[..., None] * a
            bp = b - (b @ a) * a
            A = np.einsum('ijk,ijk->ij', dp, dp)
            B = -2 * (dp @ bp)
            C = bp @ bp - s.radius ** 2
            disc = B * B - 4 * A * C
            sq = np.sqrt(np.where(disc >= 0, disc, 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                roots = ((-B - sq) / (2 * A), (-B + sq) / (2 * A))
            for t in roots:
                h = (d * t[..., None] - b) @ a
                hit = (disc >= 0) & (t > 0) & (h >= s.h0) & (h <= s.h1) & (t < best)
                best[hit], owner[hit] = t[hit], i
    if background_z is not None:
        bg = ~np.isfinite(best)
        best[bg] = background_z
    return np.where(np.isfinite(best), best, 0.0), owner


def cup(centre, axis, *, r_in=R_IN, r_out=R_OUT, depth=0.0075, lid=0.0008):
    """An UPSIDE-DOWN capsule: the cavity's ceiling (a disc of r_in) at `centre`, open toward `axis` (the camera's
    side), its wall from the ceiling up to the rim, the rim annulus, the outer wall down to the lid's underside.
    Shapes [ceiling, inner wall, rim, outer wall]."""
    a = unit(axis)
    c = np.asarray(centre, float)
    return [Disc(tuple(c), tuple(a), r_in),
            Tube(tuple(c), tuple(a), r_in, 0.0, depth),
            Disc(tuple(c + depth * a), tuple(a), r_out, r_in=r_in),
            Tube(tuple(c), tuple(a), r_out, -lid, depth)]


def cfg(**kw):
    p = dict(BASE)
    p.update(contact_radius_m=0.008, centre_estimator='known_shape_fit', expected_radius_m=R_IN)
    p.update(kw)
    return SurfaceGraspConfig.from_params(p)


def run(depth, mask, camera, **kw):
    return propose([Region(class_id='part', score=0.9, mask=mask)], depth, camera, cfg(**kw))


def err_mm(cand, centre):
    return float(np.linalg.norm(np.asarray(cand.point) - np.asarray(centre))) * 1e3


def centroid_err_mm(depth, mask, camera, centre, normal, **kw):
    """The 0.7.0 contact (the plane's inlier centroid) on the same scene, for comparison."""
    res = propose([Region('part', 0.9, mask)], depth, camera, cfg(centre_estimator='inlier_centroid', **kw))
    assert res.candidates, res.rejections
    return err_mm(res.candidates[0], centre)


TOWARD = (0.0, 0.0, -1.0)          # a face square to the camera faces -Z


# =================================================================================================================
# 1. complete face
# =================================================================================================================

def test_a_complete_disc_gives_its_centre_from_its_edges():
    cam = pinhole()
    c = (0.012, -0.008, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam)
    res = run(depth, own == 0, cam)
    assert len(res.candidates) == 1, res.rejections
    k = res.candidates[0]
    assert err_mm(k, c) < 0.2
    md = dict(k.metadata())
    assert md['centre_estimator'] == 'known_shape_fit' and md['centre_by'] == BY_EDGES


# =================================================================================================================
# 2. partial occlusion / 3. asymmetric visible region / 11. nearby objects
# =================================================================================================================

def test_a_disc_partly_hidden_by_an_object_in_front_keeps_its_centre():
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    occ = Disc((0.022, 0.0, 0.47), TOWARD, 0.015)              # 30 mm in front, over the +X third of the face
    depth, own = cast([Disc(c, TOWARD, R_IN), occ], cam)
    mask = own == 0
    res = run(depth, mask, cam)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.5
    assert centroid_err_mm(depth, mask, cam, c, TOWARD) > 2.0       # what the visible centroid did


def test_a_truncated_mask_does_not_move_the_centre():
    """A segmenter's mask SMALLER than the object (one side cut away): the depth there is still the face, the cut is
    not an edge (unknown evidence), the face's real edges and the known radius still give its centre."""
    cam = pinhole()
    c = (0.0, 0.01, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.6)
    mask = own == 0
    u0 = int(np.nonzero(mask.any(0))[0].max()) - 5            # the last 5 columns of the face removed from the mask
    mask[:, u0:] = False                                      # (the cup's contact disc stays on what remains)
    res = run(depth, mask, cam)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.5
    assert centroid_err_mm(depth, mask, cam, c, TOWARD) > 1.5


def test_a_mask_spilling_onto_a_coplanar_neighbour_does_not_move_the_centre():
    """M4.5d, measured with YOLO masks on the capsule bin: the mask covered part of a NEIGHBOUR lying in the same
    plane. The neighbour is a separate support component, outside the expected disc: it never joins the face."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    nb = (0.05, 0.0, 0.5)                                     # touching neighbour, same plane, a 1 mm gap ring
    gap = Disc((0.05, 0.0, 0.493), TOWARD, R_IN + 0.004, r_in=R_IN)   # its raised rim separates the two in depth
    depth, own = cast([Disc(c, TOWARD, R_IN), Disc(nb, TOWARD, R_IN), gap], cam, background_z=0.6)
    mask = own == 0
    spill = own == 1
    spill[:, int(np.nonzero(spill.any(0))[0].min()) + 10:] = False      # 10 columns of the neighbour
    # the plane's own extent includes the spill: the M4 pose-quality extent gate would refuse it first (an
    # independent check, unchanged by the estimator), so it is opened here to reach the estimator
    res = run(depth, mask | spill, cam, pose_quality_radius_tol=1.0)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.5
    assert centroid_err_mm(depth, mask | spill, cam, c, TOWARD, pose_quality_radius_tol=1.0) > 2.0


def test_a_face_split_by_an_occluder_across_it_counts_in_full():
    """A thin object lying ACROSS the face (a neighbour's tab) splits it into two components; both lie inside the
    expected disc, so both are the face."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    bar = [Disc((x, 0.013, 0.49), TOWARD, 0.0035) for x in np.linspace(-0.03, 0.03, 25)]   # clear of the cup's disc
    depth, own = cast([Disc(c, TOWARD, R_IN)] + bar, cam)
    mask = own == 0
    res = run(depth, mask, cam)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.5
    rec = SG.diagnostics(res, 'cam', 0, 0)
    assert rec['candidates'][0]['centre_estimator'] == 'known_shape_fit'


# =================================================================================================================
# 4. cavity / rim occlusion (the upside-down capsule) and 7. off-axis view
# =================================================================================================================

@pytest.mark.parametrize('x, bound_mm', [(0.0, 0.6), (0.12, 0.6), (0.2, 1.6)])
def test_an_upside_down_capsule_cavity_seen_off_axis(x, bound_mm):
    """The ceiling of a 7.5 mm deep cavity, seen from up to 22 deg off its axis: the rim hides the camera-side of
    the ceiling and the floor around the capsule lies 0.8 mm below it (coplanar within the band). The capsule's
    silhouette is the mask (as a segmenter gives it). The BOUND is physics, not tuning: when the rim hides a
    strip s = depth x tan(view) of the ceiling (3.0 mm at 22 deg) and the far wall gives too few true-edge pixels at
    this resolution (1.7 mm / px), the consensus disc can slide s and its mean sits s / 2 off -- the documented limit
    of the consensus, which the edge refinement removes when the far wall is resolved."""
    cam = pinhole()
    c = np.array([x, 0.03, 0.5])
    shapes = cup(c, TOWARD)
    depth, own = cast(shapes, cam, background_z=0.5008)
    mask = own >= 0
    res = run(depth, mask, cam, expected_radius_m=R_IN, inlier_mm=2.0, contact_radius_m=0.006)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < bound_mm


# =================================================================================================================
# 5. noisy depth / 6. tilted / 8. distortion
# =================================================================================================================

def test_noisy_depth():
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.56)
    noisy = depth + np.random.default_rng(7).normal(0.0, 0.0005, depth.shape)
    res = run(noisy, own == 0, cam, inlier_mm=2.0)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 1.0


@pytest.mark.parametrize('deg', [15.0, 30.0])
def test_a_tilted_disc(deg):
    cam = pinhole()
    t = math.radians(deg)
    n = (math.sin(t), 0.0, -math.cos(t))
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, n, R_IN)], cam, background_z=0.6)
    res = run(depth, own == 0, cam)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.5


def test_a_distorted_lens():
    cam = distorted()
    c = (0.06, 0.04, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.6)
    res = run(depth, own == 0, cam)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.5


# =================================================================================================================
# 9. insufficient evidence / 10. wrong expected diameter / 12. malformed or partial masks -> FAIL CLOSED
# =================================================================================================================

def test_too_little_of_the_face_is_refused_not_guessed():
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.6)
    mask = own == 0
    u = np.nonzero(mask.any(0))[0]
    mask[:, u.min() + 7:] = False                              # a sliver: ~10 % of the disc
    res = run(depth, mask, cam, min_points=10, contact_radius_m=0.002)
    assert not res.candidates
    assert res.rejections[0].code in (SG.FACE_FIT_FAILED, SG.SURFACE_TOO_SMALL, SG.CONTACT_UNSUPPORTED)
    # and the estimator itself, on the same evidence, refuses by name
    fc = known_disc_centre(mask, depth, cam.rays(), np.array(c), np.array([0, 0, -1.0]),
                           KnownDiscParams(radius_m=R_IN, band_m=0.002))
    assert not fc.ok and ('visible' in fc.why or 'slide' in fc.why)


def test_an_expected_diameter_far_too_large_is_refused():
    cam = pinhole()
    depth, own = cast([Disc((0, 0, 0.5), TOWARD, R_IN)], cam, background_z=0.6)
    res = run(depth, own == 0, cam, expected_radius_m=2.5 * R_IN)
    assert not res.candidates and res.rejections[0].code == SG.FACE_FIT_FAILED
    assert 'of the expected disc' in res.rejections[0].why


def test_an_expected_diameter_far_too_small_is_refused():
    cam = pinhole()
    depth, own = cast([Disc((0, 0, 0.5), TOWARD, R_IN)], cam, background_z=0.6)
    res = run(depth, own == 0, cam, expected_radius_m=0.4 * R_IN)
    assert not res.candidates and res.rejections[0].code == SG.FACE_FIT_FAILED


def test_a_mask_with_holes_and_specks():
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.6)
    mask = own == 0
    rng = np.random.default_rng(3)
    holes = rng.random(mask.shape) < 0.08
    specks = (rng.random(mask.shape) < 0.002) & ~mask
    res = run(depth, (mask & ~holes) | specks, cam)
    assert res.candidates, res.rejections
    assert err_mm(res.candidates[0], c) < 0.7


def test_a_mask_without_depth_is_refused_by_the_plane_checks_first():
    cam = pinhole()
    depth, own = cast([Disc((0, 0, 0.5), TOWARD, R_IN)], cam)
    res = run(np.zeros_like(depth), own == 0, cam)
    assert not res.candidates and res.rejections[0].code == SG.INSUFFICIENT_DEPTH


# =================================================================================================================
# configuration, default behaviour, determinism, helpers
# =================================================================================================================

def test_the_default_estimator_is_unchanged():
    c = SurfaceGraspConfig.from_params(dict(BASE))
    assert c.centre_estimator == 'inlier_centroid' and c.known_disc is None


@pytest.mark.parametrize('extra, says', [
    ({'centre_estimator': 'magic'}, 'centre_estimator'),
    ({'centre_estimator': 'known_shape_fit'}, 'expected_radius_m'),
    ({'centre_estimator': 'known_shape_fit', 'grasp_face': 'rect', 'expected_half_a_m': 0.01,
      'expected_half_b_m': 0.01}, 'fits a DISC'),
    ({'centre_estimator': 'known_shape_fit', 'expected_radius_m': R_IN, 'face_occlusion_mm': 1.0}, 'must exceed'),
])
def test_a_known_shape_fit_it_cannot_run_is_refused(extra, says):
    with pytest.raises(ConfigError, match=says):
        SurfaceGraspConfig.from_params({**BASE, **extra})


def test_identical_input_gives_an_identical_answer():
    cam = pinhole()
    depth, own = cast(cup((0.1, 0.03, 0.5), TOWARD), cam, background_z=0.5008)
    a = run(depth, own >= 0, cam, contact_radius_m=0.006)
    b = run(depth, own >= 0, cam, contact_radius_m=0.006)
    assert a.candidates and a.candidates[0].point == b.candidates[0].point


def test_the_circle_helpers():
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    q = np.c_[0.003 + 0.02 * np.cos(th), -0.001 + 0.02 * np.sin(th)]
    assert np.allclose(fit_circle_known_radius(q, 0.02, np.zeros(2)), [0.003, -0.001], atol=1e-9)
    assert arc_coverage_deg(q[:10], np.array([0.003, -0.001])) == pytest.approx(81.0, abs=0.1)


def test_the_consensus_is_used_and_reported_when_the_edges_cannot_decide():
    """A face whose true edges are all hidden (an occluding ring in front of its whole border) has no edge
    evidence: the consensus disc decides, and says so."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    ring = Disc((0.0, 0.0, 0.49), TOWARD, R_IN + 0.006, r_in=R_IN - 0.0015)
    depth, own = cast([Disc(c, TOWARD, R_IN), ring], cam)
    fc = known_disc_centre(own == 0, depth, cam.rays(), np.array(c), np.array([0, 0, -1.0]),
                           KnownDiscParams(radius_m=R_IN, band_m=0.002))
    assert fc.ok and fc.by == BY_CONSENSUS
    assert float(np.linalg.norm(np.asarray(fc.centre) - c)) < 0.002


# =================================================================================================================
# M4.5d independent review (geometry): the regressions it found, and the sweeps it asked for
# =================================================================================================================

def test_a_coplanar_blob_as_large_as_the_face_is_refused_not_answered():
    """Review BLOCKER: a disc and a coplanar 40 mm square touching it, both inside one mask. The square holds a disc
    of the expected radius as well as the face does: the answer is ambiguous and must be REFUSED (before the fix the
    consensus chose the square, 39 mm off, with every other gate passing)."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.5)        # the floor is the face's plane
    rays = cam.rays()
    X, Y = rays[..., 0] * 0.5, rays[..., 1] * 0.5
    square = (X > R_IN - 0.001) & (X < R_IN + 0.039) & (np.abs(Y) < 0.02)
    fc = known_disc_centre((own == 0) | square, depth, rays, np.array(c), np.array([0, 0, -1.0]),
                           KnownDiscParams(radius_m=R_IN, band_m=0.002))
    assert not fc.ok and 'second disc-sized face' in fc.why
    assert fc.quality['face_rival'] >= 0.6
    res = run(depth, (own == 0) | square, cam, pose_quality_radius_tol=1.0)
    assert not res.candidates and res.rejections[0].code == SG.FACE_FIT_FAILED


def test_a_small_coplanar_blob_is_not_a_rival():
    """The rival rule refuses only a blob that could BE the face: a coplanar patch a quarter of the disc's area
    beside it leaves the answer on the disc."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.5)
    rays = cam.rays()
    X, Y = rays[..., 0] * 0.5, rays[..., 1] * 0.5
    patch = (X > R_IN - 0.001) & (X < R_IN + 0.017) & (np.abs(Y) < 0.006)
    fc = known_disc_centre((own == 0) | patch, depth, rays, np.array(c), np.array([0, 0, -1.0]),
                           KnownDiscParams(radius_m=R_IN, band_m=0.002))
    assert fc.ok, fc.why
    # the floor is coplanar: no true edge anywhere, the consensus decides. Its resolution is its definition: the
    # discs of radius R + tol (tol 1 mm) holding the most points -- an attached coplanar patch may draw it by up to
    # tol, plus its plateau, plus one grid step (0.5 mm). A cell with real edges is decided by the edges instead.
    assert fc.by == BY_CONSENSUS
    err = float(np.linalg.norm(np.asarray(fc.centre) - c)) * 1e3
    assert err <= fc.quality['face_plateau_mm'] + 1.0 + 0.5
    assert fc.quality['face_rival'] < 0.6


@pytest.mark.parametrize('factor', [0.85, 0.9, 0.95, 0.97, 1.03, 1.05, 1.1, 1.15])
@pytest.mark.parametrize('occluded', [False, True])
def test_a_mis_measured_expected_radius_is_bounded_or_refused(factor, occluded):
    """The realistic mis-calibration regime (+-3..15 %): either FACE_FIT_FAILED, or a centre whose error is bounded
    by the radius error itself (a known-radius fit to an arc of >= 150 deg moves by <= ~1.04 |dR|) + 0.5 mm."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    shapes = [Disc(c, TOWARD, R_IN)] + ([Disc((0.022, 0.0, 0.47), TOWARD, 0.015)] if occluded else [])
    depth, own = cast(shapes, cam, background_z=0.6)
    res = run(depth, own == 0, cam, expected_radius_m=factor * R_IN)
    if res.candidates:
        assert err_mm(res.candidates[0], c) <= 1.1 * abs(factor - 1.0) * R_IN * 1e3 + 0.5
    else:
        assert res.rejections[0].code == SG.FACE_FIT_FAILED


@pytest.mark.parametrize('x_occ', [0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01])
def test_the_visible_arc_boundary(x_occ):
    """An occluder sliding over the face: while enough of it is visible the centre stays exact; below
    face_min_coverage it is refused by name -- never a guess in between."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN), Disc((x_occ, 0.0, 0.47), TOWARD, 0.02)], cam, background_z=0.6)
    fc = known_disc_centre(own == 0, depth, cam.rays(), np.array(c), np.array([0, 0, -1.0]),
                           KnownDiscParams(radius_m=R_IN, band_m=0.002))
    if fc.ok:
        assert float(np.linalg.norm(np.asarray(fc.centre) - c)) < 0.0005
    else:
        assert fc.quality['face_coverage'] < 0.35 or 'slide' in fc.why
    if x_occ >= 0.03:                       # >= ~270 deg of the edge visible: must be answered
        assert fc.ok, fc.why


@pytest.mark.parametrize('sigma, inlier_mm, occlusion_mm', [(0.0006, 2.0, 3.0), (0.001, 3.0, 4.5)])
def test_realistic_depth_noise_over_seeds(sigma, inlier_mm, occlusion_mm):
    """Depth noise of 0.6 / 1.0 mm over 20 seeds: noise holes inside the face are not its edges (they are filled
    before the edges are classified); every answer stays within 1 mm."""
    cam = pinhole()
    c = (0.0, 0.0, 0.5)
    depth, own = cast([Disc(c, TOWARD, R_IN)], cam, background_z=0.56)
    errs = []
    for seed in range(20):
        noisy = depth + np.random.default_rng(seed).normal(0.0, sigma, depth.shape)
        res = run(noisy, own == 0, cam, inlier_mm=inlier_mm, face_occlusion_mm=occlusion_mm)
        if res.candidates:
            errs.append(err_mm(res.candidates[0], c))
    assert len(errs) >= 18
    assert max(errs) < 1.0


def test_the_consensus_re_settles_a_far_start_and_refuses_a_support_that_never_settles():
    from trainit_perception.grasp.face_centre import _consensus
    prm = KnownDiscParams(radius_m=0.02, band_m=0.002)
    g = np.arange(-0.02, 0.02 + 1e-9, 0.0005)
    P = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
    disc = P[np.linalg.norm(P, axis=1) <= 0.02] + [0.004, -0.002]
    c, _ext, _best, _pl, settled = _consensus(disc, np.array([0.004 + 0.018, -0.002]), 0.02, prm)   # 0.9 R away
    assert settled and float(np.linalg.norm(c - [0.004, -0.002])) < 0.001
    strip = np.stack([np.linspace(-0.2, 0.2, 4000), np.zeros(4000)], 1)          # a line far longer than the disc
    assert not _consensus(strip, np.zeros(2), 0.02, prm)[4]


def test_the_cost_is_bounded_at_high_resolution():
    """Review MAJOR: a 1280x720 camera with f = 1400 px at 0.2 m (65k support points) once took 1.3 GB and ~8 s."""
    import time
    import tracemalloc
    from trainit_perception.grasp import CameraModel
    K2 = np.array([[1400.0, 0.0, 640.0], [0.0, 1400.0, 360.0], [0.0, 0.0, 1.0]])
    cam = CameraModel.pinhole(K2, 1280, 720, 'cam_optical')
    depth, own = cast([Disc((0.0, 0.0, 0.2), TOWARD, R_IN)], cam, background_z=0.3)
    rays = cam.rays()
    tracemalloc.start()
    t0 = time.perf_counter()
    fc = known_disc_centre(own == 0, depth, rays, np.array([0.0, 0.0, 0.2]), np.array([0, 0, -1.0]),
                           KnownDiscParams(radius_m=R_IN, band_m=0.002))
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert fc.ok and float(np.linalg.norm(np.asarray(fc.centre) - [0.0, 0.0, 0.2])) < 0.0003
    assert peak < 200e6 and elapsed < 5.0


def test_the_disc_parameters_validate_themselves():
    with pytest.raises(ValueError, match='must exceed'):
        KnownDiscParams(radius_m=R_IN, band_m=0.002, occlusion_m=0.001)
    with pytest.raises(ValueError):
        KnownDiscParams(radius_m=0.0, band_m=0.002)
