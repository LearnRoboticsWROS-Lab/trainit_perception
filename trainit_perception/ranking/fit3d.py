"""Stage 3D: from a mask to a grasp point, and to how much to trust it.

Pure functions. Each takes a boolean mask, a depth image in metres and the camera intrinsics, and
returns a `GraspFit`: where to put the tool, which way the surface faces, and the terms that say
how well formed that answer is. No ROS, no robot, and -- since ADR-0017 Decision 2 -- no shape
assumption either: what the face looks like is a `GraspFace` the caller passes in, so a detector
that only works on 41,9 mm capsules is not a detector, it is a capsule finder.

WHY THE QUALITY TERMS ARE PART OF THE OUTPUT and not an afterthought. The cell's question is not
"where is it" but "should I try this one", and that is a probability with three factors, each
owned by the layer that can see it:

    P(candidate) = P(detected)          the detector's own score
                 x P(pose well formed)  THIS FILE: inliers, residual, depth coverage, fitted size
                 x P(reachable)         the motion layer: IK, collision, the tool's grip window

Only the third needs a robot, which is why it is not here: the perception contract is explicit
that a detector reports in the camera's optical frame and knows nothing about robots. One PART of
it does live here, because only this layer has the depth image to compute it from:
`approach_obstruction` says how far something else stands in the way of the tool. It takes the
TOOL's radius as a configured length -- a length, not a robot -- and it is reported as a term of
its own rather than folded into `pose_quality`.

FRAME. Everything is in the camera OPTICAL frame: +X right, +Y down, +Z forward. The normal is
returned pointing BACK AT THE CAMERA (its Z component is negative), because that is the direction
a tool has to come from, and a plane fit has no opinion about which of its two normals is meant.

MEASURED (a private bin-picking validation set, 72 frames, 1 073 matched detections): centre error p50 2,40 mm against a
20,9 mm face, normal error p50 0,7 deg, 7,7 ms per frame on CPU. The quality this file predicts
separates a 9,38 mm centre error in its worst quintile from 1,68 mm in its best (ADR-0014).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from .faces import (DiscFace, GraspFace, all_face_params, extent_terms, mad_extent, p90_extent)


@dataclass
class GraspFit:
    """Where to grasp, which way it faces, and how much the fit is worth.

    `half_extents_m` is TWO numbers, along the fit's own principal in-plane axes, because the
    plane fit computes two and throwing one away was a disc assumption living in the output type
    (ADR-0017 Decision 2). A disc face returns them equal; a rectangular face would not.
    """
    point: Tuple[float, float, float]              # m, camera optical frame
    normal: Tuple[float, float, float]             # unit, pointing back at the camera
    half_extents_m: Tuple[float, float] = (0.0, 0.0)   # m, in-plane; (0, 0) = not estimated
    quality: Dict[str, float] = field(default_factory=dict)
    ok: bool = True
    why: str = ''

    @property
    def tilt_deg(self) -> float:
        """Angle between the surface normal and the camera's axis. 0 = facing the lens square on."""
        return math.degrees(math.acos(min(1.0, abs(self.normal[2]))))


def _refuse_face_params(params: Mapping) -> None:
    """A face parameter passed to a FIT is a silent no-op, so it is refused instead.

    Before ADR-0017 the fit took `expected_radius_m` itself and dropped the `radius_ratio` gate
    when it was unset -- the exact silent degeneration Decision 3 closes. Leaving the key
    acceptable-and-ignored here would reopen it one call site further out.
    """
    stray = sorted(set(params) & set(all_face_params()))
    if stray:
        raise TypeError(
            f'{", ".join(stray)} is a GRASP FACE parameter, not a fit parameter: pass '
            f'face=make_face("disc", params) instead. The fit measures an extent; what that '
            f'extent is SUPPOSED to be belongs to the face (ADR-0017 Decision 2).')


def _face(face: Optional[GraspFace]) -> GraspFace:
    """A fit always has a face; a caller that named none gets a disc of unknown size, which
    measures the extent and claims nothing about it."""
    return face if face is not None else DiscFace()


def _points(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
            z_min: float, z_max: float) -> Tuple[np.ndarray, float]:
    """Back-project the valid depth inside the mask. Returns (N x 3 points, depth coverage)."""
    v, u = np.nonzero(mask)
    if not len(u):
        return np.zeros((0, 3)), 0.0
    z = depth_m[v, u].astype(np.float64)
    good = np.isfinite(z) & (z > z_min) & (z < z_max)
    coverage = float(good.mean())
    u, v, z = u[good], v[good], z[good]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1), coverage


def _toward_camera(n: np.ndarray) -> np.ndarray:
    """A plane's normal is defined up to sign; the one a tool approaches along points at the
    lens."""
    return -n if n[2] > 0 else n


def centroid(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray, *,
             face: Optional[GraspFace] = None, z_min: float = 0.05,
             z_max: float = 3.0, min_points: int = 30, **_) -> GraspFit:
    """The simplest method: the centre of the masked surface, facing the camera.

    It makes no claim about orientation -- the normal is the camera axis -- so it suits objects
    that are always presented the same way up, and it is the honest fallback when the depth inside
    a mask is too sparse to fit anything. `tilt` is reported as 0 and the caller must not read a
    face orientation into it; `gives_orientation` is False for this method.
    """
    _refuse_face_params(_)
    f = _face(face)
    P, cov = _points(mask, depth_m, K, z_min, z_max)
    # three points propose a plane, so three is the floor whatever `min_points` is set to:
    # below it numpy's own sampler raises `Cannot take a larger sample than population`
    if len(P) < max(min_points, 3):
        return GraspFit((0, 0, 0), (0, 0, -1), ok=False,
                        why=f'only {len(P)} valid depth points in the mask (need {min_points})',
                        quality=dict(depth_coverage=cov, n_points=float(len(P))))
    c = np.median(P, 0)                                   # median, not mean: holes bias the mean
    # The extent, measured in the IMAGE plane because this method claims no surface orientation.
    # The FACE collapses the offsets (a disc does it radially); the estimator is this method's
    # choice, and it is the MAD because a median survives the holes a depth image is full of.
    he = f.half_extents(P[:, :2] - c[:2], mad_extent)
    q = dict(depth_coverage=cov, n_points=float(len(P)),
             depth_spread_mm=float(np.std(P[:, 2]) * 1e3))
    out = GraspFit(tuple(c), (0.0, 0.0, -1.0), half_extents_m=he, quality=q)
    out.quality.update(extent_terms(f, out))
    return out


def plane_ransac(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray, *,
                 face: Optional[GraspFace] = None, z_min: float = 0.05, z_max: float = 3.0,
                 min_points: int = 50, iters: int = 120, inlier_mm: float = 2.0,
                 seed: int = 0, **_) -> GraspFit:
    """Fit a plane to the masked surface: grasp point on the plane, normal from the plane.

    RANSAC rather than a plain least-squares fit because a mask always leaks: a few pixels of the
    neighbour behind, or of the bin floor seen through a gap, drag a least-squares plane by
    degrees. Three random points propose a plane, the best-supported one wins, and the final
    normal is re-estimated from ALL its inliers by SVD -- the proposal only has to be close.

    The point returned is the inlier centroid projected onto the fitted plane, which is the centre
    of the surface rather than the centre of the pixels that happened to have depth.
    """
    _refuse_face_params(_)
    f = _face(face)
    P, cov = _points(mask, depth_m, K, z_min, z_max)
    if len(P) < min_points:
        return GraspFit((0, 0, 0), (0, 0, -1), ok=False,
                        why=f'only {len(P)} valid depth points in the mask (need {min_points})',
                        quality=dict(depth_coverage=cov, n_points=float(len(P))))

    rng = np.random.default_rng(seed)
    thr = inlier_mm * 1e-3
    best_n, best_in = None, np.zeros(len(P), bool)
    for _i in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n = n / ln
        d = float(n @ a)
        inl = np.abs(P @ n - d) < thr
        if inl.sum() > best_in.sum():
            best_n, best_in = n, inl
    if best_n is None or best_in.sum() < min_points:
        return GraspFit((0, 0, 0), (0, 0, -1), ok=False,
                        why='no plane supported by enough points',
                        quality=dict(depth_coverage=cov, n_points=float(len(P)),
                                     inlier_ratio=float(best_in.mean()) if len(P) else 0.0))

    # refit on every inlier: the 3-point proposal fixes the support, SVD fixes the normal
    Q = P[best_in]
    c = Q.mean(0)
    _u, _s, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = _toward_camera(Vt[2] / np.linalg.norm(Vt[2]))
    resid = np.abs((Q - c) @ n)
    # The extent measured IN THE PLANE, which is the face's own size however it is tilted. The two
    # principal in-plane axes are handed to the FACE: a disc collapses them radially and returns
    # one number twice, a rectangular face would keep them apart. Before ADR-0017 this line did
    # the collapsing itself (`np.hypot`) and the second half-extent was computed and discarded.
    e1, e2 = Vt[0], Vt[1]
    he = f.half_extents(np.stack([(Q - c) @ e1, (Q - c) @ e2], 1), p90_extent)

    q = dict(depth_coverage=cov, n_points=float(len(P)),
             inlier_ratio=float(best_in.mean()),
             rms_mm=float(np.sqrt((resid ** 2).mean()) * 1e3),
             tilt_deg=math.degrees(math.acos(min(1.0, abs(n[2])))))
    out = GraspFit(tuple(c), tuple(n), half_extents_m=he, quality=q)
    out.quality.update(extent_terms(f, out))
    return out


def _pick_along_normal(fit: GraspFit, mask, depth_m, K, which: str,
                       z_min: float, z_max: float) -> GraspFit:
    """The masked point nearest to, or farthest from, the camera. Ranked by DEPTH, deliberately.

    RANKING BY `P . n` LOOKS RIGHT AND IS WRONG TWICE OVER, and it is worth saying why because the
    expression is the obvious one to write. `n` is the SVD normal of those same points, so on a
    planar fit `P . n` is constant BY CONSTRUCTION: measured on a 21 mm disc at 24 deg of tilt, the
    key spans 42 nanometres while the real depth spans 16,4 mm. The ordering is then the residual
    of the fit, a near-linear function whose extremes over a disc always sit on its boundary -- so
    the point comes back on the face's RIM, a median 0,83 of the radius from the centre over 30
    synthetic scenes, and a 10 mm cup placed there overhangs a 20,95 mm face by 6 to 10 mm and
    cannot seal. On top of that the sign is inverted: `_toward_camera` makes n[2] <= 0, so `P . n`
    is LARGEST nearest the lens and `argmin` asks for the farthest.

    Depth is the key that means what the name says: +Z is forward in the optical frame, so the
    nearest point is the smallest z. It is non-degenerate on any surface.

    WHAT THIS METHOD IS AND IS NOT FOR. On a DOME the nearest point is the apex, which is what a
    cup reaches first and why the method exists. On a FLAT FACE the nearest point is a point of the
    rim, and a cup centred there overhangs the face. A flat face wants `plane_ransac`, which
    grasps the centre of the fitted surface.
    """
    P, _ = _points(mask, depth_m, K, z_min, z_max)
    if not len(P) or not fit.ok:
        return fit
    i = int(np.argmin(P[:, 2])) if which == 'nearest' else int(np.argmax(P[:, 2]))
    pt = P[i] if which in ('nearest', 'farthest') else np.median(P, 0)
    return GraspFit(tuple(pt), fit.normal, fit.half_extents_m,
                    dict(fit.quality, point_rule=1.0), fit.ok, fit.why)


def plane_nearest(mask, depth_m, K, **kw) -> GraspFit:
    """Plane fit for the orientation, but grasp the point of the surface CLOSEST to the camera.

    FOR A DOME, where that point is the apex and a cup reaches it first. NOT for a flat face:
    there the nearest point is on the RIM, and a cup centred on the rim of a 41,9 mm cap overhangs
    it and cannot seal. A flat face wants `plane_ransac`, which grasps the centre of the fitted
    surface. See the note in `_pick_along_normal`.
    """
    f = plane_ransac(mask, depth_m, K, **kw)
    return _pick_along_normal(f, mask, depth_m, K, 'nearest',
                              kw.get('z_min', 0.05), kw.get('z_max', 3.0))


# The registry. A method is a name in a configuration file and a pure function here -- the same
# shape as `detectors/registry.py` and `faces.FACES`. `gives_orientation` is what a configurator
# greys out "align to the detected object" on.
METHODS: Dict[str, Callable[..., GraspFit]] = {
    'centroid': centroid,
    'plane_ransac': plane_ransac,
    'plane_nearest': plane_nearest,
}

GIVES_ORIENTATION: Dict[str, bool] = {
    'centroid': False,
    'plane_ransac': True,
    'plane_nearest': True,
}


def available_methods_3d() -> list:
    return sorted(METHODS)


def fit(method: str, mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray, **params) -> GraspFit:
    if method not in METHODS:
        raise KeyError(f'unknown 3D method {method!r} (have: {", ".join(sorted(METHODS))})')
    _refuse_face_params(params)
    return METHODS[method](mask, depth_m, K, **params)


def approach_obstruction(fit: GraspFit, depth_m: np.ndarray, K: np.ndarray, *,
                         tool_radius_m: float, inlier_mm: float = 2.0, max_h_m: float = 0.050,
                         min_pixels: int = 8, pct: float = 95.0, noise_k: float = 3.0,
                         z_min: float = 0.05, z_max: float = 3.0) -> float:
    """How far anything NOT on this surface stands into the tool's path, in mm.

    0 is CLEAR and **NaN is "cannot tell"** -- they are different answers and the difference
    matters, because `approach_clearance` turns the first into 1,0 and the second into 0,0. Every
    bail-out here returning 0 would mean the least information produced the most confident answer,
    in the one direction a clearance test must never err in.

    A point is in the way if it lies within `tool_radius_m` of the GRASP AXIS and more than the
    fit's own tolerance ABOVE the grasp plane -- on the camera's side, where the tool comes from.
    Points on the surface itself are within tolerance by construction, which is why no mask is
    needed. NOTE the condition on that: it holds only when `fit.normal` really is the surface's
    normal. `centroid()` returns the camera axis for every fit and declares
    `gives_orientation = False`, so on a tilted object its "plane" cuts through the surface and
    the object reports itself as its own obstruction. Call this with a fit from a method that
    gives orientation.

    WHY IT IS HERE BUT NOT IN `pose_quality`. The number depends on the tool: `tool_radius_m` is a
    configured length, the same kind of parameter as the face's own radius, so nothing here learns
    that there is a robot -- but it is not a property of the POSE, and folding it into
    pose_quality would blur the second factor of the candidate probability into the third. It is
    published as a term of its own and the layer that owns the tool multiplies it in.

    MEASURED on a real suction cup, read off its collision mesh: the mouth is a stem of 9,2 mm radius
    for its last 30 mm and the 42,5 mm body starts 37 mm further back, so within a 48 mm bin only
    the stem can be fouled -- `tool_radius_m` is 0,010, not 0,043.
    """
    if tool_radius_m <= 0:
        return 0.0                                  # the term is switched off, not unknown
    if not fit.ok:
        return float('nan')
    d = np.asarray(depth_m)
    if d.ndim == 3 and d.shape[2] == 1:             # some cv_bridge conversions yield (H, W, 1)
        d = d[:, :, 0]
    if d.ndim != 2:
        raise ValueError(f'depth must be a 2-D image, got shape {np.asarray(depth_m).shape}')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p0 = np.asarray(fit.point, float)
    n = np.asarray(fit.normal, float)
    nn = float(np.linalg.norm(n))
    if not np.isfinite(p0).all() or p0[2] <= 0 or not np.isfinite(nn) or nn <= 0:
        return float('nan')
    n = n / nn

    # ONLY THE PIXELS THAT COULD POSSIBLY BE IN THE WAY. `max_h_m` bounds how far along the grasp
    # axis the window reaches: the cylinder of radius `tool_radius_m` about that axis, taken that
    # far toward the camera. The bound is a WINDOW SIZE, not a filter -- see the mask below. Two
    # terms set the window's radius and BOTH are needed: the cylinder's own lateral size, which
    # grows as 1/Z toward the near plane, and the PARALLAX of the axis itself, because a grasp off
    # the optical centre projects to a different pixel at a different depth. Omitting the second
    # makes the bound wrong exactly where the grasp is off-axis, which is most of a 256 px ROI.
    # For a camera 600 mm over a 48 mm bin, 50 mm puts the near plane at 530 mm -- already nearer
    # than the D455's own 520 mm min-Z, so nothing the bin can hold falls outside it. Searching
    # the whole frame instead costs 8,7 ms per detection, 234 ms for the 27 a frame of this bin
    # yields, against a detector that otherwise runs in 7,7 ms. MEASURED.
    tilt = math.sqrt(max(0.0, 1.0 - min(1.0, n[2] ** 2))) / max(abs(n[2]), 1e-6)
    extent = tool_radius_m + max_h_m * tilt
    z_near = max(p0[2] - max_h_m, 1e-3)
    par = abs(1.0 / z_near - 1.0 / p0[2])
    ru = int(np.ceil(fx * (extent / z_near + abs(p0[0]) * par))) + 2
    rv = int(np.ceil(fy * (extent / z_near + abs(p0[1]) * par))) + 2
    uc, vc = int(round(fx * p0[0] / p0[2] + cx)), int(round(fy * p0[1] / p0[2] + cy))
    h_img, w_img = d.shape[:2]
    u0, u1 = max(0, uc - ru), min(w_img, uc + ru + 1)
    v0, v1 = max(0, vc - rv), min(h_img, vc + rv + 1)
    if u0 >= u1 or v0 >= v1:
        return float('nan')                         # the grasp projects off the image: no idea

    z = d[v0:v1, u0:u1].astype(np.float64)
    good = np.isfinite(z) & (z > z_min) & (z < z_max)
    if not good.any():
        return float('nan')                         # nothing to look at: no idea
    vv, uu = np.nonzero(good)
    zz = z[vv, uu]
    dx = (uu + u0 - cx) * zz / fx - p0[0]
    dy = (vv + v0 - cy) * zz / fy - p0[1]
    dz = zz - p0[2]
    h = dx * n[0] + dy * n[1] + dz * n[2]           # > 0: on the camera's side of the grasp plane
    # perpendicular distance to the axis, without forming the difference vectors:
    # |d_perp|^2 = |d|^2 - h^2
    perp2 = dx * dx + dy * dy + dz * dz - h * h
    # NO UPPER BOUND ON h, deliberately: filtering one would make the function fail OPEN and
    # report a tall obstruction as clear.
    # AN OBSTRUCTION IS A SURFACE, NOT A PIXEL, AND THE THRESHOLD IS THE SENSOR'S OWN NOISE.
    # `max()` over the ~157 pixels a 10 mm cylinder covers at 0,6 m is an extreme-value statistic:
    # against a fixed 2 mm tolerance, a millimetre of depth noise puts ~4 pixels over it on every
    # frame, and a clear face then reports millimetres of obstruction that are not there
    # (MEASURED: 2,5 mm on a perfectly clear synthetic disc under 1 mm of noise). So the bar is
    # `noise_k` times the scatter THIS FIT measured on THIS surface, never below `inlier_mm` --
    # which makes it self-calibrating: a noisier camera automatically demands a taller
    # obstruction, and a clean one keeps the full 2 mm sensitivity. At k = 3 the expected number
    # of noise pixels over the bar is 0,2 of 157, against a `min_pixels` floor of 8. The height
    # reported is then a high percentile of the pixels that cleared it, not the single worst.
    rms_mm = float(fit.quality.get('rms_mm', 0.0) or 0.0)
    thr_m = max(inlier_mm, noise_k * rms_mm) * 1e-3
    m = (perp2 <= tool_radius_m ** 2) & (h > thr_m)
    if int(m.sum()) < min_pixels:
        return 0.0
    return float(np.percentile(h[m], pct) * 1e3)
