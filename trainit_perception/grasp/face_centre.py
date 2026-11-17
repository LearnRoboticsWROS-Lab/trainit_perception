"""The centre of a KNOWN face from partial depth evidence (vNext M4.5d: `centre_estimator: known_shape_fit`).

SurfaceGrasp's default contact is the plane's INLIER CENTROID (`centre_estimator: inlier_centroid`): the middle of
whatever part of the face the depth shows inside the region's mask. That is the face's centre only when the whole
face is visible and the mask is exact. On a reference bin-picking cell (shallow cup-shaped parts, M4.5) a part's
face lay close to the bin floor's plane, so a segmenter's mask that SPILLS over the part's outline added floor pixels
to the face's plane, and a mask SMALLER than the part truncated the face: the visible centroid moved several mm off
the face's centre (the measurements are in the workspace's M4.5d design record, not here).

This module answers the question geometrically, for a face whose SHAPE is known (`grasp_face: disc` with
`expected_radius_m`), and knows nothing about capsules:

  1. THE FACE'S SUPPORT. The mask says WHICH object; the depth says how far its face extends. The plane's points
     (|height| <= the inlier band) are grown from the region's own in-plane pixels, 4-connected, inside the mask
     dilated by `face_mask_dilate_px` -- so a mask a few pixels too small no longer truncates the face, and a mask
     that spills only adds thin slivers, never a second face.
  2. CONSENSUS. The disc of the EXPECTED radius that contains the most support points (a grid search of centres in
     the face's plane; the best ones form a plateau). Spill and neighbours outside the disc stop mattering; a face
     split by an occluder (a neighbour's tab across it) still counts in full.
  3. EDGE REFINEMENT. A support pixel whose neighbour (outside the support) is NOT the same plane is a boundary of
     the face; that neighbour's height above the plane classifies it:
        below the plane (a drop-off) or rising by at most `face_occlusion_mm` (a crease into a wall) -> the face's
        TRUE EDGE: its sub-pixel position (the midpoint of the two pixels' rays on the plane) is evidence;
        in front by more than `face_occlusion_mm`                      -> OCCLUDED: it bounds what is visible, not
                                                                          the face -- not evidence;
        the same plane, or no depth                                   -> unknown -- not evidence.
     The expected circle is least-squares fitted to the true-edge points. It is used when it is well supported
     (enough points, an arc of at least `face_min_arc_deg`, an rms within `face_max_rms_mm`, a free-radius fit
     within `face_radius_tol` of the expected one) and CONSISTENT with the consensus (inside its plateau); else
     the consensus plateau's mean is the centre, provided the plateau is small (`face_max_ambiguity_mm`).
  4. NO RIVAL. The chosen disc must be the ONLY disc-sized face in the support: a disc of the expected radius
     anywhere else in the support that holds mostly points the chosen one does not (a coplanar blob the mask covered,
     as large as the face) and at least `max_rival` of the chosen disc's count makes the answer ambiguous -> refused
     (review M4.5d: a disc touching a coplanar 40 mm square inside one mask was otherwise answered 39 mm off).

FAIL CLOSED: no support, too little of the expected disc visible (`face_min_coverage`), a face larger than the
expected disc (`face_max_outside`: a wrong expected diameter or two faces merged), a rival disc-sized face in the
support, a consensus that does not settle inside its search window, or a centre the evidence does not pin down -> `ok = False` with the reason; the caller rejects the region (`FACE_FIT_FAILED`). There is NO
fallback to the visible centroid. Which evidence decided (`edges` / `consensus`) is reported in `quality`.

Known limits (documented, fail-safe): the edge classifier cannot tell a crease from an occluder standing less than
`face_occlusion_mm` in front of the plane, nor a flying pixel at an occlusion boundary from an edge (the arc / rms /
radius / plateau checks reject such fits); SurfaceGrasp's downstream contact checks still use the region's mask,
so a face whose fitted centre lies well outside a truncated mask is refused there (CONTACT_UNSUPPORTED), not moved.
Cost is bounded: the grid searches use at most `max_points` support points (a deterministic stride).

Pure numpy + OpenCV (connected components, dilation, flood fill); deterministic: no random draws.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

#: what decided the centre
BY_EDGES = 'edges'
BY_CONSENSUS = 'consensus'


@dataclass(frozen=True)
class KnownDiscParams:
    radius_m: float
    band_m: float                      # the plane's inlier band (SurfaceGrasp's inlier_mm)
    occlusion_m: float = 0.003
    dilate_px: int = 0
    grid_m: float = 0.0005
    tol_m: float = 0.001
    min_coverage: float = 0.35
    max_outside: float = 0.6
    max_ambiguity_m: float = 0.003
    min_edge_points: int = 12
    min_arc_deg: float = 150.0
    max_rms_m: float = 0.001
    radius_tol: float = 0.1
    z_min_m: float = 0.05
    z_max_m: float = 3.0
    max_rival: float = 0.6             # a rival disc with this share of the chosen count -> ambiguous (fixed rule)
    max_points: int = 4000             # the grid searches' point budget (a deterministic stride above it)

    def __post_init__(self):
        if not (self.radius_m > 0 and self.band_m > 0 and self.grid_m > 0 and self.tol_m >= 0):
            raise ValueError('KnownDiscParams: radius_m, band_m and grid_m must be > 0, tol_m >= 0')
        if not self.occlusion_m > self.band_m:
            raise ValueError(f'KnownDiscParams: occlusion_m ({self.occlusion_m}) must exceed band_m ({self.band_m})')
        if not (0 < self.max_rival <= 1 and self.max_points >= 100):
            raise ValueError('KnownDiscParams: max_rival in (0, 1], max_points >= 100')


@dataclass
class FaceCentre:
    ok: bool
    why: str = ''
    centre: Optional[Tuple[float, float, float]] = None      # on the plane, the observation frame
    quality: Dict[str, float] = field(default_factory=dict)
    by: str = ''


def plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit in-plane axes for the unit normal `n` (deterministic)."""
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0.0, 1.0, 0.0])
    a = a / np.linalg.norm(a)
    return a, np.cross(n, a)


def fit_circle_known_radius(q: np.ndarray, radius: float, c0: np.ndarray, iters: int = 50) -> np.ndarray:
    """Gauss-Newton least squares of sum (|q_i - c| - radius)^2 over the centre c (2-D), from c0."""
    c = np.asarray(c0, float).copy()
    for _ in range(iters):
        d = q - c
        r = np.linalg.norm(d, axis=1)
        r = np.where(r < 1e-12, 1e-12, r)
        J = -d / r[:, None]
        step = np.linalg.lstsq(J, -(r - radius), rcond=None)[0]
        c = c + step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    return c


def fit_circle_free(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic (Kasa) circle fit: centre and radius, for the radius CHECK only."""
    A = np.c_[2 * q, np.ones(len(q))]
    sol = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    return c, math.sqrt(max(float(sol[2] + c @ c), 0.0))


def arc_coverage_deg(q: np.ndarray, c: np.ndarray) -> float:
    """The angle of the circle (about c) that the points span: 360 minus the largest angular gap."""
    if len(q) < 2:
        return 0.0
    ang = np.sort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    return math.degrees(2 * math.pi - float(gaps.max()))


def _subsample(Q: np.ndarray, budget: int) -> np.ndarray:
    """At most `budget` points of Q, by a deterministic stride (the grid searches' cost bound)."""
    if len(Q) <= budget:
        return Q
    return Q[np.linspace(0, len(Q) - 1, budget).astype(np.int64)]


def _counts(Qs: np.ndarray, C: np.ndarray, lim: float) -> np.ndarray:
    """How many points of Qs lie within sqrt(lim) of each centre of C (bounded memory: ~16 MB per chunk)."""
    cnt = np.zeros(len(C), np.int64)
    step = max(1, int(1_000_000 // max(len(Qs), 1)))
    for i in range(0, len(C), step):
        d2 = ((Qs[None, :, :] - C[i:i + step, None, :]) ** 2).sum(-1)
        cnt[i:i + step] = (d2 <= lim).sum(1)
    return cnt


def _consensus(Q: np.ndarray, start: np.ndarray, R: float, prm: KnownDiscParams):
    """(plateau mean, plateau extent, best count (in Q's points), plateau centres, settled): the centres on a grid
    of prm.grid_m within 0.6 R of `start` whose disc (R + tol) contains the most points of Q (within 0.5 %). A
    plateau that touches the window's border is re-searched around its mean (at most 3 times); `settled` is False
    when it still touches it (the caller refuses: the maximum may lie outside the window)."""
    Qs = _subsample(Q, prm.max_points)
    scale = len(Q) / len(Qs)
    span = 0.6 * R
    g = np.arange(-span, span + 1e-12, prm.grid_m)
    lim = (R + prm.tol_m) ** 2
    start = np.asarray(start, float)
    for _ in range(4):
        C = start + np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
        cnt = _counts(Qs, C, lim)
        best = int(cnt.max())
        plateau = C[cnt >= best - max(1, int(0.005 * best))]
        c = plateau.mean(0)
        settled = float(np.max(np.abs(plateau - start))) < span - 0.5 * prm.grid_m
        if settled:
            break
        start = c
    extent = float(np.max(np.linalg.norm(plateau - c, axis=1))) if len(plateau) > 1 else 0.0
    return c, extent, int(round(best * scale)), plateau, settled


def _rival(Q: np.ndarray, chosen: np.ndarray, best: int, R: float, prm: KnownDiscParams) -> Tuple[float, float]:
    """(the strongest rival's count / the chosen count, its distance from the chosen centre [m]): a disc of the
    expected radius anywhere over the support (a grid of R / 8) holding MOSTLY points the chosen disc does not."""
    Qs = _subsample(Q, min(prm.max_points, 3000))
    scale = len(Q) / len(Qs)
    lim = (R + prm.tol_m) ** 2
    other = ((Qs - chosen) ** 2).sum(1) > lim
    if not other.any() or best <= 0:
        return 0.0, 0.0
    step = R / 8.0
    lo, hi = Qs.min(0), Qs.max(0)
    gx = np.arange(lo[0], hi[0] + 1e-12, step)
    gy = np.arange(lo[1], hi[1] + 1e-12, step)
    C = np.stack(np.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)
    total = _counts(Qs, C, lim)
    foreign = _counts(Qs[other], C, lim)
    rival = (foreign * 2 > total)
    if not rival.any():
        return 0.0, 0.0
    i = int(np.argmax(np.where(rival, total, -1)))
    return float(total[i] * scale / best), float(np.linalg.norm(C[i] - chosen))


def known_disc_centre(mask: np.ndarray, depth: np.ndarray, rays: np.ndarray, point: np.ndarray,
                      normal: np.ndarray, prm: KnownDiscParams) -> FaceCentre:
    """The centre of a disc of radius `prm.radius_m` lying in the plane (`point`, unit `normal`, the observation
    frame) from the region `mask`, the registered `depth` [m] and the camera's unit-depth `rays` (H x W x 2).
    See the module docstring for the method and its refusals."""
    import cv2

    p0 = np.asarray(point, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    R = float(prm.radius_m)
    q: Dict[str, float] = {}
    mask = np.asarray(mask, bool)
    H, W = mask.shape
    vv, uu = np.nonzero(mask)
    if not len(vv):
        return FaceCentre(False, 'the region has no pixel', quality=q)
    pad = int(prm.dilate_px) + 2
    v0, v1 = max(int(vv.min()) - pad, 0), min(int(vv.max()) + pad + 1, H)
    u0, u1 = max(int(uu.min()) - pad, 0), min(int(uu.max()) + pad + 1, W)
    m = mask[v0:v1, u0:u1]
    z = np.asarray(depth[v0:v1, u0:u1], float)
    rr = rays[v0:v1, u0:u1]
    valid = np.isfinite(z) & (z > prm.z_min_m) & (z < prm.z_max_m)
    zc = np.where(valid, z, 0.0)
    X = np.stack([rr[..., 0] * zc, rr[..., 1] * zc, zc], -1)
    h = np.where(valid, (X - p0) @ n, np.nan)
    inplane = valid & (np.abs(np.nan_to_num(h, nan=1e9)) <= prm.band_m)

    # 1. THE SUPPORT. Candidates: in-plane pixels inside the mask dilated by dilate_px (a mask a few pixels too
    # small must not truncate the face; the growth stops by itself at the face's physical edges, where the depth
    # leaves the plane). Its 4-connected components that hold region pixels are SEEDED. The PRIMARY one holds the
    # most region pixels; another seeded component joins it only if it lies inside the expected disc found on the
    # primary (one face split by an occluder across it), never when it lies outside (a coplanar NEIGHBOUR the mask
    # spilled onto -- measured: a YOLO mask over part of a neighbouring capsule moved the disc 6 mm).
    k = 2 * int(prm.dilate_px) + 1
    allowed = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) if prm.dilate_px > 0 else m
    cand = inplane & allowed
    nlab, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    seeds = np.bincount(lab[inplane & m], minlength=nlab)
    seeds[0] = 0
    q['face_region_inplane_px'] = float((inplane & m).sum())
    if int(seeds.max()) < 3:
        return FaceCentre(False, 'no in-plane depth inside the region: the face has no support', quality=q)
    e1, e2 = plane_basis(n)

    def coords(sel):
        sv_, su_ = np.nonzero(sel)
        P_ = X[sv_, su_] - p0
        return sv_, su_, np.stack([P_ @ e1, P_ @ e2], 1)

    primary = int(np.argmax(seeds))
    support = lab == primary
    _sv, _su, Qp = coords(support)
    c_p = _consensus(Qp, Qp.mean(0), R, prm)[0]       # a seed only: the final search below re-settles it
    joined = 0
    for lbl in np.nonzero(seeds)[0]:
        if int(lbl) == primary:
            continue
        _a, _b, Qo = coords(lab == lbl)
        if float(np.mean(np.linalg.norm(Qo - c_p, axis=1) <= R + prm.tol_m)) >= 0.9:
            support |= lab == lbl
            joined += 1
    q['face_support_px'] = float(support.sum())
    q['face_components_seeded'] = float(np.count_nonzero(seeds))
    q['face_components_joined'] = float(joined)
    sv, su, Q = coords(support)

    # 2. CONSENSUS: the disc of the expected radius containing the most support points (grid of centres)
    c_cons, extent, best, plateau, settled = _consensus(Q, c_p, R, prm)
    # the expected disc's area in pixels: one pixel's footprint on the plane, at the face's mean depth
    zm = float(np.mean(z[sv, su]))
    du = float(np.median(np.abs(np.diff(rr[..., 0], axis=1)))) if rr.shape[1] > 1 else 0.0
    dv = float(np.median(np.abs(np.diff(rr[..., 1], axis=0)))) if rr.shape[0] > 1 else 0.0
    # a pixel's area on the plane at depth z along the UNIT-DEPTH ray r = (x, y, 1): z^2 du dv / |r . n|
    ray_c = np.array([float(np.mean(rr[sv, su, 0])), float(np.mean(rr[sv, su, 1])), 1.0])
    px_area = (zm * du) * (zm * dv) / max(abs(float(ray_c @ n)), 1e-3)
    coverage = best * px_area / (math.pi * R * R) if px_area > 0 else 0.0
    outside = 1.0 - best / len(Q)
    q.update(face_coverage=coverage, face_outside=outside, face_plateau_mm=extent * 1e3,
             face_consensus_points=float(best))
    if coverage < prm.min_coverage:
        return FaceCentre(False, f'only {coverage:.2f} of the expected disc (r {R * 1e3:.2f} mm) is visible '
                                 f'(face_min_coverage {prm.min_coverage:g}): its centre is not determined', quality=q)
    if outside > prm.max_outside:
        return FaceCentre(False, f'{outside:.2f} of the measured face lies outside the expected disc (r '
                                 f'{R * 1e3:.2f} mm; face_max_outside {prm.max_outside:g}): a wrong expected '
                                 f'diameter, or two faces merged', quality=q)
    if not settled:
        return FaceCentre(False, 'the expected disc\'s best position does not settle inside its search window: the '
                                 'support is not one disc-sized face', quality=q)
    rival, rival_d = _rival(Q, c_cons, best, R, prm)
    q.update(face_rival=rival, face_rival_mm=rival_d * 1e3)
    if rival >= prm.max_rival:
        return FaceCentre(False, f'a second disc-sized face lies in the support {rival_d * 1e3:.1f} mm away, with '
                                 f'{rival:.2f} of the chosen disc\'s points (>= {prm.max_rival:g}): which one is the '
                                 f'face is ambiguous', quality=q)

    # 3. EDGE REFINEMENT: the face's true edges around the consensus disc
    kept = np.zeros_like(support)
    kept[sv, su] = np.linalg.norm(Q - c_cons, axis=1) <= R + prm.tol_m
    # holes INSIDE the kept face (depth outliers, specks) are not its boundary: their pairs are unknown evidence
    if kept.any():
        flood = np.pad(kept, 1).astype(np.uint8)
        cv2.floodFill(flood, None, (0, 0), 1)
        hole = (flood[1:-1, 1:-1] == 0)
    else:
        hole = np.zeros_like(kept)
    edge_pts, n_occ, n_unknown = [], 0, 0
    kv, ku = np.nonzero(kept)
    for dv_, du_ in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ov, ou = kv + dv_, ku + du_
        inside_img = (ov >= 0) & (ov < kept.shape[0]) & (ou >= 0) & (ou < kept.shape[1])
        ov, ou, iv, iu = ov[inside_img], ou[inside_img], kv[inside_img], ku[inside_img]
        out = ~kept[ov, ou]
        n_unknown += int((out & hole[ov, ou]).sum())
        out &= ~hole[ov, ou]
        ov, ou, iv, iu = ov[out], ou[out], iv[out], iu[out]
        ho = h[ov, ou]
        nan = ~np.isfinite(ho)
        same = np.isfinite(ho) & (np.abs(np.nan_to_num(ho)) <= prm.band_m)
        occ = np.isfinite(ho) & (np.nan_to_num(ho) > prm.occlusion_m)
        edge = ~nan & ~same & ~occ
        n_occ += int(occ.sum())
        n_unknown += int((nan | same).sum())
        if edge.any():
            def hit(a, b):
                ray = np.stack([rr[a, b, 0], rr[a, b, 1], np.ones(len(a))], 1)
                return ray * ((p0 @ n) / (ray @ n))[:, None]
            mid = 0.5 * (hit(iv[edge], iu[edge]) + hit(ov[edge], ou[edge])) - p0
            edge_pts.append(np.stack([mid @ e1, mid @ e2], 1))
    E = np.concatenate(edge_pts) if edge_pts else np.zeros((0, 2))
    q.update(face_edge_points=float(len(E)), face_occluded_pairs=float(n_occ), face_unknown_pairs=float(n_unknown))
    c_final, by = None, ''
    if len(E) >= prm.min_edge_points:
        c_e = fit_circle_known_radius(E, R, c_cons)
        arc = arc_coverage_deg(E, c_e)
        rms = float(np.sqrt(np.mean((np.linalg.norm(E - c_e, axis=1) - R) ** 2)))
        _, r_free = fit_circle_free(E)
        to_plateau = float(np.min(np.linalg.norm(plateau - c_e, axis=1)))
        q.update(face_edge_arc_deg=arc, face_edge_rms_mm=rms * 1e3, face_edge_radius_free_mm=r_free * 1e3,
                 face_edge_to_plateau_mm=to_plateau * 1e3)
        if (arc >= prm.min_arc_deg and rms <= prm.max_rms_m and abs(r_free - R) <= prm.radius_tol * R
                and to_plateau <= prm.grid_m + prm.tol_m):
            c_final, by = c_e, BY_EDGES
    if c_final is None:
        if extent > prm.max_ambiguity_m:
            return FaceCentre(False, f'the visible face lets the expected disc slide {extent * 1e3:.2f} mm '
                                     f'(face_max_ambiguity_mm {prm.max_ambiguity_m * 1e3:g}) and its edges do not '
                                     f'pin it: the centre is not determined', quality=q)
        c_final, by = c_cons, BY_CONSENSUS
    inside = kept[sv, su]
    if inside.any():
        q['face_centre_shift_mm'] = float(np.linalg.norm(c_final - Q[inside].mean(0))) * 1e3
    centre = p0 + c_final[0] * e1 + c_final[1] * e2
    return FaceCentre(True, centre=tuple(float(x) for x in centre), quality=q, by=by)
