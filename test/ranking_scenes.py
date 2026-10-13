"""Synthetic scenes whose answer is arithmetic, shared by the ranking tests.

The twin scores the fit against a simulator; these score it against geometry. A disc at a chosen
distance and tilt has one correct centre and one correct normal, so an error here is in the fit
and nowhere else -- no renderer, no model, no mask, no physics.
"""
from __future__ import annotations

import numpy as np

# The cell's VERIFIED intrinsics, cropped to a 256x224 ROI the way a bin detector runs.
K = np.array([[424.0, 0.0, 128.0], [0.0, 424.0, 112.0], [0.0, 0.0, 1.0]])
R_FACE = 0.02095                       # a 41,9 mm cap: the face every measured number was taken on


def synth_disc(centre, normal, radius=R_FACE, shape=(224, 256), noise_mm=0.0,
               hole_frac=0.0, seed=0):
    """Render a flat disc into a depth image, analytically. Returns (mask, depth_m, centre, n).

    Every pixel whose ray meets the disc gets the exact distance along the optical axis, so the
    only error a fit can show is its own.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    denom = d @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (c @ n) / denom
    P = d * t[..., None]
    r = np.linalg.norm(P - c, axis=-1)
    mask = np.isfinite(t) & (t > 0) & (r <= radius)
    depth = np.full(shape, np.nan, np.float32)
    z = P[..., 2].astype(np.float32)
    if noise_mm:
        z = z + rng.normal(0.0, noise_mm * 1e-3, z.shape).astype(np.float32)
    depth[mask] = z[mask]
    if hole_frac:
        holes = rng.random(shape) < hole_frac
        depth[holes] = np.nan
    return mask, depth, c, n


def synth_rect(centre, normal, half_a=0.015, half_b=0.010, in_plane_deg=0.0,
               shape=(224, 256), noise_mm=0.0, hole_frac=0.0, seed=0):
    """Render a flat RECTANGLE into a depth image, analytically.

    Same plane arithmetic as `synth_disc`, with the disc's radial test replaced by two
    half-extent tests along the plane's own in-plane axes -- which is exactly the difference
    `RectFace` exists for. `in_plane_deg` rotates the rectangle about its own normal, so a test
    can check that the projected AREA does not depend on that rotation while the two half-extents
    the fit recovers do swap over.

    Returns (mask, depth_m, centre, n, (e1, e2)) -- the in-plane axes too, because the whole
    point of a rectangular face is that the two axes are not interchangeable.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    # any pair of in-plane axes, then rotated by `in_plane_deg` about the normal
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a = np.radians(float(in_plane_deg))
    e1, e2 = np.cos(a) * e1 + np.sin(a) * e2, -np.sin(a) * e1 + np.cos(a) * e2
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    denom = d @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (c @ n) / denom
    P = d * t[..., None]
    q = P - c
    mask = (np.isfinite(t) & (t > 0) & (np.abs(q @ e1) <= half_a) & (np.abs(q @ e2) <= half_b))
    depth = np.full(shape, np.nan, np.float32)
    z = P[..., 2].astype(np.float32)
    if noise_mm:
        z = z + rng.normal(0.0, noise_mm * 1e-3, z.shape).astype(np.float32)
    depth[mask] = z[mask]
    if hole_frac:
        depth[rng.random(shape) < hole_frac] = np.nan
    return mask, depth, c, n, (e1, e2)


def synth_cylinder(centre, axis, radius=0.0325, height=0.060, shape=(224, 256),
                   noise_mm=0.0, hole_frac=0.0, seed=0):
    """Render the SIDE of a cylinder into a depth image, analytically: ray/cylinder intersection.

    This is the scene a bottle picked from the side presents, and it is the one that settles what
    a `CylinderSideFace` may honestly claim: the mask is the cylinder's SILHOUETTE (2*radius wide
    whatever the roll about the axis), while a PLANE fit over it only ever locks onto the narrow
    band where the curved surface stays inside the RANSAC tolerance.

    `centre` is the midpoint of the axis, `axis` its direction in the optical frame. Returns
    (mask, depth_m, centre, axis).
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(centre, float)
    A = np.asarray(axis, float)
    A = A / np.linalg.norm(A)
    h_img, w_img = shape
    v, u = np.mgrid[0:h_img, 0:w_img]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    # the ray is t*d; measure everything perpendicular to the axis, where the cylinder is a circle
    w_vec = -c                                        # origin - centre
    d_perp = d - (d @ A)[..., None] * A
    w_perp = w_vec - (w_vec @ A) * A
    qa = np.einsum('...i,...i->...', d_perp, d_perp)
    qb = 2.0 * (d_perp @ w_perp)
    qc = float(w_perp @ w_perp) - radius ** 2
    disc = qb ** 2 - 4.0 * qa * qc
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (-qb - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * qa)      # the NEAR surface
    P = d * t[..., None]
    along = np.abs((P - c) @ A)
    mask = (disc > 0) & np.isfinite(t) & (t > 0) & (along <= 0.5 * height)
    depth = np.full(shape, np.nan, np.float32)
    z = P[..., 2].astype(np.float32)
    if noise_mm:
        z = z + rng.normal(0.0, noise_mm * 1e-3, z.shape).astype(np.float32)
    depth[mask] = z[mask]
    if hole_frac:
        depth[rng.random(shape) < hole_frac] = np.nan
    return mask, depth, c, A
