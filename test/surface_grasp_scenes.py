"""Deterministic geometric scenes for the SurfaceGrasp tests (vNext M4-A): exact ray casting, no renderer.

Each pixel's UNDISTORTED ray (from the camera model, so a distorted lens is rendered exactly) is intersected with
discs and an optional background plane; the nearest hit wins. A disc's mask is the pixels where it is the nearest
surface. The true contact centre and normal of every disc are known exactly, so an error is the provider's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from trainit_perception.grasp import CameraModel, Region

W, H = 320, 240
K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
D455_LIKE_D = (-0.0562724731862545, 0.06413473188877106, -0.000422674318542704, 0.00022628488659393042,
               -0.020413396880030632)


def pinhole() -> CameraModel:
    return CameraModel.pinhole(K, W, H, 'cam_optical')


def distorted() -> CameraModel:
    return CameraModel.from_fields(k=K.ravel(), d=D455_LIKE_D, distortion_model='plumb_bob', width=W, height=H,
                                   frame_id='cam_optical')


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def tilted(deg: float, about: str = 'y') -> np.ndarray:
    """A normal facing the camera (-Z) tilted by `deg` about the optical x or y axis."""
    t = math.radians(deg)
    if about == 'y':
        return unit((math.sin(t), 0.0, -math.cos(t)))
    return unit((0.0, math.sin(t), -math.cos(t)))


@dataclass
class Disc:
    centre: Tuple[float, float, float]
    normal: Tuple[float, float, float]          # either sign; the renderer does not care
    radius: float
    class_id: str = 'part'
    score: float = 0.9


def render(discs: Sequence[Disc], camera: Optional[CameraModel] = None, *,
           background_z: Optional[float] = 1.0, spheres=()) -> Tuple[np.ndarray, List[np.ndarray]]:
    """-> (depth in metres, one bool mask per disc, then per sphere ((centre), radius)). Depth is 0 where nothing
    is hit."""
    cam = camera or pinhole()
    rays = cam.rays()
    d = np.concatenate([rays, np.ones(rays.shape[:2] + (1,))], -1)        # H x W x 3, z = 1
    best_t = np.full(rays.shape[:2], np.inf)
    owner = np.full(rays.shape[:2], -1)
    for i, s in enumerate(discs):
        n = unit(s.normal)
        c = np.asarray(s.centre, float)
        denom = d @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (c @ n) / denom
        X = d * t[..., None]
        inside = (np.linalg.norm(X - c, axis=-1) <= s.radius) & np.isfinite(t) & (t > 0)
        closer = inside & (t < best_t)
        best_t[closer] = t[closer]
        owner[closer] = i
    for i, sp in enumerate(spheres or ()):
        c = np.asarray(sp[0], float)
        r = float(sp[1])
        b = d @ c
        dd = np.einsum('ijk,ijk->ij', d, d)
        disc = b * b - dd * (c @ c - r * r)
        with np.errstate(invalid='ignore'):
            t = (b - np.sqrt(disc)) / dd
        hit = (disc >= 0) & (t > 0) & (t < best_t)
        best_t[hit] = t[hit]
        owner[hit] = len(discs) + i
    if background_z is not None:
        bg = (best_t == np.inf)
        best_t[bg] = background_z                 # a plane z = background_z: t = z for z = 1 rays
    depth = np.where(np.isfinite(best_t), best_t, 0.0).astype(np.float32)
    masks = [owner == i for i in range(len(discs) + len(spheres or ()))]
    return depth, masks


def regions(discs: Sequence[Disc], masks: Sequence[np.ndarray], with_boxes: bool = True) -> List[Region]:
    out = []
    for s, m in zip(discs, masks):
        box = (0.0, 0.0, 0.0, 0.0)
        if with_boxes and m.any():
            v, u = np.nonzero(m)
            box = (float(u.min()), float(v.min()), float(u.max() + 1), float(v.max() + 1))
        out.append(Region(class_id=s.class_id, score=s.score, mask=m, box_px=box))
    return out


BASE = {'eef': 'main', 'site': 'suction', 'contact_radius_m': 0.008, 'approach_cone_deg': 45.0,
        'standoff_m': 0.05}
