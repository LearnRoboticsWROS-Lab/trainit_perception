"""Draw a SurfaceGrasp result on the RGB image: headless, inspectable evidence (vNext M4). Optional, never
canonical -- what a candidate MEANS is the GraspCandidate, not this picture.

Per valid candidate: the contact point (a dot), the surface normal (green, from the contact toward the camera side),
the approach (blue, from the PreGrasp INTO the contact), the rank number; rank 0 (the selected candidate) ringed in
yellow. Per rejection: a red cross at the region's centroid pixel and its code.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .camera import CameraModel
from .surface_grasp import GraspResult


def draw(rgb: np.ndarray, result: GraspResult, camera: CameraModel, *, normal_len_m: float = 0.03,
         rejected_pixels: Optional[Sequence] = None) -> np.ndarray:
    import cv2
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    for c in result.candidates:
        p = np.asarray(c.point)
        n = np.asarray(c.normal)
        pg = np.asarray(c.pregrasp)
        pts = camera.project(np.stack([p, p + normal_len_m * n, pg]))
        P0, PN, PG = (tuple(int(round(x)) for x in q) for q in pts)
        cv2.arrowedLine(img, P0, PN, (0, 220, 0), 2, tipLength=0.25)          # normal
        cv2.arrowedLine(img, PG, P0, (40, 90, 255), 2, tipLength=0.2)         # approach (RGB image: blue-ish)
        cv2.circle(img, P0, 3, (255, 255, 255), -1)
        if c.rank == 0:
            cv2.circle(img, P0, 12, (255, 220, 0), 2)
        cv2.putText(img, str(c.rank), (P0[0] + 6, P0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    for r in result.rejections:
        try:
            u, v = (int(x) for x in r.id.split('@', 1)[1].split('#')[0].split(','))
        except (IndexError, ValueError):
            continue
        cv2.drawMarker(img, (u, v), (255, 40, 40), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(img, r.code, (u + 6, v + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 80), 1, cv2.LINE_AA)
    return img
