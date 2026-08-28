"""The detector interface, and the geometry every detector shares.

A detector is a PURE FUNCTION from (rgb, depth, K, params) to a list of detections.
No ROS inside. That single constraint is what makes the Setup Assistant trustworthy:
its live tuner and the runtime node call the same ``detect()``, so what the user sees
while tuning is, by construction, what the bundle runs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """One found object, in the CAMERA OPTICAL frame (+X right, +Y down, +Z forward).

    The frame is deliberately not the robot's: a detector knows nothing about robots.
    Transforming into the planning frame is the consumer's job (the TMR node, via TF).
    """
    class_id: str
    score: float                                  # 0..1, detector-specific meaning
    position: Tuple[float, float, float]          # metres
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # metres, approximate; 0 = unknown
    pixel: Tuple[float, float] = (0.0, 0.0)       # (u, v) centroid, for the tuner / debug
    area_px: int = 0
    extra: Dict[str, float] = field(default_factory=dict)


class Detector(ABC):
    """Base class. Subclasses are registered by ``method`` name in ``registry.py``."""

    method: str = ''

    def __init__(self, params: Optional[dict] = None):
        self.params = dict(self.defaults())
        self.params.update(params or {})

    @classmethod
    @abstractmethod
    def defaults(cls) -> dict:
        """Every parameter with its default. The profile YAML documents them."""

    @abstractmethod
    def detect(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> List[Detection]:
        """rgb: HxWx3 uint8 (RGB order). depth_m: HxW float32 METRES, 0/nan = no data.
        K: 3x3 intrinsics of the rgb image, which the depth must be registered to."""

    def debug_mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """HxW uint8 mask for the tuner / a debug topic. None if not applicable."""
        return None


# --- shared geometry -----------------------------------------------------------------

def depth_to_metres(depth: np.ndarray, encoding: str) -> np.ndarray:
    """Normalise the two encodings the camera contract allows.

    Isaac emits 32FC1 in metres; the RealSense device emits 16UC1 in MILLIMETRES.
    Everything downstream works in float metres, so the difference stops here.
    """
    if encoding == '16UC1':
        return depth.astype(np.float32) / 1000.0
    if encoding == '32FC1':
        return depth.astype(np.float32, copy=False)
    raise ValueError(f'unsupported depth encoding {encoding!r} (expected 16UC1 or 32FC1)')


def depth_at(depth_m: np.ndarray, u: float, v: float, window: int = 5) -> float:
    """Median depth in a (window x window) patch around (u, v), ignoring holes.

    A single pixel is fragile: stereo depth has holes (0 / nan / inf), and an object's
    centroid can land on one. The median over a small patch survives that; the patch
    stays small so it does not blend the object with the surface behind it.
    Returns nan when the whole patch is invalid.
    """
    h, w = depth_m.shape
    r = max(0, window // 2)
    u0, u1 = max(0, int(round(u)) - r), min(w, int(round(u)) + r + 1)
    v0, v1 = max(0, int(round(v)) - r), min(h, int(round(v)) + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    return float(np.median(valid)) if valid.size else float('nan')


def back_project(u: float, v: float, z: float, K: np.ndarray) -> Tuple[float, float, float]:
    """Pixel (u, v) at depth z -> (X, Y, Z) in the optical frame.  X = (u-cx)·Z/fx."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def pixel_extent_to_metres(px: float, z: float, f: float) -> float:
    """A length of ``px`` pixels seen at depth ``z`` is ``px·z/f`` metres."""
    return px * z / f
