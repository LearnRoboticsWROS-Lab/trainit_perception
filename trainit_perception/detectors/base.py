"""The detector interface, and the geometry every detector shares.

A detector is a PURE FUNCTION from (rgb, depth, K, params) to a list of detections.
No ROS inside. That single constraint is what makes the Setup Assistant trustworthy:
its live tuner and the runtime node call the same ``detect()``, so what the user sees
while tuning is, by construction, what the bundle runs.

It also RANKS. The published order of the detections is the ranking (ADR-0016
Decision 1) and ``candidate_rule`` is a detector parameter (ADR-0017 Decision 1), so
``detect()`` stays the pure find-the-objects function and ``detect_ranked()`` is the
one the node publishes from. A tuner must call ``detect_ranked()`` too: the picture
and the bundle have to agree about which candidate comes first.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..ranking.ranker import Ranker, ranking_defaults
from ..ranking.rules import PROVIDES_SCORE

if TYPE_CHECKING:                       # the fit is a ranking type; importing it at runtime here
    from ..ranking.fit3d import GraspFit    # would make `detectors` and `ranking` mutual


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
    # (u0, v0, u1, v1) in pixels; all zeros = unknown. A BOX is the one occlusion input every
    # detector has, including one with no mask and no depth, so `box_overlap` needs this and
    # nothing else (ADR-0014).
    box_px: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # The fitted surface normal, in the optical frame, pointing BACK AT THE CAMERA. None = this
    # method claims no orientation, and the published pose keeps an identity quaternion.
    normal: Optional[Tuple[float, float, float]] = None
    # The 3D fit this detection came from, when the detector ran one: it carries the quality
    # terms `pose_quality` reads and the surface the clearance is measured against.
    fit: Optional['GraspFit'] = None
    # The candidate TERMS ride here, and they are PUBLISHED: `msgs.to_quality_payload` sends this
    # dict on the quality sidecar in the same permutation as the detections (ADR-0017 Decision 5).
    extra: Dict[str, float] = field(default_factory=dict)


class Detector(ABC):
    """Base class. Subclasses are registered by ``method`` name in ``registry.py``."""

    method: str = ''
    #: What each ``Detection`` this detector produces carries. It decides which candidate TERMS
    #: are computable, and therefore which ``candidate_rule`` can be configured at all
    #: (``ranking.rules.PROVIDES_*``). A detector that grows a mask or a 3D fit widens this.
    PROVIDES: FrozenSet[str] = frozenset({PROVIDES_SCORE})

    def __init__(self, params: Optional[dict] = None):
        self.params = dict(self.defaults())
        self.params.update(params or {})
        # The rank is a DETECTOR parameter (ADR-0017 Decision 1), so it is read here -- and a
        # `candidate_rule` whose terms this detector cannot compute raises HERE, inside
        # `make_detector`, inside the node's constructor, before a single frame is published.
        self.ranker = Ranker.from_params(self.params, provides=self.PROVIDES)

    @classmethod
    def ranking_defaults(cls) -> dict:
        """The ranking parameters this detector should expose, for its ``defaults()`` to merge."""
        return ranking_defaults(cls.PROVIDES)

    @classmethod
    @abstractmethod
    def defaults(cls) -> dict:
        """Every parameter with its default. The profile YAML documents them."""

    @abstractmethod
    def detect(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> List[Detection]:
        """rgb: HxWx3 uint8 (RGB order). depth_m: HxW float32 METRES, 0/nan = no data.
        K: 3x3 intrinsics of the rgb image, which the depth must be registered to."""

    def detect_ranked(self, rgb: np.ndarray, depth_m: np.ndarray,
                      K: np.ndarray) -> List[Detection]:
        """``detect()``, then the candidate ranking: BEST FIRST, with the terms on ``extra``.

        This is what the node publishes and what a live tuner should show, because the PUBLISHED
        ORDER IS THE RANKING (ADR-0016 Decision 1) and a picture that circles a different
        candidate from the one the consumer will take is the one thing the tuner must not do.

        It may return FEWER detections than ``detect()``: a fit whose ``pose_quality`` is exactly
        0.000 is a refusal, not a low score (see ``Ranker.keep``).
        """
        return self.ranker.apply(self.detect(rgb, depth_m, K), K, depth_m=depth_m)

    def debug_mask(self, rgb: np.ndarray,
                   depth_m: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """HxW uint8 mask for the tuner / a debug topic. None if not applicable.
        depth_m is optional so image-only callers keep working; detectors with a
        depth pass-through filter use it when given."""
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
