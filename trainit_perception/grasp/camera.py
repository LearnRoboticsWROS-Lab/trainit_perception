"""The camera model a surface-grasp provider back-projects through (vNext M4, decision PA-9).

ROS-free. Built from the fields of a `sensor_msgs/CameraInfo` (K, D, distortion model, size, frame), so the
node, the offline harnesses and the tests build the same object.

WHY IT EXISTS. `detector_node` caches K from the first CameraInfo and ignores the distortion vector D (KNOWN_DEBT,
D0 audit). That is exact in Isaac (D = 0) and wrong on a real lens: a D455 colour stream reports `plumb_bob`
D = (-0.056, 0.064, -0.0004, 0.0002, -0.020), a ~1 % lateral error at mid-field -- about 2,5 mm at 0,25 m off-axis
and 0,5 m range, which is material against a 21 mm face and a 10 mm suction cup. So this model:

- undistorts every pixel ONCE per camera model (`rays`, cached) and the fit back-projects P = z * (x, y, 1);
- handles `plumb_bob` (OpenCV's model; ROS's name for it) and refuses any other model with a non-zero D instead of
  silently ignoring it (an `equidistant` fisheye read as a pinhole is a wrong grasp, not a degraded one);
- is compared, not cached forever: the node rebuilds it when a CameraInfo differs (`same_as`).

Registered depth (the camera contract's clause, `trainit_hardware_layer/contracts/camera.md`): depth lives on the
colour pixel grid with an identical CameraInfo, so ONE model serves both. A depth CameraInfo that differs is
refused by the node, never averaged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

#: distortion models this model can undistort when D is non-zero
SUPPORTED_DISTORTION = ('plumb_bob',)


class CameraModelError(ValueError):
    """A CameraInfo this provider must not back-project through."""


@dataclass(frozen=True)
class CameraModel:
    K: Tuple[float, ...]                    # row-major 3x3
    D: Tuple[float, ...]
    distortion_model: str
    width: int
    height: int
    frame_id: str
    _rays: list = field(default_factory=list, repr=False, compare=False, hash=False)

    @classmethod
    def from_fields(cls, *, k: Sequence[float], d: Sequence[float], distortion_model: str, width: int,
                    height: int, frame_id: str) -> 'CameraModel':
        """Validate and build. Raises `CameraModelError` naming what is wrong."""
        K = tuple(float(x) for x in k)
        D = tuple(float(x) for x in (d or ()))
        problems = []
        if len(K) != 9 or not all(np.isfinite(K)):
            problems.append(f'K must be 9 finite numbers (got {len(K)})')
        elif not (K[0] > 0 and K[4] > 0) or K[8] != 1.0:
            problems.append(f'K is not an intrinsic matrix (fx {K[0]}, fy {K[4]}, k33 {K[8]})')
        if not all(np.isfinite(D)):
            problems.append('D is not finite')
        if int(width) <= 0 or int(height) <= 0:
            problems.append(f'the image size {width}x{height} is empty')
        if not frame_id:
            problems.append('no frame_id: the candidates would have no frame')
        nonzero = any(abs(x) > 0.0 for x in D)
        model = str(distortion_model or '')
        if nonzero and model not in SUPPORTED_DISTORTION:
            problems.append(f'distortion model {model!r} with a non-zero D is not supported '
                            f'(supported: {", ".join(SUPPORTED_DISTORTION)}); refusing rather than ignoring it')
        if nonzero and model == 'plumb_bob' and len(D) not in (4, 5, 8, 12, 14):
            problems.append(f'plumb_bob D has {len(D)} coefficients (OpenCV takes 4, 5, 8, 12 or 14)')
        if problems:
            raise CameraModelError('; '.join(problems))
        return cls(K=K, D=D, distortion_model=model, width=int(width), height=int(height), frame_id=str(frame_id))

    @classmethod
    def pinhole(cls, K: np.ndarray, width: int, height: int, frame_id: str = 'camera_optical_frame') -> 'CameraModel':
        """A distortion-free model (Isaac, a synthetic test)."""
        return cls.from_fields(k=np.asarray(K, float).ravel(), d=(), distortion_model='plumb_bob', width=width,
                               height=height, frame_id=frame_id)

    # --- what the provider uses --------------------------------------------------------------------
    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.K, float).reshape(3, 3)

    @property
    def distorted(self) -> bool:
        return any(abs(x) > 0.0 for x in self.D)

    def same_as(self, other: Optional['CameraModel']) -> bool:
        return other is not None and (self.K, self.D, self.distortion_model, self.width, self.height,
                                      self.frame_id) == (other.K, other.D, other.distortion_model, other.width,
                                                         other.height, other.frame_id)

    def rays(self) -> np.ndarray:
        """H x W x 2: the undistorted normalised coordinate (x, y) of every pixel centre. Cached."""
        if self._rays:
            return self._rays[0]
        Km = self.matrix
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        if not self.distorted:
            out = np.stack([(u - Km[0, 2]) / Km[0, 0], (v - Km[1, 2]) / Km[1, 1]], -1)
        else:
            import cv2
            pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
            # the iterative form with an explicit stopping rule: the default 5 iterations leave ~1e-4 of
            # residual at the image corners; 40 / 1e-10 converge to the model's own precision
            und = cv2.undistortPointsIter(pts, Km, np.asarray(self.D, float), None, None,
                                          (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10))
            out = und.reshape(self.height, self.width, 2).astype(np.float64)
        out.setflags(write=False)
        self._rays.append(out)
        return out

    def project(self, P: np.ndarray) -> np.ndarray:
        """N x 3 points in the optical frame -> N x 2 pixels (with the lens distortion)."""
        P = np.asarray(P, float).reshape(-1, 3)
        Km = self.matrix
        if not self.distorted:
            z = P[:, 2:3]
            return np.concatenate([Km[0, 0] * P[:, 0:1] / z + Km[0, 2], Km[1, 1] * P[:, 1:2] / z + Km[1, 2]], 1)
        import cv2
        px, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Km, np.asarray(self.D, float))
        return px.reshape(-1, 2)
