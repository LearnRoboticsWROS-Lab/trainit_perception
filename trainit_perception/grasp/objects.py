"""OBJECT PRESENCE: the eligible regions a grasp provider saw, as objects (vNext M4.5b, PB-3).

ROS-free and pure. A grasp provider answers TWO questions that must stay apart (M4_5B_DESIGN.md §3):

    valid grasp available  -> the candidate array (unchanged: GraspCandidate is the neutral proposal)
    object present         -> THIS: one object per ELIGIBLE region, whatever its geometry verdict

A capsule near a wall stays PRESENT; only its grasp is invalid. The application (TSA's `presence` capture) tells an
empty bin (no object) from parts no grasp can serve (objects, no candidate) with TMR's existing `DetectObject` on
`/perception/<name>/objects` (`vision_msgs/Detection3DArray`, TP's detection contract): no new message type.

An object's position is the 3D CENTROID (the mean) of the region's VALID registered depth points, back-projected
through the same camera rays and the same depth band (`z_min_m` < z < `z_max_m`) the provider fits with; its size
is the 5th..95th percentile extent of those points per optical axis (robust to a mask edge on the background).
A region with NO valid depth point is never dropped silently: it is placed on its pixel centroid's ray at the
median valid depth of the whole frame (the best available estimate, size 0 = unknown, `depth: estimated`) and
counted `objects_without_depth`. When the frame has no valid depth at all (a depth dropout, a wrong depth band) its
regions cannot be measured: they are listed `skipped` and the result is UNMEASURABLE (`PresenceResult.unmeasurable`).
AN UNMEASURABLE FRAME IS NEVER AN EMPTY BIN: the provider publishes neither its candidates nor its objects (fail
closed: a presence capture is not served and times out, a fault), it never answers it with an empty array. A frame
with NO eligible region (or only regions without pixels) is a genuine empty answer. A region without pixels
(`REGION_EMPTY`) is not an object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import CameraModel
from .surface_grasp import Region, region_ids

MEASURED = 'measured'
ESTIMATED = 'estimated'


@dataclass(frozen=True)
class PresentObject:
    id: str                                   # the provider's region id (`<class>@<u>,<v>`), as on the diagnostics
    class_id: str
    score: float
    position: Tuple[float, float, float]      # the observation (colour optical) frame, metres
    size: Tuple[float, float, float]          # 0 = unknown
    depth: str                                # MEASURED | ESTIMATED


@dataclass
class PresenceResult:
    objects: List[PresentObject]
    without_depth: List[str]                  # ids of the regions with no valid depth (estimated or skipped)
    skipped: List[str]                        # ids not measurable (no valid depth anywhere in the frame)

    @property
    def unmeasurable(self) -> bool:
        """>= 1 eligible region with pixels, none measurable (the frame has no valid depth): NOT an empty answer --
        the provider must publish nothing for this frame (fail closed)."""
        return bool(self.skipped) and not self.objects

    def summary(self) -> Dict[str, int]:
        return {'objects': len(self.objects), 'objects_without_depth': len(self.without_depth),
                'objects_skipped': len(self.skipped)}


def _valid(z: np.ndarray, z_min_m: float, z_max_m: float) -> np.ndarray:
    return np.isfinite(z) & (z > z_min_m) & (z < z_max_m)


def present_objects(regions: Sequence[Region], depth_m: np.ndarray, camera: CameraModel, z_min_m: float,
                    z_max_m: float) -> PresenceResult:
    """The eligible `regions` (in their order, the ids `propose` gives them) -> the objects present. Check
    `.unmeasurable` before publishing: an unmeasurable result is a sensor failure, never an empty `objects`."""
    depth = np.asarray(depth_m, float)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.shape != (camera.height, camera.width):
        raise ValueError(f'the depth image is {depth.shape[1]}x{depth.shape[0]}, the camera model '
                         f'{camera.width}x{camera.height}')
    rays = camera.rays()
    frame_median: Optional[float] = None
    objects: List[PresentObject] = []
    without: List[str] = []
    skipped: List[str] = []
    for rid, r in zip(region_ids(regions), regions):
        mask = np.asarray(r.mask, bool)
        if mask.shape != depth.shape:
            raise ValueError(f'region {rid}: its mask is {mask.shape}, the depth {depth.shape}')
        v, u = np.nonzero(mask)
        if not len(u):
            continue                                       # REGION_EMPTY: no pixel, no object
        z = depth[v, u]
        good = _valid(z, z_min_m, z_max_m)
        cls, score = str(r.class_id), float(np.clip(float(r.score), 0.0, 1.0))
        if good.any():
            xy = rays[v[good], u[good]]
            P = np.stack([xy[:, 0] * z[good], xy[:, 1] * z[good], z[good]], 1)
            lo, hi = np.percentile(P, [5.0, 95.0], axis=0)
            objects.append(PresentObject(id=rid, class_id=cls, score=score,
                                         position=tuple(float(x) for x in P.mean(0)),
                                         size=tuple(float(x) for x in (hi - lo)), depth=MEASURED))
            continue
        without.append(rid)
        if frame_median is None:
            all_good = _valid(depth, z_min_m, z_max_m)
            frame_median = float(np.median(depth[all_good])) if all_good.any() else float('nan')
        if not np.isfinite(frame_median):
            skipped.append(rid)
            continue
        uc, vc = int(round(float(u.mean()))), int(round(float(v.mean())))
        x, y = rays[vc, uc]
        objects.append(PresentObject(id=rid, class_id=cls, score=score,
                                     position=(float(x * frame_median), float(y * frame_median), frame_median),
                                     size=(0.0, 0.0, 0.0), depth=ESTIMATED))
    return PresenceResult(objects=objects, without_depth=without, skipped=skipped)


def presence_diagnostics(result: PresenceResult) -> Dict[str, object]:
    """The `objects` block of the provider's diagnostics record (additive)."""
    return {'published': [{'id': o.id, 'class_id': o.class_id, 'score': o.score, 'position': list(o.position),
                           'size': list(o.size), 'depth': o.depth} for o in result.objects],
            'without_depth': list(result.without_depth), 'skipped': list(result.skipped)}
