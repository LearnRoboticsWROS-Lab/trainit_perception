"""Detection <-> vision_msgs, kept out of the node so it is unit-testable without ROS spin.

THE ORDER OF THE ARRAY IS THE RANKING (ADR-0016 Decision 1). These functions serialise the list
they are given, in the order they are given it, and the caller is `Detector.detect_ranked()` --
which has already applied ONE permutation to the detections and their terms. Nothing here sorts:
a serialiser that re-ranked would have to know what `visible_fraction` means, and the perception
contract exists precisely so that neither the publisher nor the consumer has to.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

from geometry_msgs.msg import Pose, Quaternion
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .detectors.base import Detection


def quat_from_z(n: Sequence[float]) -> Quaternion:
    """The shortest rotation taking +Z onto ``n``. Yaw about the normal is left unconstrained.

    A surface normal fixes two degrees of freedom, not three, and inventing the third would be a
    lie a consumer could not detect. A suction cup does not care; a two-finger gripper would, and
    would need a method that also returns a direction in the plane.
    """
    x, y, z = (float(v) for v in n)
    ln = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(ln) or ln <= 0.0:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    x, y, z = x / ln, y / ln, z / ln
    d = max(-1.0, min(1.0, z))                      # cos of the angle to +Z
    if d > 1.0 - 1e-9:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    if d < -1.0 + 1e-9:
        return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)     # 180 deg about X
    ax, ay = -y, x                                  # cross([0, 0, 1], n), whose z is 0
    s = math.sqrt((1.0 + d) * 2.0)
    qx, qy, qz, qw = ax / s, ay / s, 0.0, s / 2.0
    ln = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return Quaternion(x=qx / ln, y=qy / ln, z=qz / ln, w=qw / ln)


def to_detection3d_array(dets: Iterable[Detection], frame_id: str, stamp) -> Detection3DArray:
    """Build the perception contract message. Poses are in ``frame_id`` (the optical frame).

    ``detections[0]`` is the best candidate. The orientation is the rotation that takes +Z onto
    the detection's fitted ``normal``, so a consumer asking to align with the detected object gets
    a tool axis along the surface normal; a detection with no normal keeps the identity
    quaternion, which is what a centroid-only method honestly has to say (ADR-0017 Decision 5).
    """
    arr = Detection3DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = frame_id
    for i, d in enumerate(dets):
        m = Detection3D()
        m.header = arr.header
        m.id = f'{d.class_id}_{i}'
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = d.class_id
        hyp.hypothesis.score = float(d.score)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, d.position)
        pose.orientation = (quat_from_z(d.normal) if d.normal is not None
                            else Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        hyp.pose.pose = pose
        m.results.append(hyp)
        m.bbox.center = pose
        m.bbox.size.x, m.bbox.size.y, m.bbox.size.z = map(float, d.size)
        arr.detections.append(m)
    return arr


def to_quality_payload(dets: Iterable[Detection], frame_id: str, stamp) -> Dict[str, object]:
    """The candidate TERMS, one dict per detection, in the SAME order as the detections.

    ``vision_msgs/Detection3D`` has no free-form field, so the terms travel on a sidecar and the
    two are paired BY STAMP -- not by the order two topics happened to arrive in, which ROS does
    not guarantee. ``detections[i]`` here is ``detections[i]`` there, because one permutation was
    applied to both: a tuner reading the terms of candidate *i* is reading the terms of the pose
    the picker will try *i*-th.

    Non-finite values go out as ``null``: ``obstruction_mm`` is NaN for "the depth could not
    answer", which is a different answer from 0 ("clear"), and the literal ``NaN`` that
    ``json.dumps`` would emit is not valid JSON -- a strict parser rejects the whole message.
    """
    out: List[Dict[str, object]] = []
    for d in dets:
        row: Dict[str, object] = {'class_id': d.class_id}
        for k, v in d.extra.items():
            f = float(v)
            row[k] = f if math.isfinite(f) else None
        out.append(row)
    return {'stamp': {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)},
            'frame_id': frame_id, 'detections': out}


def to_quality_json(dets: Iterable[Detection], frame_id: str, stamp) -> str:
    import json
    return json.dumps(to_quality_payload(dets, frame_id, stamp))
