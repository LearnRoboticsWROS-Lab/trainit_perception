"""Detection <-> vision_msgs, kept out of the node so it is unit-testable without ROS spin."""
from __future__ import annotations

from typing import Iterable

from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .detectors.base import Detection


def to_detection3d_array(dets: Iterable[Detection], frame_id: str, stamp) -> Detection3DArray:
    """Build the perception contract message. Poses are in ``frame_id`` (the optical frame)."""
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
        pose.orientation.w = 1.0                  # identity: a colour mask gives no orientation
        hyp.pose.pose = pose
        m.results.append(hyp)
        m.bbox.center = pose
        m.bbox.size.x, m.bbox.size.y, m.bbox.size.z = map(float, d.size)
        arr.detections.append(m)
    return arr
