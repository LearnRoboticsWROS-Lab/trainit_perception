"""SurfaceGrasp result <-> ROS messages, kept out of the node so it is unit-testable without a spinning graph.

`to_candidate_array` serialises the candidates IN THE ORDER GIVEN -- the provider's ranking, best first (ADR-0016,
ADR-0026: the array order is the rank and is never re-sorted). The markers are optional debug: they draw the same
geometry and carry no grasp semantics.
"""
from __future__ import annotations

import json
from typing import Iterable, Sequence, Tuple

from diagnostic_msgs.msg import KeyValue
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from trainit_interfaces.msg import GraspCandidate, GraspCandidateArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from .grasp.surface_grasp import PROVIDER_ID, Candidate, GraspResult, diagnostics


def to_candidate(c: Candidate, header) -> GraspCandidate:
    m = GraspCandidate()
    m.header = header
    m.id = c.id
    m.provider = PROVIDER_ID
    m.grasp_pose.position.x, m.grasp_pose.position.y, m.grasp_pose.position.z = (float(x) for x in c.point)
    (m.grasp_pose.orientation.x, m.grasp_pose.orientation.y, m.grasp_pose.orientation.z,
     m.grasp_pose.orientation.w) = (float(x) for x in c.orientation_xyzw)
    m.approach = Vector3(x=float(c.approach[0]), y=float(c.approach[1]), z=float(c.approach[2]))
    m.standoff_m = float(c.standoff_m)
    m.eef, m.site = c.eef, c.site
    m.score, m.confidence = float(c.score), float(c.confidence)
    m.object_class, m.object_id = c.class_id, ''
    m.surface_normal = Vector3(x=float(c.normal[0]), y=float(c.normal[1]), z=float(c.normal[2]))
    m.metadata = [KeyValue(key=k, value=v) for k, v in c.metadata()]
    return m


def to_candidate_array(result: GraspResult, header) -> GraspCandidateArray:
    arr = GraspCandidateArray()
    arr.header = header
    arr.candidates = [to_candidate(c, header) for c in result.candidates]
    return arr


def to_diagnostics_json(result: GraspResult, header, camera=None, *, ineligible=(), segmentation=None,
                        presence=None) -> str:
    """`presence` (0.7.0, M4.5b): the frame's `PresenceResult` -- `summary.objects` / `summary.objects_without_depth`
    and an `objects` block (additive)."""
    out = diagnostics(result, header.frame_id, header.stamp.sec, header.stamp.nanosec, camera,
                      ineligible=ineligible, segmentation=segmentation)
    if presence is not None:
        from .grasp.objects import presence_diagnostics
        out['summary'].update(presence.summary())
        out['objects'] = presence_diagnostics(presence)
    return json.dumps(out, sort_keys=False)


def object_detection(object_id: str, class_id: str, score: float, position: Sequence[float],
                     size: Sequence[float], header) -> Detection3D:
    """One object present as a `Detection3D`: `id` = the provider's region id, `results[0]` = the segmentation's
    class and score at the object's position (identity orientation: presence claims no surface), `bbox` = the same
    position and the metric size (0 = unknown)."""
    m = Detection3D()
    m.header = header
    m.id = str(object_id)
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = str(class_id)
    hyp.hypothesis.score = float(score)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(x) for x in position)
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    hyp.pose.pose = pose
    m.results.append(hyp)
    m.bbox.center = pose
    m.bbox.size.x, m.bbox.size.y, m.bbox.size.z = (float(x) for x in size)
    return m


def objects_array(rows: Iterable[Tuple[str, str, float, Sequence[float], Sequence[float]]],
                  header) -> Detection3DArray:
    """(id, class_id, score, position, size) rows -> the presence array, in the order given, under `header` (the
    candidate array's own header: the same stamp and frame)."""
    arr = Detection3DArray()
    arr.header = header
    arr.detections = [object_detection(*row, header) for row in rows]
    return arr


def to_objects_array(presence, header) -> Detection3DArray:
    """A `PresenceResult` -> the presence array (`/perception/<name>/objects`)."""
    return objects_array(((o.id, o.class_id, o.score, o.position, o.size) for o in presence.objects), header)


def _rgba(r, g, b, a=1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _pt(v: Iterable[float]) -> Point:
    x, y, z = (float(t) for t in v)
    return Point(x=x, y=y, z=z)


def to_markers(result: GraspResult, header, *, normal_len_m: float = 0.04, axis_len_m: float = 0.025) -> MarkerArray:
    """contact / normal / approach / pregrasp / tcp triad / rank text / selected. Every frame starts with a
    DELETEALL so a candidate that vanished does not linger in RViz. Rejections are drawn by the offline overlay
    (`grasp.overlay`) and listed on the diagnostics sidecar: a rejected region has no 3-D grasp to draw."""
    import numpy as np
    out = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    out.markers.append(clear)
    mid = 0

    def marker(ns: str, kind: int, color: ColorRGBA) -> Marker:
        nonlocal mid
        m = Marker()
        m.header = header
        m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
        mid += 1
        m.pose.orientation.w = 1.0
        m.color = color
        out.markers.append(m)
        return m

    for c in result.candidates:
        p, n, a = np.asarray(c.point), np.asarray(c.normal), np.asarray(c.approach)
        pg = np.asarray(c.pregrasp)
        s = marker('contact', Marker.SPHERE, _rgba(1, 1, 1))
        s.pose.position = _pt(p)
        s.scale.x = s.scale.y = s.scale.z = 0.008
        an = marker('normal', Marker.ARROW, _rgba(0.1, 0.9, 0.1))
        an.points = [_pt(p), _pt(p + normal_len_m * n)]
        an.scale.x, an.scale.y, an.scale.z = 0.003, 0.006, 0.008
        aa = marker('approach', Marker.ARROW, _rgba(0.15, 0.35, 1.0))
        aa.points = [_pt(pg), _pt(p)]
        aa.scale.x, aa.scale.y, aa.scale.z = 0.003, 0.006, 0.008
        sp = marker('pregrasp', Marker.SPHERE, _rgba(0.15, 0.35, 1.0, 0.8))
        sp.pose.position = _pt(pg)
        sp.scale.x = sp.scale.y = sp.scale.z = 0.006
        qx, qy, qz, qw = c.orientation_xyzw
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        for k, col in enumerate((_rgba(1, 0, 0), _rgba(0, 1, 0), _rgba(0, 0, 1))):
            t = marker('tcp', Marker.ARROW, col)
            t.points = [_pt(p), _pt(p + axis_len_m * R[:, k])]
            t.scale.x, t.scale.y, t.scale.z = 0.002, 0.004, 0.005
        tx = marker('rank', Marker.TEXT_VIEW_FACING, _rgba(1, 1, 1))
        tx.pose.position = _pt(p + 0.02 * n)
        tx.scale.z = 0.012
        tx.text = f'{c.rank}: {c.score:.2f}'
        if c.rank == 0:
            sel = marker('selected', Marker.SPHERE, _rgba(1.0, 0.85, 0.0, 0.5))
            sel.pose.position = _pt(p)
            sel.scale.x = sel.scale.y = sel.scale.z = 0.02
    return out
