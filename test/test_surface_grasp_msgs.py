"""SurfaceGrasp -> trainit_interfaces/GraspCandidateArray (vNext M4-A): the wire form TMR consumes.

The array keeps the provider's order (the rank), the observation's frame and stamp, and every field ADR-0026 names;
the result is valid by TMR's own checks (unit quaternion, unit approach, standoff >= 0, confidence 0..1, unique ids,
non-empty frame / id / provider) -- mirrored here so a producer bug fails in this package first.
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip('trainit_interfaces')
from std_msgs.msg import Header  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from surface_grasp_scenes import BASE, Disc, pinhole, regions, render, tilted  # noqa: E402

from trainit_perception.grasp import SurfaceGraspConfig, propose  # noqa: E402
from trainit_perception.grasp_msgs import to_candidate_array, to_diagnostics_json, to_markers  # noqa: E402


def tmr_problem(c):
    """TMR's candidateProblem (grasp_candidates.cpp), restated."""
    if not c.header.frame_id or not c.id or not c.provider:
        return 'frame / id / provider'
    q = c.grasp_pose.orientation
    if abs(math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2) - 1) > 1e-3:
        return 'quaternion'
    a = c.approach
    if abs(math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2) - 1) > 1e-3:
        return 'approach'
    if not (math.isfinite(c.standoff_m) and c.standoff_m >= 0):
        return 'standoff'
    if not (0 <= c.confidence <= 1) or not math.isfinite(c.score):
        return 'confidence / score'
    return ''


@pytest.fixture
def result():
    discs = [Disc((-0.07, 0.0, 0.5), tilted(20.0), 0.021), Disc((0.07, 0.0, 0.5), (0, 0, -1), 0.021),
             Disc((0.0, 0.07, 0.5), tilted(70.0, 'x'), 0.021)]
    depth, masks = render(discs)
    return propose(regions(discs, masks), depth, pinhole(), SurfaceGraspConfig.from_params(BASE))


def header():
    return Header(stamp=Time(sec=12, nanosec=345), frame_id='cam_optical')


def test_the_array_keeps_the_rank_frame_and_stamp(result):
    arr = to_candidate_array(result, header())
    assert arr.header.frame_id == 'cam_optical' and (arr.header.stamp.sec, arr.header.stamp.nanosec) == (12, 345)
    assert [c.id for c in arr.candidates] == [c.id for c in result.candidates]
    assert [c.score for c in arr.candidates] == [c.score for c in result.candidates]
    assert len({c.id for c in arr.candidates}) == len(arr.candidates)
    for m, c in zip(arr.candidates, result.candidates):
        assert tmr_problem(m) == ''
        assert m.header == arr.header
        assert (m.grasp_pose.position.x, m.grasp_pose.position.y, m.grasp_pose.position.z) == c.point
        assert (m.approach.x, m.approach.y, m.approach.z) == c.approach
        assert (m.surface_normal.x, m.surface_normal.y, m.surface_normal.z) == c.normal
        assert m.object_class == 'part' and m.object_id == ''
        md = {kv.key: kv.value for kv in m.metadata}
        assert md['rank'] == str(c.rank)
        assert float(md['term.alignment']) == pytest.approx(c.terms['alignment'], rel=1e-8)


def test_an_empty_result_is_an_empty_array_the_valid_no_candidate_answer():
    depth, _ = render([])
    res = propose([], depth, pinhole(), SurfaceGraspConfig.from_params(BASE))
    arr = to_candidate_array(res, header())
    assert arr.candidates == [] and arr.header.frame_id == 'cam_optical'


def test_the_diagnostics_pair_by_stamp(result):
    import json
    d = json.loads(to_diagnostics_json(result, header()))
    assert d['stamp'] == {'sec': 12, 'nanosec': 345} and d['frame_id'] == 'cam_optical'
    assert [c['id'] for c in d['candidates']] == [c.id for c in result.candidates]
    assert all(r['code'] for r in d['rejected'])


def test_the_markers_draw_every_candidate_and_start_with_a_clear(result):
    from visualization_msgs.msg import Marker
    ma = to_markers(result, header())
    assert ma.markers[0].action == Marker.DELETEALL
    n = len(result.candidates)
    count = {}
    for m in ma.markers[1:]:
        count[m.ns] = count.get(m.ns, 0) + 1
        assert m.header.frame_id == 'cam_optical'
    assert count['contact'] == count['normal'] == count['approach'] == count['pregrasp'] == n
    assert count['tcp'] == 3 * n and count['selected'] == 1
    # the approach arrow runs from the PreGrasp INTO the contact point
    arrow = next(m for m in ma.markers if m.ns == 'approach')
    c = result.candidates[0]
    assert np.allclose([arrow.points[1].x, arrow.points[1].y, arrow.points[1].z], c.point)
    assert np.allclose([arrow.points[0].x, arrow.points[0].y, arrow.points[0].z], c.pregrasp)
