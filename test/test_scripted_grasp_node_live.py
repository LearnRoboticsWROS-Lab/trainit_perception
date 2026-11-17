"""scripted_grasp_node (`tp.scripted`, vNext M4.5), RUN: a real node, a real Trigger, the scenario stepped.

What only a running node shows: each Trigger advances ONE step; the array goes out on
/perception/<name>/grasp_candidates (RELIABLE, depth 1) in the configured frame, stamped now + stamp_offset_ms
(a negative offset = a stale array, older than the request); `unavailable` answers success=false and publishes
nothing; `delay_ms` delays the array, and arrays go out in Trigger order whatever their delays; after the end,
`empty` answers with an empty array; a broken scenario or a continuous entry stops the node at start-up.
M4.5b: each published step also publishes its objects PRESENT on /perception/<name>/objects, under the candidate
array's header -- by default one per candidate; explicit objects remain when the candidates run out.
"""
import json
import time

import pytest
import yaml

rclpy = pytest.importorskip('rclpy')
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from trainit_interfaces.msg import GraspCandidateArray  # noqa: E402
from vision_msgs.msg import Detection3DArray  # noqa: E402

RELIABLE = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
SCENARIO = {'steps': [
    {'candidates': [{'position': [0.0, 0.0, 0.5], 'approach': [0, 0, 1], 'standoff_m': 0.05, 'eef': 'main',
                     'site': 'suction', 'score': 0.9, 'confidence': 0.8, 'object_class': 'capsule'},
                    {'id': 'second', 'position': [0.05, 0.0, 0.5], 'approach': [0.2, 0, 1], 'standoff_m': 0.05,
                     'score': 0.5}]},
    {'candidates': [], 'delay_ms': 300},
    {'unavailable': True},
    {'candidates': [{'position': [0, 0, 0.4], 'approach': [0, 0, 1], 'standoff_m': 0.03}],
     'stamp_offset_ms': -5000},
]}


def _spin_until(executor, done, timeout_s=10.0):
    end = time.monotonic() + timeout_s
    while not done() and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)
    return done()


def _write(tmp_path, scenario=SCENARIO, **entry):
    sc = tmp_path / 'scenario.yaml'
    sc.write_text(yaml.safe_dump(scenario))
    cfg = tmp_path / 'perception.yaml'
    e = {'method': 'scripted_grasp', 'params': {'scenario_file': str(sc), 'frame_id': 'cam_optical',
                                                'after_end': 'empty'}}
    e.update(entry)
    cfg.write_text(yaml.safe_dump({'detectors': {'mock_grasp': e}}))
    return cfg


def test_each_trigger_plays_the_next_step(tmp_path):
    from trainit_perception.nodes.scripted_grasp_node import ScriptedGraspNode
    rclpy.init(args=['--ros-args', '-p', f'config_file:={_write(tmp_path)}', '-p', 'detector_name:=mock_grasp'])
    try:
        node = ScriptedGraspNode()
        peer = rclpy.create_node('scripted_live_peer')
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(peer)
        got = []
        diags = []
        peer.create_subscription(GraspCandidateArray, '/perception/mock_grasp/grasp_candidates', got.append,
                                 RELIABLE)
        peer.create_subscription(String, '/perception/mock_grasp/grasp_diagnostics',
                                 lambda m: diags.append(json.loads(m.data)), RELIABLE)
        cli = peer.create_client(Trigger, '/perception/mock_grasp/detect')
        assert cli.wait_for_service(timeout_sec=5.0)

        def trigger():
            fut = cli.call_async(Trigger.Request())
            assert _spin_until(ex, fut.done)
            return fut.result()

        _spin_until(ex, lambda: False, timeout_s=0.3)              # discovery
        before = node.get_clock().now().nanoseconds
        r = trigger()
        assert r.success and 'step 0 of 4' in r.message
        assert _spin_until(ex, lambda: len(got) == 1 and len(diags) == 1)
        arr = got[0]
        assert arr.header.frame_id == 'cam_optical'
        assert arr.header.stamp.sec * 10**9 + arr.header.stamp.nanosec >= before
        assert [c.id for c in arr.candidates] == ['scripted@0.0', 'second']
        c0, c1 = arr.candidates
        assert c0.provider == 'tp.scripted' and (c0.eef, c0.site, c0.object_class) == ('main', 'suction', 'capsule')
        assert (c0.score, c0.confidence, c0.standoff_m) == (0.9, 0.8, 0.05)
        assert (c0.grasp_pose.orientation.w, c1.approach.z) == (1.0, pytest.approx(1 / (1.04 ** 0.5)))
        assert diags[0]['provider'] == 'tp.scripted' and diags[0]['summary']['valid'] == 2

        t0 = time.monotonic()
        assert trigger().success                                     # step 1: empty, 300 ms later
        _spin_until(ex, lambda: len(got) == 2, timeout_s=0.15)
        assert len(got) == 1
        assert _spin_until(ex, lambda: len(got) == 2) and time.monotonic() - t0 >= 0.25
        assert got[1].candidates == []

        r = trigger()                                                # step 2: unavailable
        assert not r.success and 'unavailable' in r.message
        _spin_until(ex, lambda: len(got) == 3, timeout_s=0.4)
        assert len(got) == 2

        before = node.get_clock().now().nanoseconds
        assert trigger().success                                     # step 3: stale by 5 s
        assert _spin_until(ex, lambda: len(got) == 3)
        stamp = got[2].header.stamp.sec * 10**9 + got[2].header.stamp.nanosec
        assert stamp < before - 4 * 10**9

        assert trigger().success                                     # past the end: an empty answer
        assert _spin_until(ex, lambda: len(got) == 4 and len(diags) == 4) and got[3].candidates == []
        assert [d['scenario']['step'] for d in diags] == [0, 1, 3, 4]      # the unavailable step published nothing
        assert diags[-1]['scenario']['past_end'] is True
    finally:
        rclpy.shutdown()


def test_steps_are_published_in_trigger_order_whatever_their_delays(tmp_path):
    scenario = {'steps': [
        {'candidates': [{'id': 'slow', 'position': [0, 0, 0.5], 'approach': [0, 0, 1], 'standoff_m': 0.05}],
         'delay_ms': 400},
        {'candidates': [{'id': 'fast', 'position': [0, 0, 0.5], 'approach': [0, 0, 1], 'standoff_m': 0.05}],
         'delay_ms': 1},
        {'candidates': [], 'delay_ms': 50},
    ]}
    from trainit_perception.nodes.scripted_grasp_node import ScriptedGraspNode
    cfg = _write(tmp_path, scenario)
    rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', 'detector_name:=mock_grasp'])
    try:
        node = ScriptedGraspNode()
        peer = rclpy.create_node('scripted_order_peer')
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(peer)
        got = []
        diags = []
        depth10 = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        peer.create_subscription(GraspCandidateArray, '/perception/mock_grasp/grasp_candidates',
                                 lambda m: got.append((time.monotonic(), m)), depth10)
        peer.create_subscription(String, '/perception/mock_grasp/grasp_diagnostics',
                                 lambda m: diags.append(json.loads(m.data)['scenario']['step']), depth10)
        cli = peer.create_client(Trigger, '/perception/mock_grasp/detect')
        assert cli.wait_for_service(timeout_sec=5.0)
        _spin_until(ex, lambda: False, timeout_s=0.3)
        t0 = time.monotonic()
        for _ in range(3):                                         # three Triggers back to back
            fut = cli.call_async(Trigger.Request())
            assert _spin_until(ex, fut.done) and fut.result().success
        assert _spin_until(ex, lambda: len(got) == 3 and len(diags) == 3)
        assert [[c.id for c in m.candidates] for _, m in got] == [['slow'], ['fast'], []]
        assert diags == [0, 1, 2]
        assert got[0][0] - t0 >= 0.35                              # the slow step kept its own delay
        stamps = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec for _, m in got]
        assert stamps == sorted(stamps)
        assert len(node.queue) == 0 and node.timer is None
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize('scenario, entry, err, word', [
    ({'steps': [{'candidates': [{'position': [0, 0]}]}]}, {}, 'ConfigError', 'position must be three'),
    (SCENARIO, {'continuous': True}, 'RuntimeError', 'continuous must be false'),
    (SCENARIO, {'input': {}}, 'RuntimeError', 'unknown key'),
])
def test_a_broken_scenario_or_entry_stops_the_node(tmp_path, scenario, entry, err, word):
    from trainit_perception.grasp import ConfigError
    from trainit_perception.nodes.scripted_grasp_node import ScriptedGraspNode
    cfg = _write(tmp_path, scenario, **entry)
    rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', 'detector_name:=mock_grasp'])
    try:
        with pytest.raises({'ConfigError': ConfigError, 'RuntimeError': RuntimeError}[err], match=word):
            ScriptedGraspNode()
    finally:
        rclpy.shutdown()


def test_each_step_publishes_its_objects_present_under_the_arrays_header(tmp_path):
    scenario = {'steps': [
        {'candidates': [{'position': [0.0, 0.0, 0.5], 'approach': [0, 0, 1], 'standoff_m': 0.05},
                        {'position': [0.05, 0.0, 0.5], 'approach': [0, 0, 1], 'standoff_m': 0.05}]},
        {'candidates': [], 'objects': [[0.02, 0.01, 0.55]]},          # no grasp left, an object remains
        {'unavailable': True},
        {'candidates': []},                                           # nothing present: an empty bin
    ]}
    from trainit_perception.nodes.scripted_grasp_node import ScriptedGraspNode
    cfg = _write(tmp_path, scenario)
    rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', 'detector_name:=mock_grasp'])
    try:
        node = ScriptedGraspNode()
        peer = rclpy.create_node('scripted_objects_peer')
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(peer)
        got, objs = [], []
        depth10 = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        peer.create_subscription(GraspCandidateArray, '/perception/mock_grasp/grasp_candidates', got.append, depth10)
        peer.create_subscription(Detection3DArray, '/perception/mock_grasp/objects', objs.append, depth10)
        cli = peer.create_client(Trigger, '/perception/mock_grasp/detect')
        assert cli.wait_for_service(timeout_sec=5.0)
        _spin_until(ex, lambda: False, timeout_s=0.3)
        answers = []
        for _ in range(4):
            fut = cli.call_async(Trigger.Request())
            assert _spin_until(ex, fut.done)
            answers.append(fut.result())
        assert [a.success for a in answers] == [True, True, False, True]
        assert '2 candidate(s), 2 object(s) present' in answers[0].message
        assert _spin_until(ex, lambda: len(got) == 3 and len(objs) == 3)
        for arr, ob in zip(got, objs):
            assert ob.header.stamp == arr.header.stamp and ob.header.frame_id == arr.header.frame_id == 'cam_optical'
        assert [[d.id for d in o.detections] for o in objs] == [
            ['scripted_object@0', 'scripted_object@1'], ['scripted_object@0'], []]
        d = objs[0].detections[1]
        assert (d.results[0].pose.pose.position.x, d.results[0].pose.pose.position.z) == (0.05, 0.5)
        assert d.results[0].hypothesis.score == 1.0 and d.bbox.size.x == 0.0
        p = objs[1].detections[0].results[0].pose.pose.position
        assert (p.x, p.y, p.z) == (0.02, 0.01, 0.55) and got[1].candidates == []
    finally:
        rclpy.shutdown()
