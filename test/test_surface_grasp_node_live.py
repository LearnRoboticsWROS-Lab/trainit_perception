"""surface_grasp_node, RUN (vNext M4-A): a real node, real topics, a real Trigger -- the provider as published.

What only a running node shows:
  * /perception/<name>/grasp_candidates goes out BEST FIRST, in the observation's frame and stamp, every candidate
    valid by TMR's contract (unit approach and quaternion, eef / site / provider set);
  * /grasp_diagnostics pairs by stamp and lists the rejections with their codes; /grasp_markers is published;
  * the camera model follows the CameraInfo: a lens distortion is undistorted (the candidate lands on the true
    centre); a CameraInfo change rebuilds the model; a depth CameraInfo that differs from the colour one publishes
    NOTHING and the Trigger says why.
The scene: two red discs rendered exactly (surface_grasp_scenes), one square-on, one tilted 30 deg, and a third
tilted 60 deg (outside a 45 deg cone: rejected).
"""
import json
import os
import sys
import time

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip('rclpy')
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from trainit_interfaces.msg import GraspCandidateArray  # noqa: E402
from vision_msgs.msg import Detection3DArray  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from surface_grasp_scenes import (BASE, D455_LIKE_D, H, K, W, Disc, distorted, render,  # noqa: E402
                                  tilted)

NS = '/sg_live_test'
BEST_EFFORT = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=2)
RELIABLE = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=2)
DISCS = [Disc((-0.07, 0.0, 0.5), (0, 0, -1), 0.021), Disc((0.07, 0.0, 0.5), tilted(30.0), 0.021),
         Disc((0.0, 0.07, 0.5), tilted(60.0, 'x'), 0.021)]


def _spin_until(executor, done, timeout_s=10.0):
    end = time.monotonic() + timeout_s
    while not done() and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)
    return done()


def _scene():
    depth, masks = render(DISCS, distorted())
    rgb = np.full((H, W, 3), (160, 163, 170), np.uint8)
    for m in masks:
        rgb[m] = (200, 10, 15)
    return rgb, depth


class Rig:
    def __init__(self, tmp_path, name='parts', depth_info=False):
        from cv_bridge import CvBridge
        from trainit_perception.nodes.surface_grasp_node import SurfaceGraspNode
        inp = {'rgb': f'{NS}/rgb', 'depth': f'{NS}/depth', 'camera_info': f'{NS}/info'}
        if depth_info:
            inp['depth_camera_info'] = f'{NS}/dinfo'
        cfg = tmp_path / 'perception.yaml'
        cfg.write_text(yaml.safe_dump({'detectors': {name: {
            'method': 'surface_grasp', 'continuous': False, 'input': inp,
            'segmentation': {'method': 'color_mask', 'params': {'class_id': 'part', 'max_objects': 10,
                                                                'morph_kernel': 0, 'min_area_px': 30}},
            'params': dict(BASE), 'publish_markers': True}}}))
        rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', f'detector_name:={name}'])
        self.node = SurfaceGraspNode()
        self.peer = rclpy.create_node('sg_live_test_peer')
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.ex.add_node(self.peer)
        self.got = {}
        base = f'/perception/{name}'
        self.peer.create_subscription(GraspCandidateArray, f'{base}/grasp_candidates',
                                      lambda m: self.got.__setitem__('cands', m), RELIABLE)
        self.peer.create_subscription(String, f'{base}/grasp_diagnostics',
                                      lambda m: self.got.__setitem__('diag', json.loads(m.data)), RELIABLE)
        self.peer.create_subscription(MarkerArray, f'{base}/grasp_markers',
                                      lambda m: self.got.__setitem__('markers', m), RELIABLE)
        self.peer.create_subscription(Detection3DArray, f'{base}/objects',
                                      lambda m: self.got.__setitem__('objects', m), RELIABLE)
        self.p_info = self.peer.create_publisher(CameraInfo, f'{NS}/info', RELIABLE)
        self.p_dinfo = self.peer.create_publisher(CameraInfo, f'{NS}/dinfo', RELIABLE)
        self.p_rgb = self.peer.create_publisher(Image, f'{NS}/rgb', BEST_EFFORT)
        self.p_depth = self.peer.create_publisher(Image, f'{NS}/depth', BEST_EFFORT)
        self.cli = self.peer.create_client(Trigger, f'{base}/detect')
        self.bridge = CvBridge()
        self.rgb, self.depth = _scene()
        self.info = self._info(D455_LIKE_D)
        self.dinfo = None
        self.stamp = None

    @staticmethod
    def _info(d):
        info = CameraInfo(width=W, height=H, distortion_model='plumb_bob')
        info.header.frame_id = 'cam_optical'
        info.k = list(K.ravel())
        info.d = list(d)
        return info

    def feed(self):
        self.stamp = self.peer.get_clock().now().to_msg()
        m_rgb = self.bridge.cv2_to_imgmsg(self.rgb, encoding='rgb8')
        m_depth = self.bridge.cv2_to_imgmsg(self.depth, encoding='32FC1')
        for m in (m_rgb, m_depth):
            m.header.stamp, m.header.frame_id = self.stamp, 'cam_optical'
        self.info.header.stamp = self.stamp
        self.p_info.publish(self.info)
        if self.dinfo is not None:
            self.dinfo.header.stamp = self.stamp
            self.p_dinfo.publish(self.dinfo)
        self.p_rgb.publish(m_rgb)
        self.p_depth.publish(m_depth)

    def trigger(self):
        assert self.cli.wait_for_service(timeout_sec=5.0)
        fut = self.cli.call_async(Trigger.Request())
        assert _spin_until(self.ex, fut.done)
        return fut.result()

    def ready(self, cond=lambda n: n.camera is not None):
        def fed():
            self.feed()
            return self.node.latest is not None and cond(self.node)
        assert _spin_until(self.ex, fed), 'the node never received an image pair'


def test_the_node_publishes_ranked_valid_candidates_in_the_observation_frame(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        before = rig.stamp                                           # a frame the node already holds
        res = rig.trigger()
        assert res.success and 'scheduled' in res.message, res.message
        # a capture is served from the first frame stamped at/after the request -- never the held, older one
        _spin_until(rig.ex, lambda: 'cands' in rig.got, timeout_s=0.5)
        assert 'cands' not in rig.got
        assert _spin_until(rig.ex, lambda: (rig.feed(), {'cands', 'diag', 'markers', 'objects'} <= set(rig.got))[1])
        arr, diag = rig.got['cands'], rig.got['diag']
        assert (arr.header.stamp.sec, arr.header.stamp.nanosec) > (before.sec, before.nanosec)
        assert arr.header.frame_id == 'cam_optical'
        assert (arr.header.stamp.sec, arr.header.stamp.nanosec) == (diag['stamp']['sec'], diag['stamp']['nanosec'])
        assert len(arr.candidates) == 2                                   # the 60 deg disc is outside the cone
        assert [r['code'] for r in diag['rejected']] == ['APPROACH_OUTSIDE_CONE']
        scores = [c.score for c in arr.candidates]
        assert scores == sorted(scores, reverse=True)                     # BEST FIRST
        first = arr.candidates[0]
        # the square-on disc wins on alignment, and with the lens undistorted it lands on its true centre
        assert np.linalg.norm(np.array([first.grasp_pose.position.x, first.grasp_pose.position.y,
                                        first.grasp_pose.position.z]) - DISCS[0].centre) < 1e-3
        for c in arr.candidates:
            assert c.provider == 'tp.surface_grasp' and c.eef == 'main' and c.site == 'suction'
            assert c.header.frame_id == 'cam_optical' and c.header.stamp == arr.header.stamp
            a = np.array([c.approach.x, c.approach.y, c.approach.z])
            q = np.array([c.grasp_pose.orientation.x, c.grasp_pose.orientation.y, c.grasp_pose.orientation.z,
                          c.grasp_pose.orientation.w])
            assert abs(np.linalg.norm(a) - 1) < 1e-9 and abs(np.linalg.norm(q) - 1) < 1e-9
            assert c.standoff_m == 0.05 and 0 <= c.confidence <= 1
            md = {kv.key: kv.value for kv in c.metadata}
            assert md['normal_policy'] == 'toward_observer' and 'reachability' in md['not_evaluated']
        ns = {m.ns for m in rig.got['markers'].markers}
        assert {'contact', 'normal', 'approach', 'pregrasp', 'tcp', 'rank', 'selected'} <= ns
        # M4.5b: the objects PRESENT -- all three discs, the 60 deg one too (no grasp, still there), same header
        objs = rig.got['objects']
        assert objs.header == arr.header and len(objs.detections) == 3
        ids = {c.id for c in arr.candidates} | {r['id'] for r in diag['rejected']}
        assert {d.id for d in objs.detections} == ids
        for d in objs.detections:
            assert d.results[0].hypothesis.class_id == 'part' and d.bbox.size.x > 0.02
        p = objs.detections[[d.id for d in objs.detections].index(first.id)].results[0].pose.pose.position
        assert np.linalg.norm(np.array([p.x, p.y, p.z]) - DISCS[0].centre) < 2e-3
        assert diag['summary']['objects'] == 3 and diag['summary']['objects_without_depth'] == 0
        assert len(diag['objects']['published']) == 3
    finally:
        rclpy.shutdown()


def test_a_depth_camera_info_that_differs_publishes_nothing_and_says_why(tmp_path):
    try:
        rig = Rig(tmp_path, depth_info=True)
        rig.dinfo = Rig._info([0.0] * 5)                                # registered depth would carry the SAME D
        rig.ready(lambda n: n.depth_info is not None)
        res = rig.trigger()
        assert not res.success and 'not registered' in res.message
        _spin_until(rig.ex, lambda: 'cands' in rig.got, timeout_s=1.0)
        assert 'cands' not in rig.got
        rig.dinfo = Rig._info(D455_LIKE_D)                               # identical: accepted
        rig.ready(lambda n: n.camera_problem == '')
        assert rig.trigger().success
        assert _spin_until(rig.ex, lambda: (rig.feed(), 'cands' in rig.got)[1])
    finally:
        rclpy.shutdown()


def test_a_camera_info_change_rebuilds_the_camera_model(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        first = rig.node.camera
        assert first.distorted
        rig.info = Rig._info([0.0] * 5)
        rig.ready(lambda n: not n.camera.distorted)
        assert not rig.node.camera.same_as(first)
    finally:
        rclpy.shutdown()


def test_a_configuration_the_provider_refuses_stops_the_node(tmp_path):
    from trainit_perception.grasp import ConfigError
    from trainit_perception.nodes.surface_grasp_node import SurfaceGraspNode
    cfg = tmp_path / 'perception.yaml'
    params = dict(BASE)
    del params['approach_cone_deg']
    cfg.write_text(yaml.safe_dump({'detectors': {'p': {
        'method': 'surface_grasp', 'segmentation': {'method': 'color_mask'}, 'params': params}}}))
    rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', 'detector_name:=p'])
    try:
        with pytest.raises(ConfigError, match='approach_cone_deg'):
            SurfaceGraspNode()
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize('entry, word', [({'method': 'color_mask'}, 'this node serves'),
                                         ({'method': 'surface_grasp'}, 'segmentation.method'),
                                         ({'method': 'surface_grasp', 'segmentation': {'method': 'color_mask'},
                                           'colour': 1}, 'unknown key')])
def test_a_malformed_entry_is_refused_by_name(entry, word):
    from trainit_perception.nodes.surface_grasp_node import load_entry
    with pytest.raises(RuntimeError, match=word):
        load_entry({'detectors': {'p': entry}}, 'p')
