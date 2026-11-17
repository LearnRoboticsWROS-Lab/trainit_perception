"""surface_grasp_node with `segmentation: instance_topic` and an `eligibility` ROI, RUN (vNext M4.5).

A fake external segmenter (this test's peer) publishes /perception/<source>/instances (16UC1) and
/perception/<source>/instance_info (Detection2DArray) for the scene's three discs, stamped with the colour frame.
What only a running node shows:
  * a capture is served from the first frame stamped at/after the request THAT HAS ITS INSTANCES: a frame whose
    instances have not arrived is held, not processed, and is served the moment they arrive (exact-stamp pairing);
  * the instances become regions (class and score from instance_info), the geometry rejects as before;
  * an eligibility polygon keeps regions OUT of the provider: listed `ineligible` (OUTSIDE_WORKSPACE), never rejected;
  * an instance image of the wrong size is skipped with a warning and does NOT consume the capture;
  * pairing is by EXACT stamp: instances 1 ns or one frame off never pair, a stale instance of frame N arriving
    while a capture waits on N+1 serves nothing, an image and an info with different stamps never pair;
  * a capture that never resolves EXPIRES (CAPTURE_TTL_S, monotonic) and is never served late; a capture whose
    frame fails (an image of another size) is consumed, publishes nothing, and is counted, never silent;
  * the stamps going BACK (a sim-time / bag restart) clear the stores: the pairing resumes, it does not stall;
  * an UNMEASURABLE frame (regions, but a blind depth frame) is never an empty bin: no candidates, no objects, the
    capture not served (kept pending, counted `captures.unmeasurable`), then served by the next measurable frame.
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
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from trainit_interfaces.msg import GraspCandidateArray  # noqa: E402
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3DArray, ObjectHypothesisWithPose  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from surface_grasp_scenes import BASE, D455_LIKE_D, H, K, W, Disc, distorted, render, tilted  # noqa: E402

NS = '/sgi_live_test'
BEST_EFFORT = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=2)
RELIABLE = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=2)
DISCS = [Disc((-0.07, 0.0, 0.5), (0, 0, -1), 0.021), Disc((0.07, 0.0, 0.5), tilted(30.0), 0.021),
         Disc((0.0, 0.07, 0.5), tilted(60.0, 'x'), 0.021)]
LEFT_HALF = [[0, 0], [160, 0], [160, H - 1], [0, H - 1]]          # only the first disc (u ~ 118) lies inside


def _spin_until(executor, done, timeout_s=10.0):
    end = time.monotonic() + timeout_s
    while not done() and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)
    return done()


class Rig:
    def __init__(self, tmp_path, eligibility=None, name='parts'):
        from cv_bridge import CvBridge
        from trainit_perception.nodes.surface_grasp_node import SurfaceGraspNode
        entry = {'method': 'surface_grasp', 'continuous': False,
                 'input': {'rgb': f'{NS}/rgb', 'depth': f'{NS}/depth', 'camera_info': f'{NS}/info'},
                 'segmentation': {'method': 'instance_topic',
                                  'params': {'source': 'fakeseg', 'classes': ['capsule'], 'min_area_px': 30}},
                 'params': dict(BASE), 'publish_markers': False}
        if eligibility is not None:
            entry['eligibility'] = eligibility
        cfg = tmp_path / 'perception.yaml'
        cfg.write_text(yaml.safe_dump({'detectors': {name: entry}}))
        rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', f'detector_name:={name}'])
        self.node = SurfaceGraspNode()
        self.peer = rclpy.create_node('sgi_live_test_peer')
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.ex.add_node(self.peer)
        self.got = {}
        base = f'/perception/{name}'
        self.peer.create_subscription(GraspCandidateArray, f'{base}/grasp_candidates',
                                      lambda m: self.got.__setitem__('cands', m), RELIABLE)
        self.peer.create_subscription(String, f'{base}/grasp_diagnostics',
                                      lambda m: self.got.__setitem__('diag', json.loads(m.data)), RELIABLE)
        self.peer.create_subscription(Detection3DArray, f'{base}/objects',
                                      lambda m: self.got.__setitem__('objects', m), RELIABLE)
        self.p_info = self.peer.create_publisher(CameraInfo, f'{NS}/info', RELIABLE)
        self.p_rgb = self.peer.create_publisher(Image, f'{NS}/rgb', BEST_EFFORT)
        self.p_depth = self.peer.create_publisher(Image, f'{NS}/depth', BEST_EFFORT)
        self.p_inst = self.peer.create_publisher(Image, '/perception/fakeseg/instances', RELIABLE)
        self.p_iinfo = self.peer.create_publisher(Detection2DArray, '/perception/fakeseg/instance_info', RELIABLE)
        self.cli = self.peer.create_client(Trigger, f'{base}/detect')
        self.bridge = CvBridge()
        depth, masks = render(DISCS, distorted())
        self.depth = depth
        self.rgb = np.full((H, W, 3), (160, 163, 170), np.uint8)
        self.instances = np.zeros((H, W), np.uint16)
        for k, m in enumerate(masks, start=1):
            self.instances[m] = k
        self.info = CameraInfo(width=W, height=H, distortion_model='plumb_bob')
        self.info.header.frame_id = 'cam_optical'
        self.info.k = list(K.ravel())
        self.info.d = list(D455_LIKE_D)

    def feed_frame(self, stamp=None, shrink=False, depth=None):
        stamp = self.peer.get_clock().now().to_msg() if stamp is None else stamp
        rgb, depth = ((self.rgb[::2, ::2].copy(), self.depth[::2, ::2].copy()) if shrink
                      else (self.rgb, self.depth if depth is None else depth))
        m_rgb = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        m_depth = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        for m in (m_rgb, m_depth):
            m.header.stamp, m.header.frame_id = stamp, 'cam_optical'
        self.info.header.stamp = stamp
        self.p_info.publish(self.info)
        self.p_rgb.publish(m_rgb)
        self.p_depth.publish(m_depth)
        return stamp

    def feed_instances(self, stamp, image=None, info_stamp=None, halves=('image', 'info')):
        img = self.bridge.cv2_to_imgmsg(self.instances if image is None else image, encoding='16UC1')
        img.header.stamp, img.header.frame_id = stamp, 'cam_optical'
        arr = Detection2DArray()
        arr.header.stamp = stamp if info_stamp is None else info_stamp
        arr.header.frame_id = 'cam_optical'
        for k, score in ((1, 0.9), (2, 0.8), (3, 0.7)):
            d = Detection2D()
            d.header = arr.header
            d.id = str(k)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = 'capsule', score
            d.results.append(hyp)
            arr.detections.append(d)
        if 'image' in halves:
            self.p_inst.publish(img)
        if 'info' in halves:
            self.p_iinfo.publish(arr)

    def trigger(self):
        assert self.cli.wait_for_service(timeout_sec=5.0)
        fut = self.cli.call_async(Trigger.Request())
        assert _spin_until(self.ex, fut.done)
        return fut.result()

    def settle(self, seconds=0.3):
        _spin_until(self.ex, lambda: False, timeout_s=seconds)

    def stamp_of(self, key='cands'):
        h = self.got[key].header
        return h.stamp.sec, h.stamp.nanosec

    def ready(self):
        def fed():
            self.feed_frame()
            return self.node.latest is not None and self.node.camera is not None
        assert _spin_until(self.ex, fed), 'the node never received an image pair'


def test_a_capture_waits_for_its_instances_and_is_served_from_them(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        old = rig.feed_frame()
        rig.feed_instances(old)                                    # a complete frame BEFORE the request
        _spin_until(rig.ex, lambda: False, timeout_s=0.3)
        res = rig.trigger()
        assert res.success and 'that has its instances' in res.message, res.message
        _spin_until(rig.ex, lambda: 'cands' in rig.got, timeout_s=0.3)
        assert 'cands' not in rig.got                              # never served from the older frame
        stamp = rig.feed_frame()                                   # a fresh frame WITHOUT its instances: held
        _spin_until(rig.ex, lambda: 'cands' in rig.got, timeout_s=0.5)
        assert 'cands' not in rig.got
        rig.feed_instances(stamp)                                  # its instances arrive: served from that frame
        assert _spin_until(rig.ex, lambda: {'cands', 'diag'} <= set(rig.got))
        arr, diag = rig.got['cands'], rig.got['diag']
        assert (arr.header.stamp.sec, arr.header.stamp.nanosec) == (stamp.sec, stamp.nanosec)
        assert arr.header.frame_id == 'cam_optical'
        assert len(arr.candidates) == 2 and [r['code'] for r in diag['rejected']] == ['APPROACH_OUTSIDE_CONE']
        assert all(c.object_class == 'capsule' for c in arr.candidates)
        first = arr.candidates[0]
        assert np.linalg.norm(np.array([first.grasp_pose.position.x, first.grasp_pose.position.y,
                                        first.grasp_pose.position.z]) - DISCS[0].centre) < 1e-3
        seg = diag['segmentation']
        assert (seg['method'], seg['source'], seg['instances'], seg['kept']) == ('instance_topic', 'fakeseg', 3, 3)
        assert seg['pairing']['paired'] >= 2
        assert diag['ineligible'] == [] and diag['summary']['ineligible'] == 0
    finally:
        rclpy.shutdown()


def test_the_eligibility_polygon_keeps_regions_out_of_the_provider(tmp_path):
    try:
        rig = Rig(tmp_path, eligibility={'roi_px': LEFT_HALF, 'min_inside': 0.9})
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame()
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: {'cands', 'diag'} <= set(rig.got))
        diag = rig.got['diag']
        assert len(rig.got['cands'].candidates) == 1 and diag['rejected'] == []
        assert diag['summary']['regions'] == 1 and diag['summary']['ineligible'] == 2
        assert [e['code'] for e in diag['ineligible']] == ['OUTSIDE_WORKSPACE'] * 2
        assert all(e['class_id'] == 'capsule' and e['inside_fraction'] < 0.9 for e in diag['ineligible'])
    finally:
        rclpy.shutdown()


def test_a_polygon_off_the_image_refuses_the_capture(tmp_path):
    try:
        rig = Rig(tmp_path, eligibility={'roi_px': [[0, 0], [W + 50, 0], [W + 50, H], [0, H]]})
        rig.ready()
        res = rig.trigger()
        assert not res.success and 'eligibility refused' in res.message and 'outside' in res.message
    finally:
        rclpy.shutdown()


def test_an_instance_image_of_another_size_is_skipped_and_the_capture_stays_pending(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame()
        rig.feed_instances(stamp, image=np.zeros((H // 2, W // 2), np.uint16))
        _spin_until(rig.ex, lambda: 'cands' in rig.got, timeout_s=0.5)
        assert 'cands' not in rig.got and rig.node.pair_counts['skipped'] == 1
        assert rig.node.pending_ns is not None                     # not consumed by the skipped frame
        stamp = rig.feed_frame()
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: 'cands' in rig.got)
        assert (rig.got['cands'].header.stamp.sec, rig.got['cands'].header.stamp.nanosec) == (stamp.sec,
                                                                                              stamp.nanosec)
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize('entry, err, word', [
    ({'segmentation': {'method': 'instance_topic', 'params': {}}}, 'ConfigError', 'source is required'),
    ({'segmentation': {'method': 'instance_topic', 'params': {'source': 's'}},
      'eligibility': {'roi_px': [[0, 0], [1, 0]]}}, 'ConfigError', '>= 3'),
])
def test_a_bad_instance_or_eligibility_entry_stops_the_node(tmp_path, entry, err, word):
    from trainit_perception.grasp import ConfigError
    from trainit_perception.nodes.surface_grasp_node import SurfaceGraspNode
    cfg = tmp_path / 'perception.yaml'
    cfg.write_text(yaml.safe_dump({'detectors': {'p': dict({'method': 'surface_grasp', 'params': dict(BASE)},
                                                           **entry)}}))
    rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', 'detector_name:=p'])
    try:
        with pytest.raises(ConfigError, match=word):
            SurfaceGraspNode()
    finally:
        rclpy.shutdown()


def _shift(stamp, ns):
    return Time(nanoseconds=stamp.sec * 10**9 + stamp.nanosec + ns).to_msg()


def test_instances_one_ns_or_one_frame_off_never_pair_and_the_capture_stays_pending(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame()
        rig.feed_instances(_shift(stamp, 1))                       # 1 ns late
        rig.feed_instances(_shift(stamp, -1))                      # 1 ns early
        rig.feed_instances(_shift(stamp, 33_333_333))              # one 30 Hz frame off
        rig.settle(0.5)
        assert 'cands' not in rig.got and rig.node.pending_ns is not None
        assert rig.node.pair_counts['paired'] == 0
        rig.feed_instances(stamp)                                  # the exact stamp: served
        assert _spin_until(rig.ex, lambda: 'cands' in rig.got)
        assert rig.stamp_of() == (stamp.sec, stamp.nanosec) and rig.node.pending_ns is None
    finally:
        rclpy.shutdown()


def test_a_stale_instance_of_frame_n_serves_nothing_while_the_capture_waits_on_n_plus_1(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        old = rig.feed_frame()                                     # frame N, before the request, held
        rig.settle(0.2)
        assert rig.trigger().success
        new = rig.feed_frame()                                     # frame N+1, after it, held
        rig.settle(0.2)
        rig.feed_instances(old)                                    # N's instances arrive late: N pairs, too old
        rig.settle(0.5)
        assert 'cands' not in rig.got and rig.node.pending_ns is not None
        assert rig.node.pair_counts['paired'] == 1
        rig.feed_instances(new)
        assert _spin_until(rig.ex, lambda: 'cands' in rig.got)
        assert rig.stamp_of() == (new.sec, new.nanosec)
    finally:
        rclpy.shutdown()


def test_an_instance_image_and_info_with_different_stamps_never_pair(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame()
        rig.feed_instances(stamp, info_stamp=_shift(stamp, 1))     # the image on the frame, the info 1 ns off
        rig.settle(0.5)
        assert 'cands' not in rig.got and rig.node.pending_ns is not None and rig.node.pair_counts['paired'] == 0
        rig.feed_instances(stamp, halves=('info',))                # the matching info completes the pair
        assert _spin_until(rig.ex, lambda: 'cands' in rig.got)
        assert rig.stamp_of() == (stamp.sec, stamp.nanosec)
    finally:
        rclpy.shutdown()


def test_a_hold_that_never_resolves_expires_and_is_never_served_late(tmp_path, monkeypatch):
    from trainit_perception.nodes import surface_grasp_node as SGN
    monkeypatch.setattr(SGN, 'CAPTURE_TTL_S', 0.6)
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame()                                   # its instances do not come (yet)
        assert _spin_until(rig.ex, lambda: rig.node.pending_ns is None, timeout_s=3.0)   # the expiry timer
        assert rig.node.capture_counts == {'failed': 0, 'expired': 1, 'unmeasurable': 0}
        rig.feed_instances(stamp)                                  # far too late: the capture is gone
        rig.settle(0.5)
        assert 'cands' not in rig.got                              # the consumer timed out; nothing is served
        assert rig.node.pair_counts['paired'] == 1
        assert rig.trigger().success                               # a fresh capture is served, and says so
        stamp = rig.feed_frame()
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: {'cands', 'diag'} <= set(rig.got))
        assert rig.got['diag']['segmentation']['captures'] == {'failed': 0, 'expired': 1, 'unmeasurable': 0}
    finally:
        rclpy.shutdown()


def test_an_expired_capture_is_never_served_even_when_no_timer_ran(tmp_path, monkeypatch):
    from trainit_perception.nodes import surface_grasp_node as SGN
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        monkeypatch.setattr(SGN, 'CAPTURE_TTL_S', 0.0)             # already past its bound at the next frame
        time.sleep(0.01)
        stamp = rig.feed_frame()
        rig.feed_instances(stamp)
        rig.settle(0.5)
        assert 'cands' not in rig.got and rig.node.capture_counts['expired'] == 1
    finally:
        rclpy.shutdown()


def test_a_capture_consumed_by_a_failed_frame_publishes_nothing_and_is_counted(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame(shrink=True)                        # images of another size than the CameraInfo
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: rig.node.capture_counts['failed'] == 1, timeout_s=3.0)
        rig.settle(0.3)
        assert 'cands' not in rig.got and rig.node.pending_ns is None   # consumed, never silently
        stamp = rig.feed_frame()                                   # no capture pending: nothing is served
        rig.feed_instances(stamp)
        rig.settle(0.3)
        assert 'cands' not in rig.got
        assert rig.trigger().success
        stamp = rig.feed_frame()
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: {'cands', 'diag'} <= set(rig.got))
        assert rig.got['diag']['segmentation']['captures'] == {'failed': 1, 'expired': 0, 'unmeasurable': 0}
    finally:
        rclpy.shutdown()


def test_the_stamps_going_back_clear_the_stores_and_the_pairing_resumes(tmp_path):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        now = rig.peer.get_clock().now().to_msg()
        for k in range(3):                                         # a previous run, 100 s "later", held unpaired
            future = _shift(now, 100 * 10**9 + k * 33_333_333)
            rig.feed_frame(future)
            rig.feed_instances(future, halves=('image',))
        rig.settle(0.3)
        assert len(rig.node.frames) >= 1
        assert rig.trigger().success
        stamp = rig.feed_frame()                                   # the restarted clock: 100 s behind
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: 'cands' in rig.got), 'the pairing stalled on the stale stamps'
        assert rig.stamp_of() == (stamp.sec, stamp.nanosec)
        assert rig.node.pair_counts['clock_resets'] == 1
        assert all(ns <= stamp.sec * 10**9 + stamp.nanosec for ns in list(rig.node.frames) + list(rig.node.halves))
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize('blind', [0.0, float('nan')])
def test_an_unmeasurable_frame_is_never_an_empty_bin_and_does_not_serve_the_capture(tmp_path, blind):
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        stamp = rig.feed_frame(depth=np.full_like(rig.depth, blind))   # 3 instances, a blind depth frame
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: rig.node.capture_counts['unmeasurable'] == 1, timeout_s=3.0)
        rig.settle(0.5)
        assert not {'cands', 'objects', 'diag'} & set(rig.got)          # nothing at all: never an empty array
        assert rig.node.pair_counts['paired'] >= 1
        assert rig.node.pending_ns is not None                           # not served, not consumed
        assert rig.node.capture_counts['failed'] == 0
        stamp = rig.feed_frame()                                         # the depth comes back: served from it
        rig.feed_instances(stamp)
        assert _spin_until(rig.ex, lambda: {'cands', 'objects', 'diag'} <= set(rig.got))
        assert rig.stamp_of('objects') == (stamp.sec, stamp.nanosec) and len(rig.got['objects'].detections) == 3
        assert rig.node.pending_ns is None
        assert rig.got['diag']['segmentation']['captures'] == {'failed': 0, 'expired': 0, 'unmeasurable': 1}
    finally:
        rclpy.shutdown()


def test_a_persistently_blind_depth_lets_the_capture_expire_never_an_empty_answer(tmp_path, monkeypatch):
    from trainit_perception.nodes import surface_grasp_node as SGN
    monkeypatch.setattr(SGN, 'CAPTURE_TTL_S', 0.6)
    try:
        rig = Rig(tmp_path)
        rig.ready()
        assert rig.trigger().success
        blind = np.zeros_like(rig.depth)
        end = time.monotonic() + 1.5
        while time.monotonic() < end:                                    # blind frames past the capture's TTL
            stamp = rig.feed_frame(depth=blind)
            rig.feed_instances(stamp)
            rig.settle(0.1)
        assert rig.node.pending_ns is None and rig.node.capture_counts['expired'] == 1
        assert rig.node.capture_counts['unmeasurable'] >= 2 and rig.node.capture_counts['failed'] == 0
        assert not {'cands', 'objects', 'diag'} & set(rig.got)          # the consumer times out: a fault
    finally:
        rclpy.shutdown()
