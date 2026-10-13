"""detector_node, RUN: a real node, real topics, a real Trigger -- the contract as published.

vNext M0D T1. Everything else in this package tests the pure functions the node calls. What only a
running node can show is the CONTRACT the rest of the cell consumes (ADR-0016 Decision 1):

  * /perception/<name>/detections goes out BEST FIRST under the configured rule -- here a frame
    whose detector order (largest blob first) and rank order (most solid first) disagree, so a
    node that published in the detector's own order would fail;
  * /perception/<name>/quality carries the terms in the SAME permutation, paired by stamp;
  * /perception/<name>/detect answers with what it published.

The frame is test_ranked_publish.two_blobs: a 20x20 solid square (score 1,00) and a 22x22 square
with an 8x8 hole (larger, score 0,87).
"""
import json
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
from vision_msgs.msg import Detection3DArray  # noqa: E402

W, H, Z, F = 848, 480, 0.473, 424.0
NS = '/dn_live_test'
BEST_EFFORT = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=2)
RELIABLE = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=2)


def _two_blobs():
    rgb = np.full((H, W, 3), (160, 163, 170), np.uint8)
    rgb[100:120, 100:120] = (140, 13, 15)                     # A: 20x20 filled  -> 400 px
    rgb[300:322, 300:322] = (140, 13, 15)                     # B: 22x22 ...
    rgb[305:313, 305:313] = (160, 163, 170)                   # less an 8x8 hole -> 420 px
    return rgb, np.full((H, W), Z, np.float32)


def _spin_until(executor, done, timeout_s=10.0):
    end = time.monotonic() + timeout_s
    while not done() and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)
    return done()


def test_the_node_publishes_BEST_FIRST_and_the_terms_in_the_same_permutation(tmp_path):
    from cv_bridge import CvBridge
    from trainit_perception.nodes.detector_node import DetectorNode
    cfg = tmp_path / 'perception.yaml'
    cfg.write_text(yaml.safe_dump({'detectors': {'cubes': {
        'method': 'color_mask', 'continuous': False,
        'input': {'rgb': f'{NS}/rgb', 'depth': f'{NS}/depth', 'camera_info': f'{NS}/info'},
        'params': {'max_objects': 2, 'morph_kernel': 0, 'candidate_rule': 'score'}}}}))
    rclpy.init(args=['--ros-args', '-p', f'config_file:={cfg}', '-p', 'detector_name:=cubes'])
    try:
        node = DetectorNode()
        peer = rclpy.create_node('dn_live_test_peer')
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        ex.add_node(peer)
        got = {}
        peer.create_subscription(Detection3DArray, '/perception/cubes/detections',
                                 lambda m: got.__setitem__('dets', m), RELIABLE)
        peer.create_subscription(String, '/perception/cubes/quality',
                                 lambda m: got.__setitem__('quality', json.loads(m.data)), RELIABLE)
        p_info = peer.create_publisher(CameraInfo, f'{NS}/info', RELIABLE)
        p_rgb = peer.create_publisher(Image, f'{NS}/rgb', BEST_EFFORT)
        p_depth = peer.create_publisher(Image, f'{NS}/depth', BEST_EFFORT)
        info = CameraInfo(width=W, height=H)
        info.header.frame_id = 'cam_optical'
        info.k = [F, 0.0, 424.0, 0.0, F, 240.0, 0.0, 0.0, 1.0]
        rgb, depth = _two_blobs()
        bridge = CvBridge()

        def feed():
            stamp = peer.get_clock().now().to_msg()
            m_rgb = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
            m_depth = bridge.cv2_to_imgmsg(depth, encoding='32FC1')
            for m in (m_rgb, m_depth):
                m.header.stamp, m.header.frame_id = stamp, 'cam_optical'
            info.header.stamp = stamp
            p_info.publish(info)
            p_rgb.publish(m_rgb)
            p_depth.publish(m_depth)

        def fed():
            feed()
            return node.latest is not None

        assert _spin_until(ex, fed), 'the node never received an image pair'
        cli = peer.create_client(Trigger, '/perception/cubes/detect')
        assert cli.wait_for_service(timeout_sec=5.0)
        fut = cli.call_async(Trigger.Request())
        assert _spin_until(ex, fut.done)
        assert fut.result().success, fut.result().message
        assert _spin_until(ex, lambda: 'dets' in got and 'quality' in got)

        dets, quality = got['dets'], got['quality']
        scores = [d.results[0].hypothesis.score for d in dets.detections]
        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0) and scores[0] > scores[1]     # BEST FIRST
        # the sidecar: same stamp, same permutation, the terms of candidate i are candidate i's
        assert (quality['stamp']['sec'], quality['stamp']['nanosec']) == (
            dets.header.stamp.sec, dets.header.stamp.nanosec)
        assert [row['score'] for row in quality['detections']] == pytest.approx(scores)
        assert 'score 1.00' in fut.result().message                        # what it published
    finally:
        rclpy.shutdown()
