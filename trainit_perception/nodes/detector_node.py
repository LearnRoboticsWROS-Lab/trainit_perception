#!/usr/bin/env python3
"""The one ROS node of trainit_perception.

It loads ONE detector by name from a perception.yaml (the file the Setup Assistant
generates into the bundle's _app/config), subscribes to the camera contract, runs the
detector, and publishes the perception contract:

    /perception/<name>/detections   vision_msgs/Detection3DArray   RELIABLE, depth 1
    /perception/<name>/mask         sensor_msgs/Image (mono8)      BEST_EFFORT, debug
    /perception/<name>/detect       std_srvs/Trigger               one detection on demand

Continuous by default: a colour mask at camera rate costs nothing, and the consumer
(TMR's DetectObject) waits for a message FRESHER than its own tick, so it never acts
on a stale result. Set ``continuous: false`` for expensive detectors and use the
service. Both paths run the same ``detect()``.

perception.yaml shape (one node serves one entry of ``detectors``):

    detectors:
      cube:
        method: color_mask
        input:  {rgb: /camera/color/image_raw, depth: /camera/depth/image_rect_raw,
                 camera_info: /camera/color/camera_info}
        params: {class_id: cube, h: [170, 10], s: [120, 255], v: [60, 255], min_area_px: 60}
        continuous: true
        rate_hz: 10.0
"""
from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

from trainit_perception.detectors import depth_to_metres, make_detector
from trainit_perception.msgs import to_detection3d_array

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=2)
INFO_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST, depth=2)
# Detections are EVENTS, not a stream: a consumer must not miss the one it waited for.
DETECTION_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                           history=QoSHistoryPolicy.KEEP_LAST, depth=1)


class DetectorNode(Node):

    def __init__(self):
        super().__init__('detector_node')
        self.declare_parameter('config_file', '')
        self.declare_parameter('detector_name', '')
        cfg_path = self.get_parameter('config_file').value
        name = self.get_parameter('detector_name').value
        if not cfg_path or not name:
            raise RuntimeError('detector_node needs config_file:=<perception.yaml> and detector_name:=<key>')
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        try:
            entry = cfg['detectors'][name]
        except KeyError:
            raise RuntimeError(f'{cfg_path}: no detectors.{name} (have: {list((cfg.get("detectors") or {}))})')

        self.name = name
        self.detector = make_detector(entry['method'], entry.get('params') or {})
        inp = entry.get('input') or {}
        rgb_t = inp.get('rgb', '/camera/color/image_raw')
        depth_t = inp.get('depth', '/camera/depth/image_rect_raw')
        info_t = inp.get('camera_info', '/camera/color/camera_info')
        self.continuous = bool(entry.get('continuous', True))
        self.min_period = 1.0 / float(entry.get('rate_hz', 10.0))
        self.publish_mask = bool(entry.get('publish_mask', True))

        self.bridge = CvBridge()
        self.K = None
        self.frame_id = ''
        self.latest = None                     # (rgb, depth_m, stamp), guarded by lock
        self.lock = threading.Lock()
        self.last_pub = 0.0

        self.create_subscription(CameraInfo, info_t, self._on_info, INFO_QOS)
        self.sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, rgb_t, qos_profile=SENSOR_QOS),
             Subscriber(self, Image, depth_t, qos_profile=SENSOR_QOS)],
            queue_size=5, slop=0.05)
        self.sync.registerCallback(self._on_pair)

        base = f'/perception/{name}'
        self.pub = self.create_publisher(Detection3DArray, f'{base}/detections', DETECTION_QOS)
        self.pub_mask = (self.create_publisher(Image, f'{base}/mask', SENSOR_QOS)
                         if self.publish_mask else None)
        self.create_service(Trigger, f'{base}/detect', self._on_trigger)

        self.get_logger().info(
            f"detector '{name}' method={entry['method']} params={self.detector.params} "
            f"-> {base}/detections  ({'continuous' if self.continuous else 'on demand'})")

    # --- inputs ---------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.frame_id = msg.header.frame_id
            self.get_logger().info(f'intrinsics: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} '
                                   f'cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}  frame={self.frame_id}')

    def _on_pair(self, rgb_msg: Image, depth_msg: Image):
        if self.K is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)          # 32FC1 or 16UC1, as published
        depth_m = depth_to_metres(depth, depth_msg.encoding)
        with self.lock:
            self.latest = (rgb, depth_m, rgb_msg.header.stamp)
        if self.continuous and (time.monotonic() - self.last_pub) >= self.min_period:
            self._run_and_publish()

    # --- the one code path -------------------------------------------------------------
    def _run_and_publish(self):
        with self.lock:
            if self.latest is None:
                return None
            rgb, depth_m, stamp = self.latest
        dets = self.detector.detect(rgb, depth_m, self.K)
        # frame: the OPTICAL frame of the colour camera. The consumer transforms.
        self.pub.publish(to_detection3d_array(dets, self.frame_id, stamp))
        if self.pub_mask is not None:
            m = self.detector.debug_mask(rgb, depth_m)
            if m is not None:
                out = self.bridge.cv2_to_imgmsg(m, encoding='mono8')
                out.header.stamp, out.header.frame_id = stamp, self.frame_id
                self.pub_mask.publish(out)
        self.last_pub = time.monotonic()
        return dets

    def _on_trigger(self, _req, resp):
        dets = self._run_and_publish()
        if dets is None:
            resp.success, resp.message = False, 'no image received yet'
            return resp
        resp.success = len(dets) > 0
        resp.message = f'{len(dets)} detection(s)' + (
            '' if not dets else
            f'; first {dets[0].class_id} at ({dets[0].position[0]:.3f}, {dets[0].position[1]:.3f}, '
            f'{dets[0].position[2]:.3f}) score {dets[0].score:.2f}')
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
