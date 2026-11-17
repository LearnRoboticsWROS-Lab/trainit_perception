#!/usr/bin/env python3
"""The SCRIPTED grasp provider node (`tp.scripted`, vNext M4.5): candidates from a scenario, for mock product tests.

It serves ONE `perception.yaml` entry whose `method` is `scripted_grasp` (the scenario format: `grasp/scripted.py`)
behind the SAME runtime surface as `surface_grasp_node`, so an application runs unchanged in mock and in Isaac:

    /perception/<name>/detect             std_srvs/Trigger                         each call = the NEXT step
    /perception/<name>/grasp_candidates   trainit_interfaces/GraspCandidateArray   RELIABLE, depth 1, in order
    /perception/<name>/grasp_diagnostics  std_msgs/String (JSON), paired by stamp  RELIABLE, depth 1
    /perception/<name>/objects            vision_msgs/Detection3DArray             RELIABLE, depth 1, PRESENCE (M4.5b)

A step with candidates (possibly none) answers success=true and publishes the array `delay_ms` later, stamped
`now + stamp_offset_ms` on the node clock (ROS time: `use_sim_time` is honoured), in the configured `frame_id`;
the publication always follows the Trigger's response. Publications are SERIALISED in Trigger order: a step never
overtakes an earlier one, so a step whose `delay_ms` is shorter than a still-pending earlier step's goes out right
after it (its delay is then `delay_ms` AT LEAST). A step `unavailable: true` answers success=false and
publishes nothing. Trigger-only: there is no continuous mode. With each array goes the step's OBJECTS PRESENT
(`objects`, the same header; `grasp/scripted.py`): one Detection3D per object, id `scripted_object@<k>`, class ''
(a scenario states no class), score 1, size 0 (unknown), identity orientation.

perception.yaml entry:

    detectors:
      part_grasp:
        method: scripted_grasp
        params: {scenario_file: /abs/scenario.yaml, frame_id: camera_color_optical_frame, after_end: empty}
        continuous: false            # optional; `true` is refused

The configuration AND the whole scenario are validated at start-up: a refusal (ConfigError) stops the node.
"""
from __future__ import annotations

import json
from collections import deque

import rclpy
import yaml
from diagnostic_msgs.msg import KeyValue
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from trainit_interfaces.msg import GraspCandidate, GraspCandidateArray
from vision_msgs.msg import Detection3DArray

from trainit_perception import __version__
from trainit_perception.grasp.scripted import (METHOD, PROVIDER_ID, Scenario, ScriptedConfig, diagnostics,
                                               load_scenario, object_id)
from trainit_perception.grasp_msgs import objects_array

RESULT_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
ENTRY_KEYS = ('method', 'params', 'continuous')


def load_entry(cfg: dict, name: str, path: str = 'perception.yaml') -> dict:
    """The provider's entry, validated for shape (its params by ScriptedConfig, its scenario by parse_scenario)."""
    try:
        entry = (cfg.get('detectors') or {})[name]
    except KeyError:
        raise RuntimeError(f'{path}: no detectors.{name} (have: {list(cfg.get("detectors") or {})})')
    if entry.get('method') != METHOD:
        raise RuntimeError(f'{path}: detectors.{name}.method is {entry.get("method")!r}; this node serves '
                           f'method: {METHOD}')
    stray = sorted(set(entry) - set(ENTRY_KEYS))
    if stray:
        raise RuntimeError(f'{path}: detectors.{name} has unknown key(s) {", ".join(stray)} (a scripted provider '
                           f'reads no camera: {", ".join(ENTRY_KEYS)})')
    if entry.get('continuous', False) is not False:
        raise RuntimeError(f'{path}: detectors.{name}.continuous must be false: a scripted provider answers '
                           f'the capture Trigger only')
    return entry


def to_candidate_array(step, header) -> GraspCandidateArray:
    arr = GraspCandidateArray()
    arr.header = header
    for rank, c in enumerate(step.candidates):
        m = GraspCandidate()
        m.header = header
        m.id, m.provider = c.id, PROVIDER_ID
        m.grasp_pose.position.x, m.grasp_pose.position.y, m.grasp_pose.position.z = c.position
        (m.grasp_pose.orientation.x, m.grasp_pose.orientation.y, m.grasp_pose.orientation.z,
         m.grasp_pose.orientation.w) = c.orientation_xyzw
        m.approach = Vector3(x=c.approach[0], y=c.approach[1], z=c.approach[2])
        m.standoff_m = c.standoff_m
        m.eef, m.site = c.eef, c.site
        m.score, m.confidence = c.score, c.confidence
        m.object_class, m.object_id = c.object_class, ''
        m.metadata = [KeyValue(key='rank', value=str(rank)), KeyValue(key='scripted', value='true'),
                      KeyValue(key='tp_version', value=__version__)]
        arr.candidates.append(m)
    return arr


def to_objects(step, header) -> Detection3DArray:
    """A step's objects present -> the presence array (class '', score 1, size 0 = unknown)."""
    return objects_array(((object_id(k), '', 1.0, p, (0.0, 0.0, 0.0)) for k, p in enumerate(step.present())),
                         header)


class ScriptedGraspNode(Node):

    def __init__(self):
        super().__init__('scripted_grasp_node')
        self.declare_parameter('config_file', '')
        self.declare_parameter('detector_name', '')
        cfg_path = self.get_parameter('config_file').value
        name = self.get_parameter('detector_name').value
        if not cfg_path or not name:
            raise RuntimeError('scripted_grasp_node needs config_file:=<perception.yaml> and detector_name:=<key>')
        with open(cfg_path) as fh:
            entry = load_entry(yaml.safe_load(fh) or {}, name, cfg_path)
        self.name = name
        self.config = ScriptedConfig.from_params(entry.get('params') or {})      # refuses -> the node stops
        self.scenario = Scenario(load_scenario(self.config.scenario_file, self.config), self.config.after_end)
        base = f'/perception/{name}'
        self.pub = self.create_publisher(GraspCandidateArray, f'{base}/grasp_candidates', RESULT_QOS)
        self.pub_diag = self.create_publisher(String, f'{base}/grasp_diagnostics', RESULT_QOS)
        self.pub_objects = self.create_publisher(Detection3DArray, f'{base}/objects', RESULT_QOS)
        self.create_service(Trigger, f'{base}/detect', self._on_trigger)
        # steps accepted but not yet published, in Trigger order: (due ns on the node clock, index, step, n)
        self.queue = deque()
        self.timer = None
        self.get_logger().info(
            f"scripted grasp '{name}' ({PROVIDER_ID}): {len(self.scenario.steps)} step(s) from "
            f"{self.config.scenario_file}, frame {self.config.frame_id}, after the end: {self.config.after_end} "
            f"-> {base}/grasp_candidates (on demand)")

    def _on_trigger(self, _req, resp):
        try:
            index, step = self.scenario.next()
            n = len(self.scenario.steps)
            if step.unavailable:
                resp.success, resp.message = False, f'scripted: perception unavailable (step {index} of {n})'
                return resp
            # always AFTER the response (delay_ms, at least 1 ms, on the node clock) and in Trigger order
            delay_ns = max(int(step.delay_ms * 1e6), 1_000_000)
            self.queue.append((self.get_clock().now().nanoseconds + delay_ns, index, step, n))
            if self.timer is None:
                self._arm()
            resp.success = True
            resp.message = (f'scripted: step {index} of {n}, {len(step.candidates)} candidate(s), '
                            f'{len(step.present())} object(s) present in '
                            f'{step.delay_ms:g} ms')
        except Exception as exc:                                 # noqa: BLE001 -- never out of a callback
            resp.success, resp.message = False, f'scripted: {type(exc).__name__}: {exc}'
        return resp

    def _arm(self):
        """One timer at a time, for the HEAD of the queue: the queue drains in order."""
        due_ns = self.queue[0][0]
        wait_ns = max(due_ns - self.get_clock().now().nanoseconds, 1_000_000)
        self.timer = self.create_timer(wait_ns * 1e-9, self._fire)

    def _fire(self):
        try:
            if self.timer is not None:
                self.timer.cancel()
                self.destroy_timer(self.timer)
                self.timer = None
            now_ns = self.get_clock().now().nanoseconds
            while self.queue and self.queue[0][0] <= now_ns:      # every head that is due, in order
                _, index, step, n = self.queue.popleft()
                self._publish(index, step, n)
            if self.queue:
                self._arm()
        except Exception as exc:                                 # noqa: BLE001 -- never out of a callback
            self.get_logger().error(f'scripted publication failed: {type(exc).__name__}: {exc}')

    def _publish(self, index, step, n):
        try:
            now = self.get_clock().now()
            ns = max(0, now.nanoseconds + int(step.stamp_offset_ms * 1e6))     # a stale stamp never goes negative
            stamp = Time(nanoseconds=ns, clock_type=now.clock_type).to_msg()
            header = Header(stamp=stamp, frame_id=self.config.frame_id)
            self.pub.publish(to_candidate_array(step, header))
            self.pub_objects.publish(to_objects(step, header))
            self.pub_diag.publish(String(data=json.dumps(diagnostics(index, step, n, header.frame_id, stamp.sec,
                                                                     stamp.nanosec))))
        except Exception as exc:                                 # noqa: BLE001 -- never out of a callback
            self.get_logger().error(f'scripted step {index} not published: {type(exc).__name__}: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedGraspNode()
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
