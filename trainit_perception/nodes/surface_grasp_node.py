#!/usr/bin/env python3
"""The SurfaceGrasp provider node (`tp.surface_grasp`, vNext M4; design: the overlay's M4_DESIGN.md §15).

It loads ONE `perception.yaml` entry whose `method` is `surface_grasp`, subscribes to the camera contract (colour,
registered depth, colour CameraInfo; the depth CameraInfo optional and checked identical), segments each frame with
a registered TP detector that exposes per-detection masks, and publishes:

    /perception/<name>/grasp_candidates   trainit_interfaces/GraspCandidateArray   RELIABLE, depth 1, BEST FIRST
    /perception/<name>/grasp_diagnostics  std_msgs/String (JSON), paired by stamp  RELIABLE, depth 1, ADDITIVE
    /perception/<name>/grasp_markers      visualization_msgs/MarkerArray           RELIABLE, depth 1, optional debug
    /perception/<name>/objects            vision_msgs/Detection3DArray             RELIABLE, depth 1, PRESENCE (M4.5b)
    /perception/<name>/detect             std_srvs/Trigger                         one result on demand

Cell-global names, like detections (ADR-0023 Decision 4). The candidates' header is the observation: the colour
optical frame (from the CameraInfo) and the colour image's stamp.

perception.yaml entry:

    detectors:
      part_grasp:
        method: surface_grasp
        input:  {rgb: /camera/color/image_raw, depth: /camera/depth/image_rect_raw,
                 camera_info: /camera/color/camera_info, depth_camera_info: /camera/depth/camera_info}
        segmentation: {method: color_mask, params: {class_id: part, h: [170, 10], max_objects: 20}}
        eligibility: {roi_px: [[40, 30], [600, 30], [600, 450], [40, 450]], min_inside: 0.9}   # optional, 0.7.0
        params: {eef: main, site: suction, contact_radius_m: 0.009, approach_cone_deg: 45.0, standoff_m: 0.05}
        continuous: false
        rate_hz: 5.0
        publish_markers: true

A configuration the provider refuses stops the node at start-up (ConfigError), and a frame whose camera model is
refused publishes NOTHING (an error, throttled): never candidates computed through a wrong camera model.

SEGMENTATION (0.7.0): `color_mask` (or any registered TP detector exposing masks) runs in-process; `instance_topic`
reads an EXTERNAL segmenter (`grasp/instances.py`): /perception/<source>/instances (16UC1 / mono16, 0 = background,
k = instance k) + /perception/<source>/instance_info (Detection2DArray, id = str(k)), both stamped with the colour
image's stamp. The node pairs the two halves and the colour/depth frame by EXACT stamp; a frame waits for its
instances at most PAIR_WAIT_S of stamp time (and among the PAIR_BUFFER newest), then it is dropped and counted
(`segmentation.pairing` on the diagnostics); a frame without its instances is never processed. A capture is then
served from the first frame stamped at or after the request THAT HAS ITS INSTANCES. An instance image of another
size, frame or encoding than the contract is skipped with one warning (the capture stays pending). A colour FRAME
stamped more than PAIR_WAIT_S BEHIND the newest stamp held (a sim-time / bag restart: the clock went back) clears
both stores (warned, counted `pairing.clock_resets`), so the pairing never stalls on stale future stamps.

CAPTURE LIFETIME: a pending capture lives at most CAPTURE_TTL_S of monotonic (wall) time after its Trigger; then
it is dropped (an error, counted `captures.expired`) and never served, however late a matching frame arrives. A
capture whose frame fails for any reason other than a skipped instance pair or an unmeasurable frame (below) -- a
refused camera model, an image of another size, an exception -- is consumed: nothing is published, an error is
logged and it is counted `captures.failed`. The counters (`failed`, `expired`, `unmeasurable`) ride on every later
diagnostics record (`segmentation.captures`).

ELIGIBILITY (0.7.0, `grasp/eligibility.py`): the regions are filtered by the workspace polygon / depth band BEFORE
`propose`; an ineligible region is listed under `ineligible` (code OUTSIDE_WORKSPACE), never as a rejection.

PRESENCE (0.7.0, M4.5b, `grasp/objects.py`): with every candidate array (capture and continuous) the node publishes
`objects`: one Detection3D per ELIGIBLE region whatever its geometry verdict (an object present, not a grasp), under
the candidate array's own header (the same stamp and frame); counted on the diagnostics (`summary.objects`,
`summary.objects_without_depth`, the `objects` block). A presence capture is TMR's DetectObject on it.
AN UNMEASURABLE FRAME IS NEVER AN EMPTY BIN: a frame with >= 1 eligible region but no valid depth at all (a depth
dropout, a wrong depth band) publishes NOTHING -- no candidates, no objects, no diagnostics, no markers -- an error
is logged (throttled) and it is counted `captures.unmeasurable` (every such frame, captured or not). A pending
capture is NOT consumed by it (like a skipped instance pair): the next measurable frame serves it, and a blindness
that persists lets it expire (CAPTURE_TTL_S) -- the consumer times out (DETECTION_TIMEOUT, a fault), it is never
answered "no object". Kept pending rather than consumed so a one-frame dropout does not fail the capture. A frame
with NO eligible region is a genuine empty answer (an empty `objects` array).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from trainit_interfaces.msg import GraspCandidateArray
from vision_msgs.msg import Detection2DArray, Detection3DArray
from visualization_msgs.msg import MarkerArray

from trainit_perception.detectors import depth_to_metres, make_detector
from trainit_perception.grasp import (CameraModel, CameraModelError, SurfaceGraspConfig, propose,
                                      regions_from_detections)
from trainit_perception.grasp.eligibility import EligibilityConfig, split_eligible
from trainit_perception.grasp.instances import (ENCODINGS as INSTANCE_ENCODINGS, METHOD as INSTANCE_METHOD,
                                                InstanceConfig, InstanceContractError, InstanceInfo,
                                                parse_instance_id, regions_from_instance_image)
from trainit_perception.grasp.objects import present_objects
from trainit_perception.grasp_msgs import to_candidate_array, to_diagnostics_json, to_markers, to_objects_array
from trainit_perception.ranking.rules import PROVIDES_MASK_AREA

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=2)
INFO_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=2)
RESULT_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
# the external segmenter's halves: BEST_EFFORT matches a reliable or a best-effort publisher; a few frames deep
INSTANCE_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=5)

METHOD = 'surface_grasp'
ENTRY_KEYS = ('method', 'input', 'segmentation', 'eligibility', 'params', 'continuous', 'rate_hz',
              'publish_markers')
#: instance_topic pairing bounds: a frame / an instance half waits at most this long in STAMP time for its partner,
#: and at most this many of each are held; then it is dropped and counted
PAIR_WAIT_S = 2.0
PAIR_BUFFER = 10
#: a pending capture (a Trigger) is served only within this many seconds of MONOTONIC time after the request; it is
#: longer than any sensible consumer wait (TMR's DetectGraspCandidates waits 3 s by default), so a consumer that
#: has given up is never answered minutes later with an old capture
CAPTURE_TTL_S = 10.0
#: the reasons a frame due for a capture published nothing WITHOUT consuming it (it stays pending, TTL-bounded)
SKIPPED = 'skipped'
UNMEASURABLE = 'unmeasurable'


def load_entry(cfg: dict, name: str, path: str = 'perception.yaml') -> dict:
    """The provider's entry, validated for shape (its params are validated by SurfaceGraspConfig)."""
    try:
        entry = (cfg.get('detectors') or {})[name]
    except KeyError:
        raise RuntimeError(f'{path}: no detectors.{name} (have: {list(cfg.get("detectors") or {})})')
    if entry.get('method') != METHOD:
        raise RuntimeError(f'{path}: detectors.{name}.method is {entry.get("method")!r}; this node serves '
                           f'method: {METHOD}')
    stray = sorted(set(entry) - set(ENTRY_KEYS))
    if stray:
        raise RuntimeError(f'{path}: detectors.{name} has unknown key(s) {", ".join(stray)}')
    seg = entry.get('segmentation') or {}
    if not seg.get('method'):
        raise RuntimeError(f'{path}: detectors.{name}.segmentation.method is required (a TP detector whose '
                           f'detections carry masks, e.g. color_mask, or {INSTANCE_METHOD})')
    return entry


def make_eligibility(entry: dict):
    """The entry's workspace eligibility, validated (ConfigError), or None when it has none."""
    block = entry.get('eligibility')
    return None if block is None else EligibilityConfig.from_params(block)


def instance_infos(msg) -> list:
    """A `vision_msgs/Detection2DArray` (instance_info) -> InstanceInfo per detection. Raises
    InstanceContractError when an id is not a pixel value or a detection carries no hypothesis."""
    out = []
    for i, det in enumerate(msg.detections):
        k = parse_instance_id(det.id)
        if not det.results:
            raise InstanceContractError(f'instance_info detection {i} (id {det.id!r}) carries no hypothesis')
        hyp = det.results[0].hypothesis
        out.append(InstanceInfo(instance_id=k, class_id=str(hyp.class_id), score=float(hyp.score)))
    return out


def make_segmenter(seg: dict):
    """An in-process TP detector exposing masks, or -- for `instance_topic` -- the validated InstanceConfig."""
    if seg['method'] == INSTANCE_METHOD:
        return InstanceConfig.from_params(seg.get('params') or {})
    det = make_detector(seg['method'], seg.get('params') or {})
    if PROVIDES_MASK_AREA not in det.PROVIDES:
        raise RuntimeError(f'segmentation method {seg["method"]!r} provides no masks: a surface grasp needs the '
                           f'object\'s pixels')
    return det


class SurfaceGraspNode(Node):

    def __init__(self):
        super().__init__('surface_grasp_node')
        self.declare_parameter('config_file', '')
        self.declare_parameter('detector_name', '')
        cfg_path = self.get_parameter('config_file').value
        name = self.get_parameter('detector_name').value
        if not cfg_path or not name:
            raise RuntimeError('surface_grasp_node needs config_file:=<perception.yaml> and detector_name:=<key>')
        with open(cfg_path) as fh:
            entry = load_entry(yaml.safe_load(fh) or {}, name, cfg_path)
        self.name = name
        self.config = SurfaceGraspConfig.from_params(entry.get('params') or {})    # refuses -> the node stops
        self.segmenter = make_segmenter(entry['segmentation'])
        self.seg_method = entry['segmentation']['method']
        self.instance_mode = isinstance(self.segmenter, InstanceConfig)
        self.eligibility = make_eligibility(entry)                                  # refuses -> the node stops
        inp = entry.get('input') or {}
        rgb_t = inp.get('rgb', '/camera/color/image_raw')
        depth_t = inp.get('depth', '/camera/depth/image_rect_raw')
        info_t = inp.get('camera_info', '/camera/color/camera_info')
        dinfo_t = inp.get('depth_camera_info', '')
        self.continuous = bool(entry.get('continuous', True))
        self.min_period = 1.0 / float(entry.get('rate_hz', 5.0))
        self.publish_markers = bool(entry.get('publish_markers', True))

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.camera = None              # the current CameraModel, rebuilt on change
        self.camera_problem = 'no CameraInfo received yet'
        self.depth_info = None          # the depth CameraInfo's model, when subscribed
        self.latest = None              # (rgb, depth_m, stamp)
        self.last_pub = 0.0
        self.pending_ns = None          # a Trigger's time on the node clock: serve the first frame stamped at/after it
        self.pending_mono = 0.0         # ... and on the monotonic clock: it expires CAPTURE_TTL_S later
        self.capture_counts = {'failed': 0, 'expired': 0, 'unmeasurable': 0}
        self.frame_warned = False
        self.roi_problem = ''           # the eligibility polygon against the CameraInfo's resolution
        # instance_topic: frames and instance halves waiting for their partner, by stamp (ns)
        self.frames = OrderedDict()     # stamp_ns -> (rgb, depth_m, stamp, colour frame_id)
        self.halves = OrderedDict()     # stamp_ns -> {'image': Image, 'info': Detection2DArray}
        self.pair_counts = {'paired': 0, 'frames_unpaired': 0, 'instances_unpaired': 0, 'skipped': 0,
                            'clock_resets': 0}
        self.instance_warned = set()

        self.create_subscription(CameraInfo, info_t, self._on_info, INFO_QOS)
        if dinfo_t:
            self.create_subscription(CameraInfo, dinfo_t, self._on_depth_info, INFO_QOS)
        self.sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, rgb_t, qos_profile=SENSOR_QOS),
             Subscriber(self, Image, depth_t, qos_profile=SENSOR_QOS)], queue_size=5, slop=0.05)
        self.sync.registerCallback(self._on_pair)
        if self.instance_mode:
            self.create_subscription(Image, self.segmenter.instances_topic,
                                     lambda m: self._on_half('image', m), INSTANCE_QOS)
            self.create_subscription(Detection2DArray, self.segmenter.info_topic,
                                     lambda m: self._on_half('info', m), INSTANCE_QOS)

        base = f'/perception/{name}'
        self.pub = self.create_publisher(GraspCandidateArray, f'{base}/grasp_candidates', RESULT_QOS)
        self.pub_diag = self.create_publisher(String, f'{base}/grasp_diagnostics', RESULT_QOS)
        self.pub_objects = self.create_publisher(Detection3DArray, f'{base}/objects', RESULT_QOS)
        self.pub_markers = (self.create_publisher(MarkerArray, f'{base}/grasp_markers', RESULT_QOS)
                            if self.publish_markers else None)
        self.create_service(Trigger, f'{base}/detect', self._on_trigger)
        self.create_timer(0.5, self._expire_capture)        # an expired capture is reported even without frames
        c = self.config
        self.get_logger().info(
            f"surface grasp '{name}' (tp.surface_grasp) for {c.eef}/{c.site} ({c.grasp_mode}, contact radius "
            f"{c.contact_radius_m * 1e3:.1f} mm), approach cone {c.cone_deg:g} deg about {list(c.cone_axis)}, "
            f"standoff {c.standoff_m:g} m in [{c.standoff_min_m:g}, {c.standoff_max_m:g}], segmentation "
            f"{self.seg_method}"
            f"{f' from /perception/{self.segmenter.source}' if self.instance_mode else ''}, eligibility "
            f"{'a ' + str(len(self.eligibility.roi_px)) + '-vertex polygon' if self.eligibility else 'none'} -> "
            f"{base}/grasp_candidates ({'continuous' if self.continuous else 'on demand'})")

    # --- camera model ---------------------------------------------------------------------------------------------
    @staticmethod
    def _model(msg: CameraInfo) -> CameraModel:
        return CameraModel.from_fields(k=msg.k, d=msg.d, distortion_model=msg.distortion_model, width=msg.width,
                                       height=msg.height, frame_id=msg.header.frame_id)

    def _on_info(self, msg: CameraInfo):
        try:
            model = self._model(msg)
            problem = ''
        except CameraModelError as exc:
            model, problem = None, f'colour CameraInfo refused: {exc}'
        with self.lock:
            changed = not (model is not None and model.same_as(self.camera))
            if changed:
                self.camera, self.camera_problem = model, problem
                if model is not None:
                    self._check_depth_info_locked()
                    roi = self.eligibility.image_problems(model.width, model.height) if self.eligibility else []
                    self.roi_problem = ('eligibility refused: ' + '; '.join(roi)) if roi else ''
            roi_problem = self.roi_problem
        if changed and roi_problem:
            self.get_logger().error(roi_problem)
        if changed:
            if model is None:
                self.get_logger().error(problem, throttle_duration_sec=5.0)
            else:
                K = model.matrix
                self.get_logger().info(f'camera model: fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} '
                                       f'cy={K[1, 2]:.2f} {model.width}x{model.height} {model.distortion_model} '
                                       f'distorted={model.distorted} frame={model.frame_id}')

    def _on_depth_info(self, msg: CameraInfo):
        try:
            model = self._model(msg)
        except CameraModelError as exc:
            model = exc
        with self.lock:
            self.depth_info = model
            self._check_depth_info_locked()

    def _check_depth_info_locked(self):
        """The registered-depth clause: the depth CameraInfo IS the colour one. Anything else is refused."""
        d = self.depth_info
        if d is None or self.camera is None:
            return
        if isinstance(d, Exception):
            self.camera_problem = f'depth CameraInfo refused: {d}'
        elif (d.K, d.D, d.width, d.height) != (self.camera.K, self.camera.D, self.camera.width, self.camera.height):
            self.camera_problem = ('the depth CameraInfo differs from the colour one: the depth is not registered '
                                   'to colour (camera contract), so it cannot be back-projected through one model')
        else:
            self.camera_problem = ''

    # --- frames -----------------------------------------------------------------------------------------------------
    def _on_pair(self, rgb_msg: Image, depth_msg: Image):
        # a callback NEVER raises: an exception escaping it would stop spin() and kill the provider
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg)
            depth_m = depth_to_metres(depth, depth_msg.encoding)
        except Exception as exc:                                 # noqa: BLE001
            self.get_logger().error(f'frame dropped: {type(exc).__name__}: {exc}', throttle_duration_sec=5.0)
            return
        stamp = rgb_msg.header.stamp
        stamp_ns = _ns(stamp)
        with self.lock:
            self.latest = (rgb, depth_m, stamp)
            cam = self.camera
        if cam is not None and rgb_msg.header.frame_id and rgb_msg.header.frame_id != cam.frame_id \
                and not self.frame_warned:
            self.frame_warned = True
            self.get_logger().warning(f"the image's frame '{rgb_msg.header.frame_id}' is not the CameraInfo's "
                                      f"'{cam.frame_id}': the candidates are stamped in the CameraInfo's frame")
        if self.instance_mode:
            # a frame is processed only WITH its instances: it waits for them (bounded), paired by exact stamp
            with self.lock:
                self._admit_locked(stamp_ns)
                self.frames[stamp_ns] = (rgb, depth_m, stamp, rgb_msg.header.frame_id)
            self._try_pair(stamp_ns)
            return
        with self.lock:
            due, capture = self._take_capture_locked(stamp_ns)
        if due or (self.continuous and (time.monotonic() - self.last_pub) >= self.min_period):
            result, why = self._run_and_publish()
            if due and result is None:
                self._capture_not_served(capture, why)

    # --- the pending capture ----------------------------------------------------------------------------------------
    def _expire_capture_locked(self) -> bool:
        """Drop (and count) a pending capture older than CAPTURE_TTL_S of monotonic time. True when it did."""
        if self.pending_ns is None or time.monotonic() - self.pending_mono <= CAPTURE_TTL_S:
            return False
        pending = self.pending_ns
        self.pending_ns = None
        self.capture_counts['expired'] += 1
        self.get_logger().error(f'capture requested at {pending * 1e-9:.3f} s expired after {CAPTURE_TTL_S:g} s '
                                f'without a usable frame: it will not be served')
        return True

    def _expire_capture(self):
        try:
            with self.lock:
                self._expire_capture_locked()
        except Exception as exc:                                 # noqa: BLE001 -- never out of a callback
            self.get_logger().error(f'capture expiry check failed: {type(exc).__name__}: {exc}',
                                    throttle_duration_sec=5.0)

    def _take_capture_locked(self, stamp_ns: int):
        """(due, capture): whether a frame stamped `stamp_ns` serves the pending (unexpired) capture, which it
        then consumes; `capture` = (pending_ns, pending_mono) to restore it when the frame is only skipped."""
        self._expire_capture_locked()
        capture = (self.pending_ns, self.pending_mono)
        due = self.pending_ns is not None and stamp_ns >= self.pending_ns
        if due:
            self.pending_ns = None
        return due, capture

    def _capture_not_served(self, capture, why: str):
        """A frame due for a capture published nothing: a skipped instance pair or an UNMEASURABLE frame (no valid
        depth) keeps the capture pending (bounded by CAPTURE_TTL_S); any other failure consumes it."""
        if why.startswith((SKIPPED, UNMEASURABLE)):
            with self.lock:
                if self.pending_ns is None:
                    self.pending_ns, self.pending_mono = capture
                    self._expire_capture_locked()                # ... but it never outlives its bound
        else:
            self._capture_failed(capture, why)

    def _capture_failed(self, capture, why: str):
        """A capture consumed by a frame that published nothing: never silent."""
        with self.lock:
            self.capture_counts['failed'] += 1
        self.get_logger().error(f'capture requested at {capture[0] * 1e-9:.3f} s consumed WITHOUT a result: {why}')

    # --- instance_topic: pairing by exact stamp ---------------------------------------------------------------------
    def _on_half(self, kind: str, msg):
        try:
            stamp_ns = _ns(msg.header.stamp)
            with self.lock:
                self.halves.setdefault(stamp_ns, {})[kind] = msg     # a late half is merely pruned
            self._try_pair(stamp_ns)
        except Exception as exc:                                 # noqa: BLE001 -- never out of a callback
            self.get_logger().error(f'instance {kind} dropped: {type(exc).__name__}: {exc}',
                                    throttle_duration_sec=5.0)

    def _try_pair(self, stamp_ns: int):
        with self.lock:
            half = self.halves.get(stamp_ns)
            ready = stamp_ns in self.frames and half is not None and {'image', 'info'} <= set(half)
            if ready:
                frame = self.frames.pop(stamp_ns)
                self.halves.pop(stamp_ns)
                self.pair_counts['paired'] += 1
                due, capture = self._take_capture_locked(stamp_ns)
            self._prune_locked()
        if not ready:
            return
        if due or (self.continuous and (time.monotonic() - self.last_pub) >= self.min_period):
            result, why = self._run_and_publish(frame, (half['image'], half['info']))
            if due and result is None:
                self._capture_not_served(capture, why)

    def _admit_locked(self, stamp_ns: int):
        """A colour FRAME stamped more than PAIR_WAIT_S BEHIND the newest stamp held means the clock went back (a
        sim-time or bag restart; camera frames do not arrive that late): everything held is stale and would
        otherwise make every new, smaller stamp look too old to pair. Clear both stores (counted as unpaired, plus
        one `clock_resets`) and warn. (An instance half that late is only pruned: a slow segmenter is not a reset.)"""
        newest = max(max(self.frames, default=stamp_ns), max(self.halves, default=stamp_ns))
        if stamp_ns >= newest - int(PAIR_WAIT_S * 1e9):
            return
        n_frames, n_halves = len(self.frames), len(self.halves)
        self.pair_counts['frames_unpaired'] += n_frames
        self.pair_counts['instances_unpaired'] += n_halves
        self.pair_counts['clock_resets'] += 1
        self.frames.clear()
        self.halves.clear()
        self.get_logger().warning(f'the stamps went back by {(newest - stamp_ns) * 1e-9:.3f} s (a sim-time or bag '
                                  f'restart?): dropped {n_frames} held frame(s) and {n_halves} instance half(s), '
                                  f'pairing restarts from stamp {stamp_ns * 1e-9:.3f} s')

    def _prune_locked(self):
        """Drop (and count) what waited too long for its partner: older than PAIR_WAIT_S behind the newest stamp
        seen, or beyond the PAIR_BUFFER newest."""
        stamps = list(self.frames) + list(self.halves)
        if not stamps:
            return
        horizon = max(stamps) - int(PAIR_WAIT_S * 1e9)
        for store, key in ((self.frames, 'frames_unpaired'), (self.halves, 'instances_unpaired')):
            for ns in sorted(store):
                if ns < horizon or len(store) > PAIR_BUFFER:
                    store.pop(ns)
                    self.pair_counts[key] += 1

    def _warn_once(self, key: str, text: str):
        if key not in self.instance_warned:
            self.instance_warned.add(key)
            self.get_logger().warning(text)

    def _instance_regions(self, halves, colour_frame: str, camera: CameraModel):
        """(regions, report) from one paired instance image + info, or (None, why) when the pair breaks the
        instance contract (warned once per kind, the frame skipped)."""
        img_msg, info_msg = halves
        cfg = self.segmenter
        if img_msg.encoding not in INSTANCE_ENCODINGS:
            why = f'the instance image is {img_msg.encoding!r}, the contract {"/".join(INSTANCE_ENCODINGS)}'
            self._warn_once('encoding', f'instance frames skipped: {why}')
            return None, why
        if (img_msg.width, img_msg.height) != (camera.width, camera.height):
            why = (f'the instance image is {img_msg.width}x{img_msg.height}, the colour CameraInfo '
                   f'{camera.width}x{camera.height}: it must be registered to colour, same size')
            self._warn_once('size', f'instance frames skipped: {why}')
            return None, why
        expected = colour_frame or camera.frame_id
        for what, frame in (('instance image', img_msg.header.frame_id), ('instance_info', info_msg.header.frame_id)):
            if frame and frame != expected:
                why = f"the {what}'s frame '{frame}' is not the colour image's '{expected}'"
                self._warn_once(f'frame:{what}', f'instance frames skipped: {why}')
                return None, why
        try:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            regions, report = regions_from_instance_image(image, instance_infos(info_msg), cfg)
        except InstanceContractError as exc:
            self.get_logger().warning(f'instance frame skipped: {exc}', throttle_duration_sec=5.0)
            return None, str(exc)
        return regions, report

    def _run_and_publish(self, frame=None, halves=None):
        try:
            return self._run_and_publish_unsafe(frame, halves)
        except Exception as exc:                                 # noqa: BLE001 -- never out of a callback
            msg = f'no candidates published: {type(exc).__name__}: {exc}'
            self.get_logger().error(msg, throttle_duration_sec=5.0)
            return None, msg

    def _run_and_publish_unsafe(self, frame=None, halves=None):
        with self.lock:
            if frame is None:
                if self.latest is None:
                    return None, 'no image received yet'
                frame = self.latest
            camera, problem = self.camera, self.camera_problem or self.roi_problem
        rgb, depth_m, stamp = frame[:3]
        colour_frame = frame[3] if len(frame) > 3 else ''
        self.last_pub = time.monotonic()
        if camera is None or problem:
            self.get_logger().error(f'no candidates published: {problem}', throttle_duration_sec=5.0)
            return None, problem
        if depth_m.shape[:2] != (camera.height, camera.width) or rgb.shape[:2] != (camera.height, camera.width):
            msg = (f'no candidates published: the images are {rgb.shape[1]}x{rgb.shape[0]} / '
                   f'{depth_m.shape[1]}x{depth_m.shape[0]}, the CameraInfo {camera.width}x{camera.height}')
            self.get_logger().error(msg, throttle_duration_sec=5.0)
            return None, msg
        if self.instance_mode:
            if halves is None:
                return None, f'{SKIPPED}: no instances for this frame'
            regions, report = self._instance_regions(halves, colour_frame, camera)
            if regions is None:
                with self.lock:
                    self.pair_counts['skipped'] += 1
                return None, f'{SKIPPED}: {report}'
            with self.lock:
                report = dict(report, pairing=dict(self.pair_counts), captures=dict(self.capture_counts))
        else:
            seg = self.segmenter
            found = (seg.components(rgb, depth_m, camera.matrix) if hasattr(seg, 'components')
                     else seg.detect(rgb, depth_m, camera.matrix))
            regions = regions_from_detections(found)
            with self.lock:
                report = {'method': self.seg_method, 'captures': dict(self.capture_counts)}
        regions, ineligible = split_eligible(regions, depth_m, self.eligibility)
        presence = present_objects(regions, depth_m, camera, self.config.z_min_m, self.config.z_max_m)
        if presence.unmeasurable:
            # fail closed: an empty array would read "no object" (an empty bin); publish nothing for this frame
            with self.lock:
                self.capture_counts['unmeasurable'] += 1
            msg = (f'{UNMEASURABLE}: {len(presence.skipped)} eligible region(s) but no valid depth in '
                   f'({self.config.z_min_m:g}, {self.config.z_max_m:g}) m anywhere in the frame (a depth dropout? '
                   f'a wrong depth band?): nothing published, never an empty answer')
            self.get_logger().error(msg, throttle_duration_sec=5.0)
            return None, msg
        result = propose(regions, depth_m, camera, self.config)
        header = Header(stamp=stamp, frame_id=camera.frame_id)
        self.pub.publish(to_candidate_array(result, header))
        self.pub_objects.publish(to_objects_array(presence, header))
        self.pub_diag.publish(String(data=to_diagnostics_json(result, header, camera, ineligible=ineligible,
                                                              segmentation=report, presence=presence)))
        if self.pub_markers is not None:
            self.pub_markers.publish(to_markers(result, header))
        return result, ''

    def _on_trigger(self, _req, resp):
        """A CAPTURE: the result comes from the first frame stamped at or after this request (on the node clock,
        the clock the consumer's capture boundary is on), so a capture is never answered with an older image
        (vNext M4 review). success = the request was accepted (a result will be published); refused when the
        camera model is refused or no camera model exists yet. The capture expires CAPTURE_TTL_S (monotonic) later."""
        with self.lock:
            camera, problem = self.camera, self.camera_problem or self.roi_problem
        if camera is None or problem:
            resp.success, resp.message = False, problem or 'no CameraInfo received yet'
            return resp
        now = self.get_clock().now().nanoseconds
        with self.lock:
            self.pending_ns, self.pending_mono = now, time.monotonic()
        resp.success = True
        resp.message = (f'scheduled on the first frame stamped at or after {now * 1e-9:.3f} s'
                        + (' that has its instances' if self.instance_mode else ''))
        return resp


def _ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main(args=None):
    rclpy.init(args=args)
    node = SurfaceGraspNode()
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
