# Contract — perception

What a detector exposes to ROS. Consumers (TMR's `DetectObject`) depend on this and on
nothing else; detectors are swappable behind it.

## Message

`vision_msgs/Detection3DArray` — the ROS standard. Isaac ROS, YOLO wrappers and ArUco
nodes emit it, so a learned-model detector drops in with no change to any consumer.

| field | meaning |
|---|---|
| `header.frame_id` | the **optical** frame the poses are in (`camera_color_optical_frame`) |
| `header.stamp` | the stamp of the image the detection came from — consumers check freshness |
| `detections[i].results[0].hypothesis.class_id` | what it is (`cube`); consumers filter on it |
| `detections[i].results[0].hypothesis.score` | 0..1, detector-specific confidence |
| `detections[i].results[0].pose.pose` | position; orientation = the rotation taking +Z onto the fitted surface normal, identity when the method gives none |
| `detections[i].bbox.size` | approximate metric size; 0 = unknown |
| **`detections` order** | **the RANKING: `detections[0]` is the best candidate** |

## The ORDER is the ranking

`detections` goes out **best first**, by the detector's configured `candidate_rule`
(ADR-0014, ADR-0016 Decision 1, ADR-0017). A consumer that takes `detections[0]` gets
the configured candidate with nothing extra to subscribe to; a consumer that wants the
second takes `detections[1]`, which is what makes "try the next one" an **index**
(TMR's `DetectObject.keep` and `SetWaypointFromDetection.index`) instead of a program.

Ordering is the one channel that carries a rank across this contract without either
side learning the other's vocabulary: the detector knows nothing about robots, the
runtime knows nothing about masks, and *best first* needs neither. **A consumer must
not re-rank.** It filters by `class_id`, truncates, and takes them in the order it was
given — a runtime that ranked would have to know what `visible_fraction` means.

A rule whose terms a detector cannot compute **stops the detector at start-up**, with
the accepted names in the message. An order that is not the ranking that was asked for
looks exactly like a perception fault, which is the expensive kind of mistake.

Two honest limits of the rank, both measured: it predicts the **pose's** error (9,38 mm
in the worst quintile against 1,68 mm in the best) and it does **not** predict whether
the pick works (correlation −0,17). Reachability is the runtime's business, and the
runtime resolves it by trying rather than predicting it (ADR-0014).

## Topics and service

| | name | type | QoS |
|---|---|---|---|
| out | `/perception/<name>/detections` | `Detection3DArray` | **RELIABLE**, depth 1 |
| additive | `/perception/<name>/quality` | `std_msgs/String` (JSON) | **RELIABLE**, depth 1 |
| debug | `/perception/<name>/mask` | `sensor_msgs/Image` mono8 | BEST_EFFORT |
| on demand | `/perception/<name>/detect` | `std_srvs/Trigger` | — |

Detections are **RELIABLE** where images are BEST_EFFORT: an image is a stream and a
dropped frame is replaced by the next; a detection is an event a consumer waited for
and must not miss.

## The quality sidecar — additive, and paired by stamp

`vision_msgs/Detection3D` has no free-form field, and the candidate terms are what a
person needs to decide whether to **try** a grasp rather than where to put the tool. So
they travel beside the detections, as JSON on `/perception/<name>/quality`:

```json
{"stamp": {"sec": 7, "nanosec": 11}, "frame_id": "camera_color_optical_frame",
 "detections": [{"class_id": "cap", "score": 0.91, "pose_quality": 0.68,
                 "visible_fraction": 0.95, "box_overlap": 0.10, "clearance": 1.0,
                 "obstruction_mm": 0.0, "edge_on": 0.22, "tilt_deg": 9.9,
                 "candidate_x_visible": 0.59}]}
```

Two rules, and they are what make it usable:

- **`detections[i]` there is `detections[i]` here.** One permutation is applied to both,
  so the terms of candidate *i* are the terms of the pose a consumer will try *i*-th.
- **Pair the two topics by `stamp`**, never by the order they arrived in — ROS does not
  guarantee that. The sidecar carries the same `stamp` and `frame_id` as the array it
  describes.

A non-finite term goes out as `null`: `obstruction_mm` is NaN for *the depth could not
answer*, which is a different answer from 0 (*clear*), and the literal `NaN` that a JSON
encoder would emit is not valid JSON. The terms are additive — a consumer that does not
care never subscribes, and nothing in the standard message changed to make room for them.

## Freshness, not a return value

The consumer does not call a service that returns a pose. It waits for a detection
whose `header.stamp` is **newer than the moment it started waiting**, with a timeout.
That gives "detect now and give me the result" semantics with no custom interface, and
it works unchanged when the detector runs on another machine (a Jetson) across DDS.

The `Trigger` service is for detectors too expensive to run continuously; it runs the
same code path and publishes on the same topic.

## Frame discipline

A detector reports in the camera's optical frame and knows nothing about robots.
Transforming into the planning frame is the consumer's job, through TF. Keeping the
detector robot-agnostic is what lets the same node serve any cell.

## Inputs

The camera contract from `trainit_hardware_layer/contracts/camera.md`: colour image,
depth image registered to colour, and `CameraInfo`. Depth may be `16UC1` millimetres
(device) or `32FC1` metres (Isaac); `depth_to_metres()` normalises it and nothing
downstream cares.
