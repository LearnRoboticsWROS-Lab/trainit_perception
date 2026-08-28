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
| `detections[i].results[0].pose.pose` | position; orientation identity unless the method gives one |
| `detections[i].bbox.size` | approximate metric size; 0 = unknown |

## Topics and service

| | name | type | QoS |
|---|---|---|---|
| out | `/perception/<name>/detections` | `Detection3DArray` | **RELIABLE**, depth 1 |
| debug | `/perception/<name>/mask` | `sensor_msgs/Image` mono8 | BEST_EFFORT |
| on demand | `/perception/<name>/detect` | `std_srvs/Trigger` | — |

Detections are **RELIABLE** where images are BEST_EFFORT: an image is a stream and a
dropped frame is replaced by the next; a detection is an event a consumer waited for
and must not miss.

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
