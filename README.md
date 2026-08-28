# TrainIt perception

Detectors that turn the camera contract into the perception contract
(`vision_msgs/Detection3DArray`).

```
contracts/perception.md      what a detector exposes — consumers depend on this only
profiles/<method>.yaml       each detector's parameters, the schema TSA renders a form from
trainit_perception/
  detectors/                 PURE FUNCTIONS: (rgb, depth, K, params) -> [Detection]
  nodes/detector_node.py     the one ROS node: loads a detector from perception.yaml
```

## The rule

A detector is a pure function with no ROS inside. The Setup Assistant's live tuner and
the runtime node call **the same `detect()`**, so what the user sees while tuning is, by
construction, what the bundle runs.

## Run

```bash
ros2 run trainit_perception detector_node --ros-args \
    -p config_file:=/path/to/perception.yaml -p detector_name:=cube
ros2 topic echo /perception/cube/detections
ros2 service call /perception/cube/detect std_srvs/srv/Trigger
```

## Methods

| method | goal | orientation | status |
|---|---|---|---|
| `color_mask` | locate | no (top-down grasp) | v0.1 |
| fiducial (ArUco) | locate | full 6-DoF | next |
| learned model | locate / classify | per model | later |

Design: `fr3wml_digital_twin/docs/design_perception_integration.md`.
