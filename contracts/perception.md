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

## Grasp candidates — the SurfaceGrasp provider (0.6.0, vNext M4)

A grasp PROVIDER is a node of its own (`surface_grasp_node`, provider id `tp.surface_grasp`) configured by one
`perception.yaml` entry with `method: surface_grasp`. It consumes the same camera contract and publishes the
canonical manipulation contract of ADR-0026 instead of detections:

| | name | type | QoS |
|---|---|---|---|
| out | `/perception/<name>/grasp_candidates` | `trainit_interfaces/GraspCandidateArray` | **RELIABLE**, depth 1, **best first** |
| additive | `/perception/<name>/grasp_diagnostics` | `std_msgs/String` (JSON), paired by stamp | **RELIABLE**, depth 1 |
| debug | `/perception/<name>/grasp_markers` | `visualization_msgs/MarkerArray` | **RELIABLE**, depth 1 |
| on demand | `/perception/<name>/detect` | `std_srvs/Trigger` (success = sampled; an empty array is an answer) | — |

What a candidate means (every value in the observation frame = the colour optical frame, at the colour image's
stamp):

- `grasp_pose.position` — the contact point: the inlier centroid of the plane fitted to the region's registered
  depth, back-projected through the camera model (K and the `plumb_bob` distortion; any other non-zero distortion,
  or a depth CameraInfo that differs from the colour one, publishes nothing);
- `surface_normal` — the unit plane normal, pointing TOWARD THE OBSERVER along the view ray (`normal_policy =
  toward_observer`);
- `approach` = -normal: the unit direction the tool travels, INTO the surface;
- `grasp_pose.orientation` — the provider's reference (+Z = the configured approach-cone axis, +X = the configured
  `reference_x_axis` projected) tilted minimally onto the approach: its +Z IS the approach;
- `standoff_m` — the configured suggestion, inside its validated band; `eef` / `site` — the configured ids;
- `score` — the product of the evaluated terms (`term.*` in `metadata`); `confidence` — the fit's `pose_quality`;
- `metadata.not_evaluated` — always `reachability` and `collision` (the motion layer resolves them by trying),
  plus any term the configuration does not enable. A term not evaluated is never scored as passed.

A region that cannot support a valid candidate is never published as one: it is listed on the diagnostics with a
deterministic code (`REGION_EMPTY`, `INSUFFICIENT_DEPTH`, `FIT_FAILED`, `GEOMETRY_INVALID`, `NORMAL_GRAZING`,
`APPROACH_OUTSIDE_CONE`, `POSE_QUALITY_REFUSED`, `POSE_QUALITY_LOW`, `SURFACE_TOO_SMALL`, `VISIBILITY_LOW`,
`CLEARANCE_UNKNOWN`, `CLEARANCE_LOW`, `ORIENTATION_UNDEFINED`, `BEYOND_MAX_CANDIDATES`; since 0.8.0
`FACE_FIT_FAILED`, only with `centre_estimator: known_shape_fit`). The approach cone
(`approach_cone_deg`, `approach_cone_axis`) is REQUIRED and inclusive at its boundary; an approach outside it is
rejected, never rotated into it. Ties in score are ordered by candidate id; the same input gives the same
candidates, ids, terms, order and rejections.

A consumer takes the array in order and never re-ranks it; it never reads `metadata`.

## Objects present — `/perception/<name>/objects` (0.7.0, vNext M4.5b)

A grasp provider answers two questions that stay apart: **is a valid grasp available** (the candidate array,
unchanged) and **is an object present** (this). With EVERY candidate array (capture and continuous) a provider also
publishes the objects it saw, as TP's detection contract above:

| | name | type | QoS |
|---|---|---|---|
| presence | `/perception/<name>/objects` | `vision_msgs/Detection3DArray` | **RELIABLE**, depth 1 |

- **header** — the candidate array's own header: the SAME stamp and frame (the observation), so a capture that
  triggers `/perception/<name>/detect` gets both from one frame.
- **one `Detection3D` per ELIGIBLE region** (after the workspace eligibility, BEFORE the geometry), whatever its
  geometry verdict: a part near a wall stays present, only its grasp is invalid. An ineligible region (outside the
  workspace) is not present. Order: region order (no ranking: presence is not a candidate list).
- `id` = the provider's region id (`<class>@<u>,<v>`, the id its candidate or rejection carries);
  `results[0].hypothesis.class_id` / `.score` = the segmentation's; `results[0].pose.pose.position` = the 3D
  centroid of the region's VALID registered depth (the provider's rays and depth band), orientation identity;
  `bbox.size` = the 5th..95th percentile metric extent of those points (0 = unknown).
- **no valid depth** — the region is placed on its pixel centroid's ray at the frame's median valid depth (size 0),
  never dropped silently, and counted on the diagnostics: `summary.objects`, `summary.objects_without_depth`, and
  an `objects` block `{published: [{id, class_id, score, position, size, depth: measured|estimated}],
  without_depth, skipped}` (`summary.objects_skipped` / `skipped` stay 0 / empty on a published record).
- **an unmeasurable frame is never an empty bin** (fail closed) — a frame with >= 1 eligible region but NO valid
  depth anywhere in the depth band (a depth dropout, a wrong `z_min_m`/`z_max_m`) cannot measure its objects: the
  provider publishes NOTHING for it (no candidates, no objects, no diagnostics, no markers), logs an error
  (throttled) and counts it `segmentation.captures.unmeasurable` (every such frame). A pending capture is NOT
  consumed by it: the next measurable frame serves it; if the blindness persists the capture expires
  (`CAPTURE_TTL_S`) and the consumer times out — a fault (TMR: `DETECTION_TIMEOUT`), never "no object". Only a
  frame with NO eligible region (or only regions without pixels) is a genuine empty answer.

A consumer that must tell an EMPTY workspace from parts no grasp can serve captures this topic (TMR's `DetectObject`
with `trigger="true"`): a fresh empty array = nothing present (it is published only for a frame with no eligible
region, never for one whose depth is blind). The scripted provider publishes a scenario's
`objects` the same way (below).

## Instance segmentation from an external segmenter — `instance_topic` (0.7.0, vNext M4.5)

A grasp provider can take its regions from ANY external instance segmenter (a YOLO-seg node, Mask R-CNN, SAM, a
vendor SDK) instead of an in-process TP detector: `segmentation: {method: instance_topic, params: {source: <s>}}`.
The segmenter is outside TP and imports nothing of it; TP never sees a model, only this contract (D0's
`INSTANCE_MASK` representation: a compact-id image plus a class per id):

| topic | type | content |
|---|---|---|
| `/perception/<source>/instances` | `sensor_msgs/Image`, encoding **`16UC1`** (or `mono16`) | `0` = background, `k` = instance `k` (1..65535); **registered to colour, the colour image's size** |
| `/perception/<source>/instance_info` | `vision_msgs/Detection2DArray` | one `Detection2D` per instance: `id` = `str(k)` (the pixel value), `results[0].hypothesis.class_id` / `.score` (0..1), `bbox` (informative) |

Rules (the provider refuses a pair that breaks them):

- **stamp** — both messages carry the COLOUR image's `header.stamp`, exactly; **frame** — the colour optical frame
  (the colour image's `frame_id`). The provider pairs image + info + the colour/depth frame by EXACT stamp.
- **ids** — unique per frame; an id that is not an ASCII decimal pixel value, a duplicate id, or a detection without a
  hypothesis skips the frame. A pixel value no detection labels is never a region (counted `unlabelled`).
- **QoS** — any: the provider subscribes BEST_EFFORT, depth 5 (it matches a reliable or a best-effort publisher).
- **waiting** — a colour frame is processed ONLY with its instances. It waits for them at most **2.0 s of stamp
  time** and among the **10** newest frames (`PAIR_WAIT_S`, `PAIR_BUFFER`); an unpaired frame or instance half is
  then dropped and counted (`segmentation.pairing` on the diagnostics). A capture (`/detect`) is served from the
  first frame stamped at or after the request **that has its instances**. A colour frame stamped more than 2.0 s
  BEHIND the newest stamp held (a sim-time / bag restart: the clock went back) clears both stores, warned and
  counted `pairing.clock_resets`: the pairing resumes on the new clock instead of stalling.
- **capture lifetime** (every segmentation) — a pending capture is served only within **10 s of monotonic time**
  after its Trigger (`CAPTURE_TTL_S`, longer than any sensible consumer wait); then it expires (an error, counted
  `segmentation.captures.expired`) and is never served, however late its frame or instances arrive. A capture whose
  frame fails for any reason other than a skipped instance pair or an unmeasurable frame (no valid depth, above:
  kept pending, counted `segmentation.captures.unmeasurable`) — a refused camera model, images of another size
  than the CameraInfo, an exception — is consumed: nothing is published, an error is logged, and it is counted
  `segmentation.captures.failed`. The counters ride on every later diagnostics record.
- **size / frame / encoding** mismatch — warned ONCE per kind, the frame skipped (counted `skipped`); a skipped frame
  does not consume a pending capture.

Parameters: `source` (REQUIRED, one ROS name token), `classes` (allow-list; empty = every class), `min_score`
(0..1, inclusive), `max_objects` (the best by score, then area, then id; default 20), `min_area_px`. The kept
instances become regions in instance-id order: mask = `image == k`, class and score from the info (the score is
the provider's `detection` term, clipped to 0..1).

## Workspace eligibility — `eligibility` (0.7.0, vNext M4.5)

An entry key of a grasp provider that reads images (`surface_grasp`), beside `segmentation`:

```yaml
eligibility:
  roi_px: [[u, v], [u, v], [u, v], ...]   # REQUIRED: >= 3 vertices, a simple polygon, colour pixels
  min_inside: 0.9                         # the share of a region's mask inside the polygon, in (0, 1]
  z_min_m: 0.30                           # optional depth band (observation frame) for the region's MEDIAN
  z_max_m: 0.90                           #   depth, inclusive; default [0, 20] = no band
```

It runs BEFORE the provider: a region with `inside_fraction < min_inside` (a pixel counts when its centre is
inside the polygon or ON its boundary; `min_inside` itself is eligible) or whose median valid depth is outside the
band never enters `propose`. It is NOT a geometry rejection: the diagnostics list it under `ineligible` as
`{id, class_id, code: OUTSIDE_WORKSPACE, why, inside_fraction}` and count it in `summary.ineligible`
(`summary.regions` counts the regions that entered the provider); ADR-0031's 15 geometry codes are unchanged. A
region without pixels or without valid depth is left to the provider (`REGION_EMPTY`, `INSUFFICIENT_DEPTH`). The
polygon is validated at start-up (shape) and against the CameraInfo's resolution (every vertex in
`[0, width] x [0, height]`); a polygon off the image publishes nothing and the Trigger answers why.

The diagnostics JSON of a frame is then:

```
{provider, stamp: {sec, nanosec}, frame_id,
 summary: {regions, valid, rejected, rejections_by_code, ineligible},
 camera, candidates: [...], rejected: [{id, class_id, code, why, terms}],
 ineligible: [{id, class_id, code: OUTSIDE_WORKSPACE, why, inside_fraction}], timing_ms,
 segmentation: {method, captures: {failed, expired, unmeasurable}, ...},
 objects: {published: [...], without_depth: [...], skipped: [...]}}     # M4.5b; summary adds objects,
                                                                        # objects_without_depth, objects_skipped
                                   # instance_topic adds: source, instances, kept, dropped_by, unlabelled, pairing
```

## The scripted grasp provider — `tp.scripted` (0.7.0, vNext M4.5)

For MOCK product tests (a mock camera is a flat plane, so no measured provider can propose anything). The node
`scripted_grasp_node` serves one entry `{method: scripted_grasp, params: {scenario_file, frame_id, after_end:
empty|repeat_last, approach_cone_axis, reference_x_axis}, continuous: false}` behind the SAME runtime surface as
SurfaceGrasp: `/perception/<name>/detect` (each Trigger = the NEXT scenario step), `/perception/<name>/
grasp_candidates` (RELIABLE, depth 1, in scenario order and serialised in Trigger order: a step never overtakes
an earlier one, so its delay is `delay_ms` at least, `provider = tp.scripted`), `/perception/<name>/
grasp_diagnostics` (the same minimal JSON shape plus `scenario: {step, steps, past_end}`). A step publishes its
candidates `delay_ms` after the Trigger's response, stamped `now + stamp_offset_ms` on the node clock
(`use_sim_time` honoured; negative = stale); `unavailable: true` answers success=false and publishes nothing. A
candidate without `orientation` gets SurfaceGrasp's rule (+Z = the approach). The scenario is validated in full at
start-up. It measures nothing: it is never a perception result. M4.5b: each published step also publishes its
objects present on `/perception/<name>/objects` (the array's header; id `scripted_object@<k>`, class `''`, score 1,
size 0): a step's `objects: [[x, y, z], ...]` (positions in `frame_id`), or an int `n` (the step's first `n`
candidates' positions), or -- without the key -- one per candidate; `objects: []` = nothing present; past the end
`empty` presents nothing. The diagnostics add `summary.objects`, `summary.objects_without_depth` (0) and the
`objects` block.

## The contact's centre — `centre_estimator` (0.8.0, vNext M4.5d)

Where on a measured face SurfaceGrasp puts the contact. A SurfaceGrasp parameter (dated 0.8.0 in the vocabulary's
`param_since`, so a configurator that writes one pins TP >= 0.8.0):

```yaml
centre_estimator: inlier_centroid   # DEFAULT, the 0.6.0 behaviour: the plane's inlier centroid
centre_estimator: known_shape_fit   # the centre of the KNOWN face fitted to the partial evidence
grasp_face: disc                    #   needs a disc face ...
expected_radius_m: 0.020           #   ... of a known radius (a measured part dimension)
face_occlusion_mm: 3.0              # a boundary whose neighbour stands more than this IN FRONT of the plane is an
                                    #   occlusion (bounds the visible part, not the face); must exceed inlier_mm
face_mask_dilate_px: 0              # how far the face's support may grow beyond the mask (0 = the mask)
face_min_coverage: 0.35             # refused below this share of the expected disc visible
face_max_outside: 0.6               # refused when more of the support than this lies outside the expected disc
face_max_ambiguity_mm: 3.0          # refused when the consensus disc can slide further than this and the edges
                                    #   do not pin it
face_min_arc_deg: 150.0             # the edge refinement needs an arc this wide ...
face_max_rms_mm: 1.0                # ... fitted within this rms ...
face_radius_tol: 0.1                # ... with a free-radius fit within this fraction of expected_radius_m
```

`known_shape_fit` (module `trainit_perception.grasp.face_centre`): the region's in-plane pixels form the face's
support (the largest seeded 4-connected component; another joins only if it lies inside the expected disc found on
it -- a face split by an occluder, never a coplanar neighbour); the disc of the expected radius containing the most
support points (a grid consensus) and, when the face's TRUE edges (a drop-off, or a crease rising at most
`face_occlusion_mm`; never an occlusion, never a mask cut over the same plane) support a least-squares circle of the
expected radius that agrees with the consensus, that circle's centre. Which decided is reported. A face whose
centre the evidence does not determine is refused with the new geometry code **`FACE_FIT_FAILED`** (checked right
after `GEOMETRY_INVALID`; ADR-0031 amendment M4.5d) -- there is NO fallback to the visible centroid. The candidate's
metadata add `centre_estimator` and, for `known_shape_fit`, `centre_by` (`edges` | `consensus`) and
`centre_shift_mm` (the fitted centre vs the visible centroid); the diagnostics' candidate records carry the same.
Two more refusals are fixed rules, not parameters: a SECOND disc-sized face in the support (a disc of the expected
radius elsewhere holding mostly other points and at least 0.6 of the chosen disc's count: which one is the face is
ambiguous), and a consensus that does not settle inside its search window. The grid searches use at most 4000
support points (a deterministic stride). Known limit: SurfaceGrasp's contact checks downstream still use the
region's mask, so a fitted centre well outside a truncated mask is refused there (`CONTACT_UNSUPPORTED`), not moved.
`centre_estimator: inlier_centroid` leaves every 0.7.0 result's GEOMETRY unchanged (pose, point, score, codes); the
metadata and diagnostics gain the keys above (additive only).
