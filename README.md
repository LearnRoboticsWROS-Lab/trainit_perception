# TrainIt perception

Detectors that turn the camera contract into the perception contract
(`vision_msgs/Detection3DArray`).

```
contracts/perception.md      what a detector exposes — consumers depend on this only
profiles/<method>.yaml       each detector's parameters, documented (the schema is code)
trainit_perception/
  detectors/                 PURE FUNCTIONS: (rgb, depth, K, params) -> [Detection]
  ranking/                   WHICH one to pick: the terms, the 3D fit, the grasp face, the rule
  vocabulary.py              the CONFIGURATION VOCABULARY, exported for the Setup Assistant
  nodes/detector_node.py     the one ROS node: loads a detector from perception.yaml
scripts/score_term.py        scores a term against EXTERNAL labels — a term ships with it
```

## The rule

A detector is a pure function with no ROS inside. The Setup Assistant's live tuner and
the runtime node call **the same `detect()`**, so what the user sees while tuning is, by
construction, what the bundle runs. It is `detect_ranked()` that the node publishes
from — `detect()` plus the candidate ranking — and a tuner must call that one too, so
the picture and the bundle agree about which candidate comes first.

## The candidate ranking — which object to pick, and why

The published **order** of the detections is the ranking (ADR-0016 Decision 1), and the
rule that decides it is a **detector parameter** (ADR-0014, ADR-0017): a dropdown and a
number, not a model. Every term is closed form, nothing is learned, and nothing is
blended by default — publishing one fused number would hide exactly the distinction that
matters.

| term | stage | what it needs | what it sees |
|---|---|---|---|
| `score` | 2D | nothing | the detector's own confidence |
| `pose_quality` | 3D | a fit over the depth in the mask | inliers, residual, coverage, fitted extent |
| `box_overlap` | 2D | a pixel box only | how much of **this** box another covers — works with any detector |
| `visible_fraction` | 2D+ | a mask area and the **face's** size | how much of the face the camera actually gets |
| `clearance` | 3D | depth + the tool's radius and grip window | how far anything stands into the tool's path |
| `edge_on` | 3D | a fit that reports a tilt + the tool's tilt limit | how far the face is turned away from the lens |

```yaml
params:
  candidate_rule: auto          # or candidate_x_visible | score_x_nonoverlap | detector_order | …
  grasp_face: disc              # or rect | cylinder_side
  expected_radius_m: 0.02095    # the face the tool comes down on, not the whole object
  tool_max_tilt_deg: 0.0        # the tilt the TOOL can still hold; 0 = the edge_on term is off
```

What the measurement decided, and it is in the code's comments with its numbers:

- **`candidate_x_visible` is the default wherever a face area is available.** Over 40 pours
  on two independent seeds it lifted 39 of 40 against 35, brought exactly one 34 times
  against 30, and reached 55 % success at a 1,44 mm hold tolerance against 40 % — **without**
  refusing more candidates or leaving fewer plans, so it wins by choosing better among the
  same candidates. `box_overlap` is the fallback for a detector with no mask and no depth.
- **No refusal ships.** The same occlusion idea as a 20 % overlap *filter* throws away 18,1
  of 22,1 candidates and halves the pick rate. A rule is a permutation, never a filter.
- **`clearance` ships as a term and not as a default.** Measured 0,00 on every candidate the
  picker actually chose, while discriminating across the frame on 12 of 27 — it buys nothing
  at the top of *that* ranking and would bite with a wider tool or a fuller bin.
- **A rule whose terms are not computable stops the node** (ADR-0017 Decision 3). An unset
  face radius used to turn `candidate_x_visible` into `candidate_2d3d` silently; both halves
  of that asymmetry are now refusals.
- **`visible_fraction` has no cos(tilt) correction, and that is a measurement.** Dividing it
  out saturated the term at 1,00 on two thirds of detections where the truth said 28 %. The
  signal is real, so it comes back as the **`edge_on` term** instead — a factor of its own,
  linear in the angle, with no cosine in it (ADR-0018 Decision 5). It ships as a term and two
  rules, **not** as a default: like `clearance`, a factor nobody has scored on a real cell has
  no business deciding the order.
- No unit test could see the cosine bug, because each one fed the function a tilt it had
  computed itself. So every term ships with a scorer:

```bash
python3 scripts/score_term.py --term visible_fraction --labels labels.jsonl \
        --expected-radius-m 0.02095
python3 scripts/score_term.py --term edge_on --labels labels.jsonl --tool-max-tilt-deg 45
python3 scripts/score_term.py --demo              # self-test of the harness, not of a camera
python3 scripts/score_term.py --demo --demo-cos-bug            # it catches the historical bug
python3 scripts/score_term.py --demo --term edge_on --demo-constant-tilt   # and this one
```

### The grasp FACE is the pluggable unit, not the object

Four lines in the whole library carry a shape assumption, and they all live in
`ranking/faces.py`: the expected face area, the two robust extent estimators and the
extent ratio `pose_quality` gates on. It is the **face**, not the object, because a bottle
picked from above is a disc of the **cap's** radius and not of the body's — and the same
bottle gripped round the waist is a different face of the same object.

| face | parameters | extent terms it gates on | what it is for |
|---|---|---|---|
| `disc` | `expected_radius_m` | `radius_ratio` | a cap, a lid, a capsule's end |
| `rect` | `expected_half_a_m`, `expected_half_b_m` | `extent_a_ratio`, `extent_b_ratio` | a label panel, a carton's end, a machined pad |
| `cylinder_side` | `expected_radius_m`, `expected_height_m` | *none, deliberately* | a bottle picked from the **side** |

A new face is a class and a row in a registry; no fit function changes. `rect` measures its
two axes separately and gates on both, which catches a mask that leaked along one axis —
MEASURED, a 40×40 surface where a 60×40 face was configured halves `pose_quality` to 0,334
while a disc of the mean radius still scores 0,825.

`cylinder_side` is the honest half-face. Its area is sound — the silhouette, `2r × h`
projected, which survives the one rotation a bottle in a crate is guaranteed to have, and
reads 0,958–0,975 on a fully visible cylinder across `r/z` from 0,008 to 0,054. Its
**extents are not**: a plane fit keeps only the band where the curve stays inside its
tolerance, `sqrt(2·tol·r)` = 11,40 mm on a 32,5 mm bottle, so the transverse "radius" reads
0,38 of the truth and a `rect` face configured with the honest numbers scores that bottle
exactly 0.000 — a refusal. So it declares **no extent terms at all** until there is a
cylinder *fit* to measure them. A half-done face that silently mis-measures is worse than
none.

### A face declares what it needs; a configurator asks

`face_schema(name)` returns the three things that make a face configurable — the parameters
it reads, the extent terms it produces, the candidate terms it unlocks:

```python
>>> face_schema('rect')
{'name': 'rect', 'params': ('expected_half_a_m', 'expected_half_b_m'),
 'extent_term_keys': ('extent_a_ratio', 'extent_b_ratio'),
 'enables_terms': ('visible_fraction',)}
```

Nothing outside `faces.py` names a face parameter (ADR-0018 Decision 6). The asymmetry that
forced this was verified: a hand-written `rect` face was **accepted by the runtime and
refused by the Setup Assistant**, whose ranking-parameter list was a fixed four-tuple that
never wrote `grasp_face` — the key this package already read — and whose computable-terms
mirror hardcoded `expected_radius_m`. Keep a second copy of the answer and every new face
family becomes a two-repo change.

Sizes are **metres, and the name says so**, because the unit error does not raise: a metres
field holding millimetres just makes `visible_fraction` a constant. Anything outside
0,1 mm … 1 m is refused at start-up (ADR-0018 Decision 4); `0` stays the honest *unknown*,
and the rule that needed it is then refused rather than silently multiplied by one.

### The configuration vocabulary is exported, not asked for at generation

`trainit_perception.vocabulary.export()` reads this package's built-in tables into one plain
record (schema 1): the methods and their kind, what each method's detections carry, the colour
mask's own parameter schema, the rules and the terms each multiplies, what makes a term computable
(`ranking.rules.TERM_REQUIRES`, held to `computable_terms` by a test over every configuration), the
faces and the parameters each is sized by, the default face, the size band, the version floors.
The Setup Assistant commits a snapshot of that record with an explicit refresh and reads only the
snapshot when it generates, so a bundle does not depend on which perception package happens to be
installed (ADR-0021, which supersedes ADR-0018 Decision 6 for the vocabulary). Names, kinds and
requirements cross; notes, formulas, complements, gate values and extent terms never do. A rule or
face registered at run time from outside the package is not exported.

### What stays out

The **loop** — the retry over candidates, the stall recovery, the refill. That is the
cell's business logic, it is a behaviour tree the Setup Assistant generates (ADR-0016
Decision 4), and pulling it in here is the one move that would turn this into custom
engineering wearing a framework's name.

## Run

```bash
ros2 run trainit_perception detector_node --ros-args \
    -p config_file:=/path/to/perception.yaml -p detector_name:=cube
ros2 topic echo /perception/cube/detections      # BEST FIRST: [0] is the chosen candidate
ros2 topic echo /perception/cube/quality         # the per-candidate terms, same order
ros2 service call /perception/cube/detect std_srvs/srv/Trigger
```

## Methods

| method | goal | orientation | ranks by | status |
|---|---|---|---|---|
| `color_mask` | locate | no (top-down grasp) | score, box_overlap, visible_fraction | v0.5 |
| fiducial (ArUco) | locate | full 6-DoF | — | next |
| learned model | locate / classify | per model | + pose_quality, clearance | later |

A detector declares what it produces per detection (`Detector.PROVIDES`), and that is
what decides which terms are computable and therefore which rules it can be configured
with. A colour mask runs no 3D fit, so `pose_quality` is not computable for it and a
rule that needs one is refused rather than ignored.

Design: `fr3wml_digital_twin/docs/design_perception_integration.md`.
