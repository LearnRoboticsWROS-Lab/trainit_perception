"""The candidate TERMS: "which one should I pick", and WHY -- as pure functions, no ROS.

The detector says where objects are. It does not say which one to take, and in a bin that is the
decision that makes or breaks the cell: an object with another lying across it comes up with its
neighbour, or does not come up at all.

Four measures, kept apart on purpose so they can be COMPARED rather than blended. Each is one
number per detection, each is closed form from what the detector already produced, and each can be
read off a picture by a person:

    score             2D   the detector's own confidence. Not computed here; it is given.
    pose_quality      3D   inliers, residual, depth coverage, fitted extent. P(pose well formed).
    box_overlap       2D   how much of THIS object's box another object's box covers. Needs only
                           boxes, so it works with ANY detector, including one that returns no
                           mask and no depth. The intuitive one.
    visible_fraction  2D+  the mask's area against the area the FACE's own size says it should
                           have. Catches an object half-buried under another, which overlapping
                           boxes alone do not distinguish from two objects merely side by side.
    clearance         3D   from `fit3d.approach_obstruction`: how far anything stands into the
                           tool's path. Needs depth AND the tool's radius AND its grip window.
    edge_on           3D   how far the fitted face is turned away from the lens, against the
                           tilt the TOOL can still hold. The owner's second signal: well
                           visible but presented EDGE-ON is a bad candidate.

They do not measure the same thing and the cheapest is not obviously the worst: measured, the
correlation between `1 - box_overlap` and `visible_fraction` is only **+0,54**. In a full bin
almost every box overlaps another, so a rule like "reject any overlap" rejects everything -- the
threshold has to come from the measured distribution, not from intuition, and that is why this
module computes the numbers and does not pick for you.

WHICH of them decides the order is configuration (`candidate_rule`, ADR-0014), and the ORDER the
detections are published in is where the answer is carried (ADR-0016 Decision 1,
`rules.rank_order`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

#: what `pose_quality` gates on by default -- the terms `faces.DiscFace` produces. A caller with a
#: different face passes `face.extent_term_keys`.
DEFAULT_EXTENT_KEYS: Tuple[str, ...] = ('radius_ratio',)


def box_overlap(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """Per box, the largest fraction of ITS OWN area that another box covers. 0 = nothing on it.

    Deliberately asymmetric: what matters is how much of THIS object is involved with another, not
    how much of the pair's union they share. A small object under a big one is nearly 100 %
    covered while the IoU of the two is small, and it is the small one that cannot be picked.

    Boxes are `(u0, v0, u1, v1)` in pixels. MEASURED over 20 pours of a 25-object bin: median
    overlap 28 %, p90 56 % -- which is why the 20 % REFUSAL this term invites is NOT shipped
    (ADR-0017 Decision 4: measured, it halves the pick rate).
    """
    n = len(boxes)
    out = np.zeros(n)
    if n < 2:
        return out
    b = np.asarray(boxes, float)
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    for i in range(n):
        if area[i] <= 0:
            continue
        iw = np.maximum(0.0, np.minimum(b[:, 2], b[i, 2]) - np.maximum(b[:, 0], b[i, 0]))
        ih = np.maximum(0.0, np.minimum(b[:, 3], b[i, 3]) - np.maximum(b[:, 1], b[i, 1]))
        inter = iw * ih
        inter[i] = 0.0
        out[i] = float(inter.max() / area[i])
    return np.clip(out, 0.0, 1.0)


def visible_fraction(mask_area_px: float, expected_area_px: Optional[float],
                     tilt_deg: float = 0.0) -> float:
    """How much of the face the camera gets: 1 = all of it, 0.5 = half of it gone.

    `expected_area_px` comes from the GRASP FACE (`faces.expected_area_px`): for a disc of radius
    `r` seen at depth `z`, pi * (f * r / z)^2. The mask's area against THAT is the fraction, and it
    folds two causes together on purpose: what another object covers, and what the face loses to
    its own tilt. A picker wants both -- an object at 60 degrees gives the cup less to seal on
    whether it is covered or merely leaning.

    `None` means the face has no configured size, so the question cannot be asked. The answer is
    then 1.0 **and the term is not computable**: `rules.computable_terms` leaves it out and a rule
    that needs it STOPS the node (ADR-0017 Decision 3). Before that, an unset face radius turned
    `candidate_x_visible` into `candidate_2d3d` silently -- VERIFIED by execution: the two read
    0,7000 and 0,7000, while the `radius_ratio` gate vanished from the quality dict and
    `pose_quality` ROSE from 0,8913 to 1,0000 on the same frame.

    NO cos(tilt) CORRECTION, AND THAT IS A MEASUREMENT, NOT A SIMPLIFICATION. Dividing by the
    cosine is the textbook way to separate occlusion from foreshortening, and here it destroys the
    number. VALIDATED against the twin's own labels over 224 detections in 12 frames: the mask
    area agrees with the truth's to a correlation of +0,937 (ratio 1,004) and the expected circle
    to 1,003, so both halves are right -- but the fit's tilt in a bin runs to a MEDIAN of 33
    degrees and a p90 of 63, and dividing by cos(63 deg) inflates the fraction by 2,2x. With the
    correction the term saturated: 65 % of detections read 1,00 against the truth's 28 %, and on
    the 98 objects the truth called less than 80 % visible it still said 0,96. Without it, the
    same 224 detections correlate with the truth at +0,94, median error 0,03.

    `tilt_deg` is kept in the signature, unused, because the argument for leaving it out belongs
    next to the place a reader would otherwise put it back. `scripts/score_term.py` is how the
    claim stays checked rather than remembered (ADR-0017 Decision 6).

    This is what overlapping boxes cannot tell you: two objects side by side have boxes that
    overlap and faces that are both completely visible.
    """
    if expected_area_px is None or expected_area_px <= 1.0:
        return 1.0
    return float(np.clip(float(mask_area_px) / float(expected_area_px), 0.0, 1.0))


def approach_clearance(obstruction_mm: float, *, reach_mm: float) -> float:
    """P(the tool can get there), from the obstruction alone: 1 clear, 0 filled or UNKNOWN.

    `reach_mm` is the TOOL's own grip window and is required, not defaulted: it is a property of
    the gripper (a suction cup in Isaac reports it as `isaac:maxGripDistance`, e.g. 8 mm), and a shipped detector whose
    clearance threshold is a Python default cannot be tuned in a configurator or reproduced from a
    bundle (ADR-0017 Decision 4). An obstruction of that much or more and the tool cannot come
    within grasping distance of the face however well the pose was fitted.

    A NaN obstruction means the depth could not answer the question, and the answer to a question
    that could not be asked is 0, not 1.

    MEASURED, and the reason this ships as a term and NOT as a ranking default: 0,00 on every
    candidate the picker actually chose over 60 pours, while discriminating across the frame on 12
    of 27 detections. It buys nothing at the top of THAT ranking and would bite with a wider tool
    or a fuller bin -- a framework that shipped only the terms with a measured gain in one cell
    would throw away the terms the next cell needs.
    """
    if not np.isfinite(obstruction_mm):
        return 0.0
    if reach_mm <= 0:
        return 1.0
    return float(np.clip(1.0 - max(obstruction_mm, 0.0) / reach_mm, 0.0, 1.0))


def edge_on(tilt_deg: float, *, max_tilt_deg: float) -> float:
    """HOW EDGE-ON the face is presented: 0 square to the lens, 1 at or past the tool's limit.

    The owner's second signal, and the one the framework was missing: *"se l'oggetto e ben
    visibile nella zona in cui lo voglio prendere, allora e un buon candidato; se e ben visibile
    ma messo di taglio allora non e un buon candidato"* (ADR-0018 Decision 5). A BADNESS, like
    `box_overlap`, so it enters a rule complemented as `(1 - edge_on)` and the published number
    reads the way the sentence does: 0,82 means *strongly edge-on*.

    IT IS A TERM AND NOT A CORRECTION, AND THAT DISTINCTION IS THE WHOLE DECISION. The same
    signal was first tried as a correction -- dividing `visible_fraction` by `cos(tilt)` to
    separate occlusion from foreshortening -- and it DESTROYED the only term that measures: the
    fraction saturated at 1,00 on 65 % of detections where the labelled truth said 28 %, and on
    the 98 objects the truth called less than 80 % visible it still read 0,96 `[verified]`. A
    correction claims to know *how much* area the tilt costs. A term only says *how tilted it
    is*, which is what `GraspFit.tilt_deg` already computes, and it leaves the area measurement
    alone. `scripts/score_term.py --term edge_on` is how that stays checked rather than
    remembered (ADR-0017 Decision 6).

    LINEAR IN THE ANGLE, deliberately, and there is no cosine anywhere in it. An angle is what a
    person can read off a picture and argue with ("that bottle is lying at 50 degrees"), and the
    saturation point is a number the TOOL owns rather than a curve the framework chose: a
    compliant suction cup seals up to some tilt and not past it, a two-finger gripper has a
    different limit, and `max_tilt_deg` is where that belongs -- the same reasoning that made
    `approach_clearance` require the tool's grip window instead of defaulting it (ADR-0017
    Decision 4).

    `max_tilt_deg <= 0` SWITCHES THE TERM OFF (0,0 = no penalty), exactly as `reach_mm <= 0`
    switches the clearance off -- and `rules.computable_terms` then leaves `edge_on` out, so a
    rule that needs it is refused at start-up instead of multiplying by a constant.

    A NON-FINITE TILT IS 1,0, NOT 0,0: a tilt that could not be measured must not make a
    candidate look square to the lens. Same asymmetry as a NaN obstruction, and in the same
    direction -- the answer to a question that could not be asked is the pessimistic one.

    ONE HONEST LIMIT, and it is the failure this term could have: a 3D method that claims no
    orientation reports a tilt of 0 for every candidate, and a rule multiplying by a constant is
    not a ranking. `centroid` returns the camera axis by construction and declares
    `gives_orientation = False`, so `Ranker` computes this term only from a fit whose method
    actually reported a tilt, and the scorer's third check is exactly "it discriminates".
    """
    if max_tilt_deg <= 0:
        return 0.0                              # the term is switched off, not unknown
    if not np.isfinite(tilt_deg):
        return 1.0                              # could not tell -> do not call it square-on
    return float(np.clip(float(tilt_deg) / float(max_tilt_deg), 0.0, 1.0))


@dataclass(frozen=True)
class PoseQualityGates:
    """`pose_quality`'s four gates, as SERIALISABLE CONFIGURATION rather than function defaults.

    They are what makes "exactly 0.000 is a refusal and not a low score" true, so they decide
    which surfaces a detector will not offer at all -- and a refusal threshold that lives as a
    Python default cannot be tuned in the Setup Assistant or reproduced from a bundle (ADR-0017
    Decision 4). The values below are the measured ones; they travel in the detector's params
    dict under `PREFIX`, which is what puts them in `perception.yaml`.
    """
    min_coverage: float = 0.4
    min_inliers: float = 0.6
    max_rms_mm: float = 3.0
    radius_tol: float = 0.25

    PREFIX = 'pose_quality_'

    @classmethod
    def from_params(cls, params: Mapping) -> 'PoseQualityGates':
        """Read `pose_quality_min_coverage` and friends out of a flat params dict."""
        d = cls()
        kw = {f.name: float(params.get(cls.PREFIX + f.name, getattr(d, f.name)))
              for f in cls.__dataclass_fields__.values()}
        return cls(**kw)

    def as_params(self) -> Dict[str, float]:
        """Back to the flat, prefixed keys -- the defaults a profile YAML publishes."""
        return {self.PREFIX + k: v for k, v in asdict(self).items()}


def pose_quality(fit, *, gates: Optional[PoseQualityGates] = None,
                 extent_keys: Sequence[str] = DEFAULT_EXTENT_KEYS) -> float:
    """P(pose well formed): one number in 0..1 from the fit's own terms.

    Deliberately a product of soft gates rather than a learned score. Every term is something a
    person can read off and argue with, and when a pick fails the term that was low says why. It
    is a prior, not a success rate: what fraction of attempts actually succeed is measured by
    trying (ADR-0014 Decision 2).

    EXACTLY 0.000 IS A REFUSAL AND NOT A LOW SCORE. It means one of the gates clipped flat -- the
    masked depth covers almost nothing, almost none of it lies on the fitted surface, the residual
    is twice the tolerance, or the fitted extent is outside half-to-one-and-a-half of the face's.
    Whatever that surface is, it is not the object. MEASURED on the live cell (2026-10-01): one
    mask over a bin's wall edge fitted a 78 mm "capsule" 0,4 m OUTSIDE the bin and scored exactly
    0.000, while the poorest fit on a real object scored 0.085. The gates separate them with room
    to spare, which is why `Ranker` drops a 0.000 whatever `min_pose_quality` is set to.

    `extent_keys` is the face's own `extent_term_keys`, so the gate follows the face rather than
    assuming a radius (ADR-0017 Decision 2).
    """
    if not fit.ok:
        return 0.0
    g = gates or PoseQualityGates()
    q = fit.quality

    def soft(v: float, good: float, bad: float) -> float:
        # good == bad is ZERO tolerance, which is a step at that value -- not a gate switched off.
        # Returning 1.0 there made `max_rms_mm=0` and `radius_tol=0`, the values a reader reaches
        # for to make a gate strict, silently remove it instead.
        if good == bad:
            return 1.0 if (v <= good if good >= 0 else v >= good) else 0.0
        return float(np.clip((v - bad) / (good - bad), 0.0, 1.0))

    s = soft(q.get('depth_coverage', 1.0), 1.0, g.min_coverage * 0.5)
    if 'inlier_ratio' in q:
        s *= soft(q['inlier_ratio'], 1.0, g.min_inliers * 0.5)
    if 'rms_mm' in q:
        s *= soft(q['rms_mm'], 0.0, g.max_rms_mm * 2)
    for key in extent_keys:
        if key in q:
            s *= soft(abs(q[key] - 1.0), 0.0, g.radius_tol * 2)
    return float(s)
