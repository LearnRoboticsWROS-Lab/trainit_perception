"""The ONE place a frame's candidate terms are assembled, refused and permuted.

Not a node, not a behaviour tree, not the application: the DETECTOR ranks (ADR-0017 Decision 1).
`candidate_rule` is a per-detector parameter that flows verbatim from the Setup Assistant's form
to `perception.yaml` to `make_detector`, and its only effect is the permutation applied before the
array goes out. Putting it on `DetectObject` instead would make the motion runtime learn what
`visible_fraction` means, which is the boundary ADR-0016 Decision 1 exists to keep.

It is ONE place on purpose. The term assembly existed in three divergent copies inside the
customer package that first measured it, one of which applied the rank with a raw sort that
bypassed the rule check -- and a measured rule drifting across copies is exactly the failure a
framework prevents.

WHAT IT DOES, in the order it does it:

    1. refuse       a fit whose `pose_quality` is exactly 0.000 is not a low score, it is one of
                    the four gates clipped flat, and the surface is not the object. Dropped
                    whatever `min_pose_quality` says.
    2. per-detection score, pose_quality and the fit's own terms, obstruction/clearance, edge_on
    3. per-frame    box_overlap (a box can only be judged against the other boxes) and
                    visible_fraction, then the value of every rule that is computable
    4. permute      best first by the configured rule, ONE permutation for every list that goes
                    out with the frame
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .faces import expected_area_px, face_params, make_face
from .fit3d import approach_obstruction
from .rules import (AUTO, PROVIDES_BOX, PROVIDES_FIT3D, PROVIDES_MASK_AREA, PROVIDES_SCORE,
                    RULES, computable_rules, computable_terms, rank_order, resolve_rule)
from .terms import (PoseQualityGates, approach_clearance, box_overlap, edge_on, pose_quality,
                    visible_fraction)

if TYPE_CHECKING:                      # no runtime import: `ranking` never imports `detectors`
    from ..detectors.base import Detection


# WHICH RELEASE EACH RANKING PARAMETER ARRIVED IN, so a configurator can pin the runtime it needs
# instead of remembering. The failure this prevents is SILENT and specific: a parameter written
# into a bundle by a newer Setup Assistant MERGES into an older runtime's params dict and is
# ignored -- no exception, no log, just a knob that does nothing. A new FACE name is not like
# that and needs no pin: `make_face` raises on a name it does not know, so the node stops at
# start-up and says which faces it has. The face's own size parameters are listed anyway, because
# they are only safe through that loud failure and a reader should not have to know it.
#
# Add a row here when a release adds a parameter. That is the whole maintenance, and it lives
# beside the defaults it describes rather than in whatever generator happens to consume it.
PARAM_SINCE: Dict[str, str] = {
    'candidate_rule': '0.3.0',        # ADR-0017: the rank is a detector parameter
    'grasp_face': '0.3.0',
    'expected_radius_m': '0.3.0',     # DiscFace, and later CylinderSideFace's transverse radius
    'tool_radius_m': '0.3.0',
    'tool_reach_mm': '0.3.0',
    'min_pose_quality': '0.3.0',      # on top of the 0.000 refusal, never instead of it
    # ADR-0017 Decision 4: the four gates that make "exactly 0.000 is a REFUSAL" true. They are
    # configuration and not function defaults precisely so a shipped detector's refusal threshold
    # can be tuned and reproduced from a bundle.
    'pose_quality_min_coverage': '0.3.0',
    'pose_quality_min_inliers': '0.3.0',
    'pose_quality_max_rms_mm': '0.3.0',
    'pose_quality_radius_tol': '0.3.0',
    'tool_max_tilt_deg': '0.4.0',     # ADR-0018: the EDGE-ON term
    'expected_half_a_m': '0.4.0',     # RectFace            } safe through make_face's refusal,
    'expected_half_b_m': '0.4.0',     # RectFace            } declared so nobody has to know that
    'expected_height_m': '0.4.0',     # CylinderSideFace    }
}


def param_since(param: str) -> str:
    """The earliest release that understands `param`, or '' when this runtime never heard of it."""
    return PARAM_SINCE.get(param, '')


def ranking_defaults(provides: Iterable[str] = (PROVIDES_SCORE,),
                     grasp_face: str = 'disc') -> Dict[str, object]:
    """The ranking parameters a detector that provides `provides` should expose, with defaults.

    Only the ones that can matter: a detector with no 3D fit has no use for a tool radius, and a
    knob that cannot do anything is a knob a configurator should not render. Merge the result into
    the detector's own `defaults()`; the profile YAML documents them for the Setup Assistant's
    form (ADR-0017 Decision 1).

    THIS IS ALSO THE ANSWER TO "what are the ranking parameters" (ADR-0018 Decision 6). A
    generator must not keep its own fixed list: the one that did never wrote `grasp_face` -- the
    key this package already read -- so a hand-written `rect` face was accepted by the runtime
    and refused by the Setup Assistant. The face's own parameters come from `face_params`, so a
    face registered from outside this package appears here too, and `faces.all_face_params()`
    gives the full set across every face when a caller has to know which keys it may clear.
    """
    have = set(provides)
    out: Dict[str, object] = {'candidate_rule': AUTO}
    if PROVIDES_MASK_AREA in have:
        out['grasp_face'] = grasp_face
        for key in face_params(grasp_face):
            out[key] = 0.0                       # 0 = size unknown -> visible_fraction refused
    if PROVIDES_FIT3D in have:
        out['tool_radius_m'] = 0.0               # 0 = the clearance term is switched off
        out['tool_reach_mm'] = 0.0               # the tool's grip window; 0 = switched off
        out['tool_max_tilt_deg'] = 0.0           # the tilt the tool can still hold; 0 = off
        out['min_pose_quality'] = 0.0            # on top of the 0.000 refusal, never instead of it
        out.update(PoseQualityGates().as_params())
    return out


class Ranker:
    """Built from a detector's params. Immutable per detector instance, like the params."""

    def __init__(self, *, rule: str, face, gates: PoseQualityGates, provides: Iterable[str],
                 computable: Iterable[str], tool_radius_m: float = 0.0,
                 tool_reach_mm: float = 0.0, tool_max_tilt_deg: float = 0.0,
                 min_pose_quality: float = 0.0):
        self.rule = rule
        self.face = face
        self.gates = gates
        self.provides = frozenset(provides)
        self.computable = frozenset(computable)
        self.tool_radius_m = float(tool_radius_m)
        self.tool_reach_mm = float(tool_reach_mm)
        self.tool_max_tilt_deg = float(tool_max_tilt_deg)
        self.min_pose_quality = float(min_pose_quality)

    @classmethod
    def from_params(cls, params: Mapping, *,
                    provides: Iterable[str] = (PROVIDES_SCORE,)) -> 'Ranker':
        """Read the configuration, and REFUSE it here if the rule is not computable.

        This is the call that stops the node (ADR-0017 Decision 3): it runs inside
        `make_detector`, which runs inside the node's constructor, before a single frame is
        processed and before anything downstream could mistake an unranked array for a ranking.
        """
        face = make_face(str(params.get('grasp_face', 'disc') or 'disc'), params)
        computable = computable_terms(params, provides)
        return cls(rule=resolve_rule(params.get('candidate_rule', AUTO), computable=computable),
                   face=face, gates=PoseQualityGates.from_params(params), provides=provides,
                   computable=computable,
                   tool_radius_m=float(params.get('tool_radius_m', 0.0) or 0.0),
                   tool_reach_mm=float(params.get('tool_reach_mm', 0.0) or 0.0),
                   tool_max_tilt_deg=float(params.get('tool_max_tilt_deg', 0.0) or 0.0),
                   min_pose_quality=float(params.get('min_pose_quality', 0.0) or 0.0))

    def describe(self) -> str:
        """One line for a node's start-up log: what it will rank by, and what it can compute."""
        return (f'candidates published best first by {self.rule}; terms: '
                f'{", ".join(sorted(self.computable))}; face: {self.face!r}; '
                f'rules available here: {", ".join(computable_rules(self.computable))}')

    # --- step 1: the refusal --------------------------------------------------------------
    def keep(self, det: 'Detection') -> bool:
        """False for a detection a detector must not offer as a grasp at all.

        A fit that failed, or a `pose_quality` of exactly 0.000 -- which is a REFUSAL and not a
        low score: one of the four gates clipped flat, so whatever that surface is, it is not the
        object. MEASURED: a mask on a bin's wall fitted a 78 mm "capsule" 0,4 m outside the bin
        and scored exactly 0.000, while the poorest fit on a real object scored 0.085.
        `min_pose_quality` is the FURTHER threshold a consumer may ask for on top.
        """
        fit = getattr(det, 'fit', None)
        if fit is None:
            return True                      # no 3D fit ran: there is nothing to refuse it on
        if not fit.ok:
            return False
        pq = pose_quality(fit, gates=self.gates, extent_keys=self.face.extent_term_keys)
        return pq > 0.0 and pq >= self.min_pose_quality

    # --- steps 2 and 3: the terms ---------------------------------------------------------
    def terms(self, dets: Sequence['Detection'], K: np.ndarray,
              depth_m: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
        """One dict per detection, in the detections' own order. Never blended, never rounded."""
        out: List[Dict[str, float]] = []
        for d in dets:
            t: Dict[str, float] = {'score': float(d.score)}
            fit = getattr(d, 'fit', None)
            if fit is not None and fit.ok:
                t.update({k: float(v) for k, v in fit.quality.items()})
                t['pose_quality'] = pose_quality(fit, gates=self.gates,
                                                 extent_keys=self.face.extent_term_keys)
                a, b = fit.half_extents_m
                t['half_extent_a_mm'], t['half_extent_b_mm'] = a * 1e3, b * 1e3
                # HOW MUCH ROOM THE TOOL HAS. A term of its own and NOT multiplied into
                # pose_quality: that number is "is this pose well formed" and this is part of
                # "can the tool get there", which the layer owning the tool multiplies in. Only
                # this layer can compute it at all, because only this layer has the depth image.
                if self.tool_radius_m > 0.0 and depth_m is not None:
                    obs = approach_obstruction(fit, depth_m, K, tool_radius_m=self.tool_radius_m)
                    t['obstruction_mm'] = obs            # NaN = "cannot tell", not "clear"
                    t['clearance'] = approach_clearance(obs, reach_mm=self.tool_reach_mm)
                # HOW EDGE-ON IT IS PRESENTED, against the tilt the tool can still hold. ONLY
                # from a fit whose method actually reported a tilt: `centroid` returns the camera
                # axis for every fit and declares `gives_orientation = False`, so reading
                # `tilt_deg` off it would hand every candidate a 0 and make the term a CONSTANT
                # -- which is a rule multiplying by one, wearing the name of a measurement
                # (ADR-0017 Decision 6). Absent is loud: `rank_order` refuses a rule whose term
                # no detection carries.
                if self.tool_max_tilt_deg > 0.0 and 'tilt_deg' in fit.quality:
                    t['edge_on'] = edge_on(fit.tilt_deg, max_tilt_deg=self.tool_max_tilt_deg)
            out.append(t)

        # OCCLUSION, once the whole frame is known -- a box can only be judged against the other
        # boxes. Published as separate terms and never blended: two objects side by side have
        # overlapping BOXES and two entirely visible FACES, while one half under another has a
        # small overlap and a face that is half gone. Correlation between the two: +0,54.
        if dets and PROVIDES_BOX in self.provides:
            boxes = [tuple(d.box_px) for d in dets]
            if all((b[2] - b[0]) > 0 and (b[3] - b[1]) > 0 for b in boxes):
                for t, o in zip(out, box_overlap(boxes)):
                    t['box_overlap'] = float(o)
        if PROVIDES_MASK_AREA in self.provides and self.face.has_known_area:
            for t, d in zip(out, dets):
                t['visible_fraction'] = visible_fraction(float(d.area_px),
                                                         expected_area_px(self.face, d, K))
        # Every rule this detector can compute goes out as a term under its own name, so a tuner
        # can see what the order WOULD have been under another rule without re-deriving it. A
        # rule with no factors (`detector_order`) is a constant, and a constant is not a term.
        for name in computable_rules(self.computable):
            r = RULES[name]
            if not r.factors:
                continue
            for t in out:
                if r.requires() <= set(t):
                    t[name] = r.value(t)
        return out

    # --- step 4: the permutation ----------------------------------------------------------
    def order(self, terms: Sequence[Mapping[str, float]]) -> List[int]:
        return rank_order(terms, self.rule)

    def apply(self, dets: Sequence['Detection'], K: np.ndarray,
              depth_m: Optional[np.ndarray] = None) -> List['Detection']:
        """Refuse, measure, order. Returns the detections BEST FIRST, terms merged into `extra`.

        May return FEWER detections than it was given: see `keep`. The terms ride on `extra`,
        which `msgs.to_quality_payload` publishes on the sidecar in the same permutation -- the
        dict the publisher used to drop (ADR-0017 Decision 5).
        """
        kept = [d for d in dets if self.keep(d)]
        terms = self.terms(kept, K, depth_m)
        order = self.order(terms)
        for i in order:
            kept[i].extra.update(terms[i])
        return [kept[i] for i in order]
