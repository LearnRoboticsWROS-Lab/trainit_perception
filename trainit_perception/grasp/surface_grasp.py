"""SurfaceGrasp: measured surface geometry -> ranked, constrained, explicit grasp candidates (vNext M4).

ROS-free and pure: `propose(regions, depth_m, camera, config)` returns the same candidates, ids, terms, scores,
order and rejections for the same input (the RANSAC draws are seeded; no clock or hash order enters a result).
The node (`nodes/surface_grasp_node.py`) and the offline harnesses call this one function. Design record:
the overlay's `docs/engineering/M4_DESIGN.md`; the wire contract: ADR-0026 (`trainit_interfaces/GraspCandidate`).

THE LAYERS IT KEEPS APART. A REGION is 2-D evidence of an object (a mask, a class, a score); the SURFACE is what the
depth measures inside it (a plane: a point, a normal, extents, a fit quality); a CANDIDATE is a proposed TOOL
placement on that surface for ONE named end effector / grasp site. No object pose is claimed (`object_id` stays
empty) and nothing here knows a robot: reachability and collision are NOT EVALUATED, by design -- MoveIt, through
the motion runtime, resolves them by trying (ADR-0014).

CONVENTIONS (M4_DESIGN.md §6-§10), all in the OBSERVATION frame O = the colour optical frame (+X right, +Y down,
+Z forward, the camera at the origin):

- the contact point p is ON the measured surface: the plane's inlier centroid moved along the normal to the median
  height of the region's depth points within the contact radius of the approach axis (0 on a plane);
- the normal n points TOWARD THE OBSERVER along the view ray (n . (-p) > 0): the canonical sign policy, recorded in
  every candidate (`normal_policy`), never an implicit flip by n_z;
- the approach a = -n, the unit direction the tool travels, INTO the surface (ADR-0026);
- the full orientation R = MinRot(c -> a) * R_ref: the provider's reference (+Z = the approach-cone axis c, +X =
  `reference_x_axis` projected) tilted as little as possible -- the M3.6-1 `align_approach` rule, on a reference
  this provider owns. R e_z = a exactly;
- PreGrasp = p - standoff * a (ADR-0026 Decision 4).

FAIL CLOSED. A region that cannot support a valid candidate becomes a `Rejection` with a deterministic code (the
first failing check in `CHECK_ORDER`), never a candidate with an assumed normal, a clamped angle or a substituted
end effector. A term that was not evaluated is listed in `not_evaluated`, never scored as 1.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ranking.faces import make_face
from ..ranking.fit3d import GraspFit, approach_obstruction, plane_ransac_points
from ..ranking.terms import (PoseQualityGates, approach_clearance, box_overlap, pose_quality,
                             visible_fraction)
from .camera import CameraModel
from .face_centre import KnownDiscParams, known_disc_centre

PROVIDER_ID = 'tp.surface_grasp'
NORMAL_POLICY = 'toward_observer'
#: never evaluated by a perception provider: the motion layer resolves them by trying (ADR-0014)
ALWAYS_NOT_EVALUATED = ('reachability', 'collision')
#: the grasp modes this provider can propose a contact for
GRASP_MODES = ('suction',)

# --- rejection codes (M4_DESIGN.md §14), in the order they are checked ---------------------------------------------
REGION_EMPTY = 'REGION_EMPTY'
INSUFFICIENT_DEPTH = 'INSUFFICIENT_DEPTH'
FIT_FAILED = 'FIT_FAILED'
GEOMETRY_INVALID = 'GEOMETRY_INVALID'
FACE_FIT_FAILED = 'FACE_FIT_FAILED'          # 0.8.0: known_shape_fit could not determine the face's centre
NORMAL_GRAZING = 'NORMAL_GRAZING'
APPROACH_OUTSIDE_CONE = 'APPROACH_OUTSIDE_CONE'
POSE_QUALITY_REFUSED = 'POSE_QUALITY_REFUSED'
POSE_QUALITY_LOW = 'POSE_QUALITY_LOW'
SURFACE_TOO_SMALL = 'SURFACE_TOO_SMALL'
CONTACT_UNSUPPORTED = 'CONTACT_UNSUPPORTED'
VISIBILITY_LOW = 'VISIBILITY_LOW'
CLEARANCE_UNKNOWN = 'CLEARANCE_UNKNOWN'
CLEARANCE_LOW = 'CLEARANCE_LOW'
ORIENTATION_UNDEFINED = 'ORIENTATION_UNDEFINED'
BEYOND_MAX_CANDIDATES = 'BEYOND_MAX_CANDIDATES'
CHECK_ORDER = (REGION_EMPTY, INSUFFICIENT_DEPTH, FIT_FAILED, GEOMETRY_INVALID, FACE_FIT_FAILED, NORMAL_GRAZING,
               APPROACH_OUTSIDE_CONE,
               POSE_QUALITY_REFUSED, POSE_QUALITY_LOW, SURFACE_TOO_SMALL, VISIBILITY_LOW, CLEARANCE_UNKNOWN,
               CLEARANCE_LOW, CONTACT_UNSUPPORTED, ORIENTATION_UNDEFINED, BEYOND_MAX_CANDIDATES)

#: the ranking terms, in the order they are reported
TERMS = ('detection', 'alignment', 'fit', 'seal', 'visibility', 'clearance', 'nonoverlap')

# the cone boundary is inclusive; a numerical epsilon so an angle computed AT the boundary is not refused
_CONE_EPS_RAD = 1e-9


class ConfigError(ValueError):
    """A SurfaceGrasp configuration this provider refuses to run with."""


# =================================================================================================================
# configuration
# =================================================================================================================

#: The provider's OWN parameters: the configuration vocabulary a configurator may write (exported by
#: `trainit_perception.vocabulary.export()` under `grasp_providers`). `required` = no default crosses the package
#: boundary: the value is the cell's (a tool's number, an envelope), never the framework's.
PARAMETERS: Dict[str, Dict[str, object]] = {
    'eef': {'type': 'str', 'required': True},
    'site': {'type': 'str', 'required': True},
    'grasp_mode': {'type': 'str', 'choices': list(GRASP_MODES)},
    'contact_radius_m': {'type': 'float', 'min': 0.0, 'max': 0.5, 'required': True},
    'approach_cone_deg': {'type': 'float', 'min': 0.0, 'max': 90.0, 'required': True},
    'approach_cone_axis': {'type': 'vec3'},
    'reference_x_axis': {'type': 'vec3'},
    'max_view_angle_deg': {'type': 'float', 'min': 0.0, 'max': 90.0},
    'standoff_m': {'type': 'float', 'min': 0.0, 'max': 2.0, 'required': True},
    'standoff_min_m': {'type': 'float', 'min': 0.0, 'max': 2.0},
    'standoff_max_m': {'type': 'float', 'min': 0.0, 'max': 2.0},
    'max_candidates': {'type': 'int', 'min': 1, 'max': 100},
    'min_points': {'type': 'int', 'min': 3, 'max': 100000},
    'inlier_mm': {'type': 'float', 'min': 0.0, 'max': 50.0},
    'ransac_iters': {'type': 'int', 'min': 1, 'max': 10000},
    'seed': {'type': 'int', 'min': 0, 'max': 2147483647},
    'z_min_m': {'type': 'float', 'min': 0.0, 'max': 20.0},
    'z_max_m': {'type': 'float', 'min': 0.0, 'max': 20.0},
    'min_pose_quality': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'min_visible_fraction': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'tool_radius_m': {'type': 'float', 'min': 0.0, 'max': 0.5},
    'tool_reach_mm': {'type': 'float', 'min': 0.0, 'max': 500.0},
    'min_clearance': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'grasp_face': {'type': 'str'},
    'expected_radius_m': {'type': 'float', 'min': 0.0, 'max': 2.0},
    'expected_half_a_m': {'type': 'float', 'min': 0.0, 'max': 2.0},
    'expected_half_b_m': {'type': 'float', 'min': 0.0, 'max': 2.0},
    'expected_height_m': {'type': 'float', 'min': 0.0, 'max': 2.0},
    'pose_quality_min_coverage': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'pose_quality_min_inliers': {'type': 'float', 'min': 0.0, 'max': 1.0},
    'pose_quality_max_rms_mm': {'type': 'float', 'min': 0.0, 'max': 100.0},
    'pose_quality_radius_tol': {'type': 'float', 'min': 0.0, 'max': 10.0},
    # M4.5d (0.8.0): WHERE on the face the contact goes (grasp/face_centre.py)
    'centre_estimator': {'type': 'str', 'choices': ['inlier_centroid', 'known_shape_fit'], 'since': '0.8.0'},
    'face_occlusion_mm': {'type': 'float', 'min': 0.0, 'max': 100.0, 'since': '0.8.0'},
    'face_mask_dilate_px': {'type': 'int', 'min': 0, 'max': 20, 'since': '0.8.0'},
    'face_min_coverage': {'type': 'float', 'min': 0.0, 'max': 1.0, 'since': '0.8.0'},
    'face_max_outside': {'type': 'float', 'min': 0.0, 'max': 1.0, 'since': '0.8.0'},
    'face_max_ambiguity_mm': {'type': 'float', 'min': 0.0, 'max': 100.0, 'since': '0.8.0'},
    'face_min_arc_deg': {'type': 'float', 'min': 0.0, 'max': 360.0, 'since': '0.8.0'},
    'face_max_rms_mm': {'type': 'float', 'min': 0.0, 'max': 50.0, 'since': '0.8.0'},
    'face_radius_tol': {'type': 'float', 'min': 0.0, 'max': 1.0, 'since': '0.8.0'},
}

#: the contact's position on the face: the plane's inlier centroid (M4), or the centre of the KNOWN face fitted to
#: the partial depth evidence (M4.5d; needs grasp_face: disc and expected_radius_m)
CENTRE_ESTIMATORS = ('inlier_centroid', 'known_shape_fit')


def defaults() -> Dict[str, object]:
    """Every parameter with its default. A required one defaults to its 'unset' value (empty / 0), which
    `SurfaceGraspConfig.from_params` refuses -- so it must be configured."""
    g = PoseQualityGates()
    return {
        'eef': '', 'site': '', 'grasp_mode': 'suction',
        'contact_radius_m': 0.0, 'approach_cone_deg': 0.0,
        'approach_cone_axis': [0.0, 0.0, 1.0], 'reference_x_axis': [1.0, 0.0, 0.0],
        # a SENSOR-validity threshold, not a tool envelope: a plane seen within 10 deg of edge-on is not measured
        'max_view_angle_deg': 80.0,
        'standoff_m': 0.0, 'standoff_min_m': 0.001, 'standoff_max_m': 0.5,
        'max_candidates': 10, 'min_points': 50, 'inlier_mm': 2.0, 'ransac_iters': 120, 'seed': 0,
        'z_min_m': 0.05, 'z_max_m': 3.0,
        'min_pose_quality': 0.0, 'min_visible_fraction': 0.0,
        'tool_radius_m': 0.0, 'tool_reach_mm': 0.0, 'min_clearance': 0.0,
        'grasp_face': 'disc', 'expected_radius_m': 0.0, 'expected_half_a_m': 0.0, 'expected_half_b_m': 0.0,
        'expected_height_m': 0.0,
        'centre_estimator': 'inlier_centroid',
        'face_occlusion_mm': 3.0, 'face_mask_dilate_px': 0, 'face_min_coverage': 0.35, 'face_max_outside': 0.6,
        'face_max_ambiguity_mm': 3.0, 'face_min_arc_deg': 150.0, 'face_max_rms_mm': 1.0, 'face_radius_tol': 0.1,
        **g.as_params(),
    }


def _vec3(value, name: str, problems: List[str]) -> Optional[np.ndarray]:
    try:
        v = np.asarray([float(x) for x in value], float)
    except (TypeError, ValueError):
        problems.append(f'{name} must be three numbers, got {value!r}')
        return None
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
        problems.append(f'{name} must be a finite, non-zero 3-vector, got {value!r}')
        return None
    return v / np.linalg.norm(v)


def reference_rotation(axis, x_axis) -> Optional[np.ndarray]:
    """The provider's reference frame R_ref (columns = axes): +Z = the unit `axis`, +X = `x_axis` projected onto the
    plane normal to it. None when `x_axis` is parallel to `axis` (the rotation about the approach is undefined)."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    xref = np.asarray(x_axis, float)
    x = xref - (xref @ axis) * axis
    if np.linalg.norm(x) < 1e-6:
        return None
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(axis, x), axis], 1)


@dataclass(frozen=True)
class SurfaceGraspConfig:
    eef: str
    site: str
    grasp_mode: str
    contact_radius_m: float
    cone_deg: float
    cone_axis: Tuple[float, float, float]
    reference: Tuple[Tuple[float, float, float], ...]     # R_ref as 3 rows (a rotation matrix, columns = axes)
    max_view_angle_deg: float
    standoff_m: float
    standoff_min_m: float
    standoff_max_m: float
    max_candidates: int
    min_points: int
    inlier_mm: float
    ransac_iters: int
    seed: int
    z_min_m: float
    z_max_m: float
    min_pose_quality: float
    min_visible_fraction: float
    tool_radius_m: float
    tool_reach_mm: float
    min_clearance: float
    face: object = field(compare=False)
    gates: PoseQualityGates = field(default_factory=PoseQualityGates)
    centre_estimator: str = 'inlier_centroid'
    known_disc: Optional[KnownDiscParams] = field(default=None, compare=False)
    params: Dict[str, object] = field(default_factory=dict, compare=False)

    @property
    def R_ref(self) -> np.ndarray:
        return np.asarray(self.reference, float)

    @classmethod
    def from_params(cls, params: Mapping) -> 'SurfaceGraspConfig':
        """Merge onto `defaults()` and VALIDATE. Raises `ConfigError` naming every problem: a provider that
        cannot state its end effector, its envelope or a valid standoff does not start (M4_DESIGN.md §8-§10)."""
        unknown = sorted(set(params) - set(PARAMETERS))
        p = dict(defaults())
        p.update(params or {})
        problems: List[str] = []
        if unknown:
            problems.append(f'unknown parameter(s) {", ".join(unknown)} (have: {", ".join(sorted(PARAMETERS))})')

        def num(key: str) -> float:
            try:
                v = float(p[key])
            except (TypeError, ValueError):
                problems.append(f'{key} must be a number, got {p[key]!r}')
                return float('nan')
            if not math.isfinite(v):
                problems.append(f'{key} must be finite, got {p[key]!r}')
            return v

        def integer(key: str) -> int:
            v = p[key]
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) and not (
                    isinstance(v, float) and v.is_integer()):
                problems.append(f'{key} must be a whole number, got {v!r}')
                return 0
            return int(v)

        eef, site = str(p['eef'] or ''), str(p['site'] or '')
        if not eef:
            problems.append('eef is required: the end effector these candidates are for (no default: a candidate '
                            'never claims an end effector it was not configured for)')
        if not site:
            problems.append('site is required: the grasp site these candidates are for')
        mode = str(p['grasp_mode'])
        if mode not in GRASP_MODES:
            problems.append(f'grasp_mode {mode!r}: SurfaceGrasp proposes a contact on a measured surface for '
                            f'{", ".join(GRASP_MODES)} only')
        contact = num('contact_radius_m')
        if not contact > 0:
            problems.append('contact_radius_m must be > 0: the radius of the contact the site needs on the surface '
                            '(the suction cup\'s seal)')
        cone = num('approach_cone_deg')
        if not (0.0 < cone < 90.0):
            problems.append(f'approach_cone_deg must be in (0, 90) deg, got {p["approach_cone_deg"]!r}: the '
                            f'allowed approach cone has no default (an envelope is the cell\'s, never the '
                            f'framework\'s)')
        axis = _vec3(p['approach_cone_axis'], 'approach_cone_axis', problems)
        xref = _vec3(p['reference_x_axis'], 'reference_x_axis', problems)
        R_ref = None
        if axis is not None and xref is not None:
            R_ref = reference_rotation(axis, xref)
            if R_ref is None:
                problems.append('reference_x_axis is parallel to approach_cone_axis: the rotation about the approach '
                                'would be undefined (no axis is invented)')
        view = num('max_view_angle_deg')
        if not (0.0 < view <= 90.0):
            problems.append(f'max_view_angle_deg must be in (0, 90], got {p["max_view_angle_deg"]!r}')
        s, smin, smax = num('standoff_m'), num('standoff_min_m'), num('standoff_max_m')
        if not (smin > 0):
            problems.append(f'standoff_min_m must be > 0, got {p["standoff_min_m"]!r}')
        if not (smax >= smin):
            problems.append(f'standoff_max_m ({p["standoff_max_m"]!r}) must be >= standoff_min_m '
                            f'({p["standoff_min_m"]!r})')
        if not (smin <= s <= smax) or not s > 0:
            problems.append(f'standoff_m {p["standoff_m"]!r} must be in [standoff_min_m, standoff_max_m] = '
                            f'[{p["standoff_min_m"]}, {p["standoff_max_m"]}] m: the PreGrasp distance is never '
                            f'clamped silently')
        maxc, minpts, iters, seed = (integer('max_candidates'), integer('min_points'), integer('ransac_iters'),
                                     integer('seed'))
        if maxc < 1:
            problems.append('max_candidates must be >= 1')
        if minpts < 3:
            problems.append('min_points must be >= 3 (three points propose a plane)')
        if iters < 1:
            problems.append('ransac_iters must be >= 1')
        if seed < 0:
            problems.append('seed must be >= 0')
        inl, zmin, zmax = num('inlier_mm'), num('z_min_m'), num('z_max_m')
        if not inl > 0:
            problems.append('inlier_mm must be > 0')
        if not (0 <= zmin < zmax):
            problems.append(f'the depth band [z_min_m, z_max_m] = [{p["z_min_m"]}, {p["z_max_m"]}] is empty')
        mpq, mvf, mcl = num('min_pose_quality'), num('min_visible_fraction'), num('min_clearance')
        for k, v in (('min_pose_quality', mpq), ('min_visible_fraction', mvf), ('min_clearance', mcl)):
            if not (0.0 <= v <= 1.0):
                problems.append(f'{k} must be in [0, 1]')
        trad, treach = num('tool_radius_m'), num('tool_reach_mm')
        if trad < 0 or treach < 0:
            problems.append('tool_radius_m and tool_reach_mm must be >= 0 (0 = the clearance term is off)')
        if trad > 0 and not treach > 0:
            problems.append('tool_radius_m > 0 needs tool_reach_mm > 0: the clearance term needs the tool\'s grip '
                            'window (ADR-0017 Decision 4)')
        if mcl > 0 and not trad > 0:
            problems.append('min_clearance > 0 needs tool_radius_m > 0: a threshold on a term that is not '
                            'evaluated would never refuse anything')
        face = None
        try:
            face = make_face(str(p['grasp_face'] or 'disc'), p)
        except (KeyError, ValueError, TypeError) as exc:
            problems.append(f'grasp_face: {exc}')
        if mvf > 0 and face is not None and not face.has_known_area:
            problems.append('min_visible_fraction > 0 needs the face\'s size (e.g. expected_radius_m): visibility '
                            'is not evaluated without it')
        gates = None
        try:
            gates = PoseQualityGates.from_params(p)
        except (TypeError, ValueError) as exc:
            problems.append(f'pose_quality gates: {exc}')
        est = str(p['centre_estimator'])
        known = None
        if est not in CENTRE_ESTIMATORS:
            problems.append(f'centre_estimator {est!r}: one of {", ".join(CENTRE_ESTIMATORS)}')
        elif est == 'known_shape_fit':
            fk = {k: num(k) for k in ('face_occlusion_mm', 'face_min_coverage', 'face_max_outside',
                                      'face_max_ambiguity_mm', 'face_min_arc_deg', 'face_max_rms_mm',
                                      'face_radius_tol')}
            dil = integer('face_mask_dilate_px')
            if str(p['grasp_face'] or 'disc') != 'disc':
                problems.append(f'centre_estimator known_shape_fit fits a DISC face; grasp_face is '
                                f'{p["grasp_face"]!r} (no fit is implemented for it: never a silent centroid)')
            elif face is not None and not face.has_known_area:
                problems.append('centre_estimator known_shape_fit needs the face\'s size: expected_radius_m > 0 (the '
                                'disc whose centre is fitted)')
            elif face is not None:
                if not (0.0 < fk['face_min_arc_deg'] <= 360.0):
                    problems.append('face_min_arc_deg must be in (0, 360]')
                if not fk['face_occlusion_mm'] > inl:
                    problems.append(f'face_occlusion_mm ({fk["face_occlusion_mm"]:g}) must exceed inlier_mm ({inl:g}): '
                                    f'a neighbour inside the plane band is the face, not an occluder')
                if not fk['face_max_rms_mm'] > 0 or not fk['face_max_ambiguity_mm'] > 0:
                    problems.append('face_max_rms_mm and face_max_ambiguity_mm must be > 0')
                if not problems:
                    try:
                        known = KnownDiscParams(
                            radius_m=float(p['expected_radius_m']), band_m=inl * 1e-3,
                            occlusion_m=fk['face_occlusion_mm'] * 1e-3, dilate_px=dil,
                            min_coverage=fk['face_min_coverage'], max_outside=fk['face_max_outside'],
                            max_ambiguity_m=fk['face_max_ambiguity_mm'] * 1e-3, min_arc_deg=fk['face_min_arc_deg'],
                            max_rms_m=fk['face_max_rms_mm'] * 1e-3, radius_tol=fk['face_radius_tol'],
                            z_min_m=zmin, z_max_m=zmax)
                    except ValueError as exc:
                        problems.append(str(exc))
        if problems:
            raise ConfigError('SurfaceGrasp configuration refused: ' + '; '.join(problems))
        return cls(eef=eef, site=site, grasp_mode=mode, contact_radius_m=contact, cone_deg=cone,
                   cone_axis=tuple(float(x) for x in axis), reference=tuple(tuple(float(x) for x in r) for r in R_ref),
                   max_view_angle_deg=view, standoff_m=s, standoff_min_m=smin, standoff_max_m=smax,
                   max_candidates=maxc, min_points=minpts, inlier_mm=inl, ransac_iters=iters, seed=seed,
                   z_min_m=zmin, z_max_m=zmax, min_pose_quality=mpq, min_visible_fraction=mvf, tool_radius_m=trad,
                   tool_reach_mm=treach, min_clearance=mcl, face=face, gates=gates, params=p,
                   centre_estimator=est, known_disc=known)


# =================================================================================================================
# inputs / outputs
# =================================================================================================================

@dataclass(frozen=True)
class Region:
    """2-D evidence of one object: its pixels, its class, the segmenter's score. An internal type, not a ROS
    contract (M4_DESIGN.md §5). `box_px` = (u0, v0, u1, v1); all zero = unknown."""
    class_id: str
    score: float
    mask: np.ndarray = field(repr=False, compare=False)
    box_px: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class Candidate:
    id: str
    class_id: str
    point: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    approach: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]
    standoff_m: float
    eef: str
    site: str
    score: float
    confidence: float
    terms: Dict[str, float]
    not_evaluated: Tuple[str, ...]
    view_angle_deg: float
    approach_angle_deg: float
    region_pixel: Tuple[int, int]
    half_extents_m: Tuple[float, float]
    rank: int = -1
    centre_estimator: str = 'inlier_centroid'
    centre_by: str = ''                           # known_shape_fit: 'edges' | 'consensus'
    centre_shift_mm: float = float('nan')         # known_shape_fit: the fitted centre vs the visible centroid

    @property
    def pregrasp(self) -> Tuple[float, float, float]:
        p, a = np.asarray(self.point), np.asarray(self.approach)
        return tuple(float(x) for x in (p - self.standoff_m * a))

    def metadata(self) -> List[Tuple[str, str]]:
        """The KeyValue pairs of the wire candidate (never interpreted by the runtime, ADR-0026)."""
        from .. import __version__
        out = [('rank', str(self.rank))]
        out += [(f'term.{k}', _fmt(self.terms[k])) for k in TERMS if k in self.terms]
        out += [('not_evaluated', ','.join(self.not_evaluated)),
                ('normal_policy', NORMAL_POLICY),
                ('view_angle_deg', _fmt(self.view_angle_deg)),
                ('approach_angle_deg', _fmt(self.approach_angle_deg)),
                ('region', f'{self.region_pixel[0]},{self.region_pixel[1]}'),
                ('fit_method', 'plane_ransac'),
                ('centre_estimator', self.centre_estimator)]
        if self.centre_estimator == 'known_shape_fit':
            out += [('centre_by', self.centre_by), ('centre_shift_mm', _fmt(self.centre_shift_mm))]
        out += [('tp_version', __version__)]
        return out


@dataclass
class Rejection:
    id: str
    class_id: str
    code: str
    why: str
    terms: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraspResult:
    candidates: List[Candidate]                  # BEST FIRST, at most max_candidates
    rejections: List[Rejection]                  # in region order (then the BEYOND_MAX_CANDIDATES ones)
    n_regions: int
    timing_ms: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        codes: Dict[str, int] = {}
        for r in self.rejections:
            codes[r.code] = codes.get(r.code, 0) + 1
        return {'regions': self.n_regions, 'valid': len(self.candidates), 'rejected': len(self.rejections),
                'rejections_by_code': dict(sorted(codes.items()))}


def _fmt(v: float) -> str:
    return 'nan' if not math.isfinite(v) else f'{float(v):.9g}'


# =================================================================================================================
# geometry
# =================================================================================================================

def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two unit vectors [rad], exact near 0 and pi (atan2 of |a x b| and a . b)."""
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(a @ b))


def minimal_rotation(z: np.ndarray, a: np.ndarray) -> Optional[np.ndarray]:
    """The smallest rotation taking unit z onto unit a (3x3); None when they are antiparallel (axis undefined)."""
    z, a = z / np.linalg.norm(z), a / np.linalg.norm(a)
    k = np.cross(z, a)
    s, c = float(np.linalg.norm(k)), float(z @ a)
    if s < 1e-12:
        return np.eye(3) if c > 0 else None
    k /= s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = math.atan2(s, c)
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """A rotation matrix -> a unit quaternion (x, y, z, w) with w >= 0 (one deterministic sign)."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(x) for x in q)


def orient_toward_observer(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The canonical sign policy (debt G): the normal faces the camera along the VIEW RAY, n . (0 - p) > 0.
    A plane has no sign of its own; this rule picks it, by the view ray, never by n_z alone."""
    return -normal if float(normal @ (-point)) < 0 else normal


def grasp_orientation(approach: np.ndarray, cfg: SurfaceGraspConfig) -> Optional[np.ndarray]:
    """R = MinRot(c -> a) * R_ref (M4_DESIGN.md §7); None if undefined."""
    M = minimal_rotation(np.asarray(cfg.cone_axis), approach)
    if M is None:
        return None
    return M @ cfg.R_ref


def contact_unsupported(mask: np.ndarray, depth: np.ndarray, camera: CameraModel, contact: np.ndarray,
                        normal: np.ndarray, radius: float) -> int:
    """How many of 24 points of the contact disc (8 at radius/2, 16 at radius, in the plane through `contact` with
    `normal`) are NOT on the region's measured surface: projected into the image (with the lens), the pixel (or one
    of its 8 neighbours, for mask / depth holes) must be in `mask` with a valid depth whose back-projected point lies
    within `radius` of the plane. A point that projects off the image counts as unsupported."""
    a = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(normal, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    pts = [contact + rr * (math.cos(t) * a + math.sin(t) * b)
           for rr, k in ((0.5 * radius, 8), (radius, 16)) for t in np.linspace(0, 2 * np.pi, k, endpoint=False)]
    px = camera.project(np.asarray(pts))
    rays = camera.rays()
    H, W = mask.shape
    bad = 0
    for (u, v) in px:
        ok = False
        if np.isfinite(u) and np.isfinite(v):
            uc, vc = int(round(u)), int(round(v))
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    uu, vv = uc + du, vc + dv
                    if 0 <= uu < W and 0 <= vv < H and mask[vv, uu]:
                        z = float(depth[vv, uu])
                        if math.isfinite(z) and z > 0:
                            X = np.array([rays[vv, uu, 0] * z, rays[vv, uu, 1] * z, z])
                            if abs((X - contact) @ normal) <= radius:
                                ok = True
                                break
                if ok:
                    break
        bad += 0 if ok else 1
    return bad


def inscribed_half_extent(P: np.ndarray, centre: np.ndarray, normal: np.ndarray, band_m: float) -> float:
    """The half-extent of the measured planar patch along its NARROWER principal in-plane axis [m]: the points
    within `band_m` of the plane through `centre` with `normal`, their in-plane offsets from the centre, the two
    principal axes of those offsets, the p90 of |offset| along each, the smaller. NaN when fewer than 3 points.
    CONSERVATIVE, not a true inscribed radius: ~0.79 R on a full disc of radius R, 0.9 A on a band of half-width A
    (the support under the cup is checked separately, `contact_unsupported`)."""
    rel = P - centre
    h = rel @ normal
    Q = rel[np.abs(h) <= band_m] - np.outer(h[np.abs(h) <= band_m], normal)
    if len(Q) < 3:
        return float('nan')
    a = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(normal, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    uv = np.stack([Q @ a, Q @ b], 1)
    w, V = np.linalg.eigh(np.cov(uv.T))
    coords = uv @ V
    return float(min(np.percentile(np.abs(coords[:, 0]), 90), np.percentile(np.abs(coords[:, 1]), 90)))


def region_id(class_id: str, mask: np.ndarray) -> Tuple[str, Tuple[int, int]]:
    """A region's provider id `<class>@<u>,<v>` (its mask's rounded pixel centroid; `<class>@empty` without
    pixels) and that centroid. Shared by the provider and the eligibility filter, so both name a region alike."""
    v, u = np.nonzero(mask)
    if not len(u):
        return f'{class_id}@empty', (-1, -1)
    uc, vc = int(round(float(u.mean()))), int(round(float(v.mean())))
    return f'{class_id}@{uc},{vc}', (uc, vc)


def region_ids(regions: Sequence[Region]) -> List[str]:
    """The provider ids of `regions`, in their order: `region_id`, a collision suffixed `#k` in region order.
    Shared by `propose` and the presence objects (0.7.0, M4.5b), so a candidate, a rejection and an object of one
    region carry one id."""
    ids, seen = [], {}
    for r in regions:
        rid, _ = region_id(str(r.class_id), np.asarray(r.mask, bool))
        k = seen.get(rid, 0)
        seen[rid] = k + 1
        ids.append(rid if k == 0 else f'{rid}#{k}')
    return ids


# =================================================================================================================
# the provider
# =================================================================================================================

def propose(regions: Sequence[Region], depth_m: np.ndarray, camera: CameraModel,
            cfg: SurfaceGraspConfig) -> GraspResult:
    """Regions + registered depth + the camera model -> ranked candidates and explicit rejections."""
    t0 = time.perf_counter()
    depth = np.asarray(depth_m)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.shape != (camera.height, camera.width):
        raise ValueError(f'the depth image is {depth.shape[1]}x{depth.shape[0]}, the camera model '
                         f'{camera.width}x{camera.height}: registered depth lives on the colour grid')
    K = camera.matrix
    rays = camera.rays()
    axis = np.asarray(cfg.cone_axis)
    cone_rad = math.radians(cfg.cone_deg)

    # ids: stable for identical input; a collision gets '#k' in region order
    ids = region_ids(regions)
    pixels = [region_id(str(r.class_id), np.asarray(r.mask, bool))[1] for r in regions]

    # nonoverlap: a box can only be judged against the frame's other boxes
    boxes = [tuple(float(x) for x in r.box_px) for r in regions]
    box_ok = len(regions) >= 2 and all((b[2] - b[0]) > 0 and (b[3] - b[1]) > 0 for b in boxes)
    overlap = box_overlap(boxes) if box_ok else None

    valid: List[Candidate] = []
    rejections: List[Rejection] = []
    t_fit = 0.0
    for i, r in enumerate(regions):
        rid, cls = ids[i], str(r.class_id)

        def reject(code: str, why: str, terms: Optional[Dict[str, float]] = None) -> None:
            rejections.append(Rejection(id=rid, class_id=cls, code=code, why=why, terms=dict(terms or {})))

        mask = np.asarray(r.mask, bool)
        if mask.shape != depth.shape:
            raise ValueError(f'region {rid}: its mask is {mask.shape}, the depth {depth.shape}')
        n_px = int(mask.sum())
        if n_px == 0:
            reject(REGION_EMPTY, 'the region has no pixel')
            continue
        tf0 = time.perf_counter()
        v, u = np.nonzero(mask)
        z = depth[v, u].astype(np.float64)
        good = np.isfinite(z) & (z > cfg.z_min_m) & (z < cfg.z_max_m)
        coverage = float(good.mean())
        if int(good.sum()) < cfg.min_points:
            reject(INSUFFICIENT_DEPTH, f'{int(good.sum())} valid depth points in the region (need {cfg.min_points}; '
                                       f'coverage {coverage:.3f})', {'depth_coverage': coverage})
            t_fit += time.perf_counter() - tf0
            continue
        xy = rays[v[good], u[good]]
        P = np.stack([xy[:, 0] * z[good], xy[:, 1] * z[good], z[good]], 1)
        fit: GraspFit = plane_ransac_points(P, coverage, face=cfg.face, min_points=cfg.min_points,
                                            iters=cfg.ransac_iters, inlier_mm=cfg.inlier_mm, seed=cfg.seed)
        t_fit += time.perf_counter() - tf0
        if not fit.ok:
            reject(FIT_FAILED, fit.why or 'no plane fitted', dict(fit.quality))
            continue
        p = np.asarray(fit.point, float)
        n_raw = np.asarray(fit.normal, float)
        nn = float(np.linalg.norm(n_raw)) if np.isfinite(n_raw).all() else float('nan')
        if not (np.isfinite(p).all() and math.isfinite(nn) and nn > 1e-9 and p[2] > 0):
            reject(GEOMETRY_INVALID, 'the fitted point or normal is not finite, or the normal has zero length',
                   dict(fit.quality))
            continue
        n = orient_toward_observer(p, n_raw / nn)
        centre_by, centre_shift = '', float('nan')
        if cfg.known_disc is not None:
            # WHERE on the face (0.8.0, M4.5d): the centre of the KNOWN disc from the partial evidence, never the
            # visible centroid; a face whose centre the evidence does not determine is refused (no fallback)
            fc = known_disc_centre(mask, depth, rays, p, n, cfg.known_disc)
            if not fc.ok:
                reject(FACE_FIT_FAILED, fc.why, {**{k: float(x) for k, x in fit.quality.items()}, **fc.quality})
                continue
            centre_by, centre_shift = fc.by, float(fc.quality.get('face_centre_shift_mm', float('nan')))
            p = np.asarray(fc.centre, float)
        # the CONTACT is on the MEASURED surface, not on the fitted plane's chord: the region's depth points within
        # the site's contact radius of the approach axis, their median height along the normal. On a plane it is
        # the inlier centroid (offset 0); on a curved face (a dome, a sphere) the plane through the inlier band lies
        # inside the surface -- by 2.4 mm on a 50 mm sphere, measured on the D0.7 Replicator sessions -- and the
        # tool would be sent through it. BOUNDED: only points within one contact radius of the plane count (a recess,
        # a bottle's mouth or background seen through the mask is not the surface the cup touches -- review M4,
        # a 150 mm snap into a bottle); whether the contact is SUPPORTED at all is checked below.
        rel = P - p
        h = rel @ n
        band = cfg.contact_radius_m
        near = ((np.einsum('ij,ij->i', rel, rel) - h * h) <= cfg.contact_radius_m ** 2) & (np.abs(h) <= band)
        snap = float(np.median(h[near])) if int(near.sum()) >= 3 else 0.0
        p = p + snap * n
        view_dir = -p / np.linalg.norm(p)
        view_angle = math.degrees(_angle(n, view_dir))
        a = -n
        app_angle = _angle(a, axis)
        terms: Dict[str, float] = {}
        quality = {k: float(x) for k, x in fit.quality.items()}
        quality['contact_offset_mm'] = snap * 1e3
        if view_angle >= cfg.max_view_angle_deg:
            reject(NORMAL_GRAZING, f'the surface is seen {view_angle:.2f} deg from square-on (max_view_angle_deg '
                                   f'{cfg.max_view_angle_deg:g}): its normal is not measured', quality)
            continue
        if app_angle > cone_rad + _CONE_EPS_RAD:
            reject(APPROACH_OUTSIDE_CONE, f'the approach is {math.degrees(app_angle):.4f} deg from the cone axis '
                                          f'(approach_cone_deg {cfg.cone_deg:g}): outside the permitted envelope, '
                                          f'never clamped', quality)
            continue
        pq = pose_quality(fit, gates=cfg.gates, extent_keys=cfg.face.extent_term_keys)
        terms['fit'] = pq
        if pq <= 0.0:
            reject(POSE_QUALITY_REFUSED, 'pose_quality is exactly 0: one of its gates clipped flat (coverage, '
                                         'inliers, rms, extent) -- the surface is not the object', {**quality, **terms})
            continue
        if pq < cfg.min_pose_quality:
            reject(POSE_QUALITY_LOW, f'pose_quality {pq:.4f} < min_pose_quality {cfg.min_pose_quality:g}',
                   {**quality, **terms})
            continue
        # THE SEAL: the measured patch's own two in-plane extents (its principal axes, p90 of the inliers' offsets),
        # never the face's -- a DiscFace collapses both axes into one radius, so a 7 mm band seen edge-on (the side
        # of a cap) read as a 20 mm disc and a 9 mm cup was offered a surface it cannot seal on (found by M4-B)
        r_in = inscribed_half_extent(P, np.asarray(fit.point, float), n, cfg.inlier_mm * 1e-3)
        quality['inscribed_mm'] = r_in * 1e3 if math.isfinite(r_in) else float('nan')
        if not (math.isfinite(r_in) and r_in >= cfg.contact_radius_m):
            reject(SURFACE_TOO_SMALL, f'the measured patch\'s inscribed half-extent {r_in * 1e3:.2f} mm (its narrower '
                                      f'principal axis) is below the site\'s contact radius '
                                      f'{cfg.contact_radius_m * 1e3:.2f} mm: {cfg.eef}/{cfg.site} cannot seal on it',
                   {**quality, **terms})
            continue
        terms['seal'] = (r_in - cfg.contact_radius_m) / r_in if r_in > 0 else 0.0
        not_eval = list(ALWAYS_NOT_EVALUATED)
        if cfg.face.has_known_area:
            exp = cfg.face.expected_area_px(float(p[2]), K)
            terms['visibility'] = visible_fraction(float(n_px), exp)
            if terms['visibility'] < cfg.min_visible_fraction:
                reject(VISIBILITY_LOW, f'visible_fraction {terms["visibility"]:.4f} < min_visible_fraction '
                                       f'{cfg.min_visible_fraction:g}', {**quality, **terms})
                continue
        else:
            not_eval.append('visibility')
        if cfg.tool_radius_m > 0:
            obs = approach_obstruction(GraspFit(tuple(p), tuple(n), fit.half_extents_m, fit.quality, True, ''), depth,
                                       K, tool_radius_m=cfg.tool_radius_m, inlier_mm=cfg.inlier_mm,
                                       z_min=cfg.z_min_m, z_max=cfg.z_max_m, rays=rays)
            if not math.isfinite(obs):
                reject(CLEARANCE_UNKNOWN, 'the depth cannot tell whether the approach is clear (obstruction NaN): a '
                                          'clearance that could not be measured is not a pass', {**quality, **terms})
                continue
            terms['clearance'] = approach_clearance(obs, reach_mm=cfg.tool_reach_mm)
            quality['obstruction_mm'] = obs
            if terms['clearance'] < cfg.min_clearance:
                reject(CLEARANCE_LOW, f'clearance {terms["clearance"]:.4f} < min_clearance {cfg.min_clearance:g} '
                                      f'(obstruction {obs:.2f} mm)', {**quality, **terms})
                continue
        else:
            not_eval.append('clearance')
        # SUPPORT under the cup (review M4): the contact disc (radius contact_radius_m about the contact, in the plane)
        # must lie on THIS region's measured surface -- a ring around a hole, an L-shaped patch, a contact on an
        # occlusion edge all have a large patch and nothing under part of the cup. 24 points of the disc (8 at r/2,
        # 16 at r) projected into the image: each must fall in the region with depth on the plane (within one
        # contact radius); resolution-independent; at most 2 may fail. Checked after the clearance, so something
        # standing ON the contact reads as an obstruction when the clearance term is evaluated.
        unsupported = contact_unsupported(mask, depth, camera, p, n, cfg.contact_radius_m)
        quality['contact_unsupported'] = float(unsupported)
        if unsupported > 2:
            reject(CONTACT_UNSUPPORTED, f'{unsupported} of 24 points of the contact disc (radius '
                                        f'{cfg.contact_radius_m * 1e3:.2f} mm) are not on the measured surface: '
                                        f'{cfg.eef}/{cfg.site} would not be supported there', {**quality, **terms})
            continue
        if overlap is not None:
            terms['nonoverlap'] = float(1.0 - overlap[i])
        else:
            not_eval.append('nonoverlap')
        R = grasp_orientation(a, cfg)
        if R is None:
            reject(ORIENTATION_UNDEFINED, 'the minimal rotation from the cone axis onto the approach is undefined',
                   {**quality, **terms})
            continue
        terms['detection'] = float(np.clip(float(r.score), 0.0, 1.0)) if math.isfinite(float(r.score)) else 0.0
        terms['alignment'] = float(max(0.0, 1.0 - app_angle / cone_rad))
        ordered = {k: float(terms[k]) for k in TERMS if k in terms}
        score = float(np.prod(list(ordered.values())))
        valid.append(Candidate(
            id=rid, class_id=cls, point=tuple(float(x) for x in p), normal=tuple(float(x) for x in n),
            approach=tuple(float(x) for x in a), orientation_xyzw=quat_xyzw(R), standoff_m=cfg.standoff_m,
            eef=cfg.eef, site=cfg.site, score=score, confidence=float(pq), terms=ordered,
            not_evaluated=tuple(sorted(not_eval)), view_angle_deg=view_angle,
            approach_angle_deg=math.degrees(app_angle), region_pixel=pixels[i],
            half_extents_m=tuple(float(x) for x in fit.half_extents_m),
            centre_estimator=cfg.centre_estimator, centre_by=centre_by, centre_shift_mm=centre_shift))

    t_rank0 = time.perf_counter()
    valid.sort(key=lambda c: (-c.score, c.id))          # score descending, ties by id: total and deterministic
    for k, c in enumerate(valid):
        c.rank = k
    kept, beyond = valid[:cfg.max_candidates], valid[cfg.max_candidates:]
    for c in beyond:
        rejections.append(Rejection(id=c.id, class_id=c.class_id, code=BEYOND_MAX_CANDIDATES,
                                    why=f'valid, ranked {c.rank} >= max_candidates {cfg.max_candidates}',
                                    terms=dict(c.terms)))
    t_end = time.perf_counter()
    return GraspResult(candidates=kept, rejections=rejections, n_regions=len(regions),
                       timing_ms={'total': (t_end - t0) * 1e3, 'fit': t_fit * 1e3, 'rank': (t_end - t_rank0) * 1e3})


def regions_from_detections(dets) -> List[Region]:
    """`trainit_perception.detectors.Detection`s that carry a mask -> regions, in the detector's own order.
    A detection without a mask is refused: a surface grasp needs the object's pixels, not its centroid."""
    out = []
    for i, d in enumerate(dets):
        m = getattr(d, 'mask', None)
        if m is None:
            raise ValueError(f'detection {i} ({d.class_id}) carries no mask: the segmentation method must expose '
                             f'per-detection masks for a surface grasp')
        out.append(Region(class_id=str(d.class_id), score=float(d.score), mask=np.asarray(m, bool),
                          box_px=tuple(float(x) for x in d.box_px)))
    return out


def diagnostics(result: GraspResult, frame_id: str, stamp_sec: int, stamp_nanosec: int,
                camera: Optional[CameraModel] = None, *, ineligible: Sequence[Mapping[str, object]] = (),
                segmentation: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """The provider's inspection record of one frame (published as JSON on `grasp_diagnostics`, paired with the
    candidate array by stamp). Non-finite values become null (strict JSON).

    `ineligible` (0.7.0): the regions the workspace eligibility filter kept OUT of `propose` (never geometry
    rejections: `grasp.eligibility`), counted in `summary.ineligible`; `summary.regions` counts the regions that
    entered `propose`. `segmentation` (0.7.0): the segmentation method's own report of this frame, when it has one."""
    def clean(d: Mapping[str, float]) -> Dict[str, Optional[float]]:
        return {k: (float(v) if math.isfinite(float(v)) else None) for k, v in d.items()}

    def finite(v):
        return float(v) if v is not None and math.isfinite(float(v)) else None

    summary = result.summary()
    summary['ineligible'] = len(ineligible)
    out = {
        'provider': PROVIDER_ID,
        'stamp': {'sec': int(stamp_sec), 'nanosec': int(stamp_nanosec)},
        'frame_id': frame_id,
        'summary': summary,
        'camera': None if camera is None else {'distortion_model': camera.distortion_model,
                                               'distorted': camera.distorted,
                                               'size': [camera.width, camera.height]},
        'candidates': [{'rank': c.rank, 'id': c.id, 'class_id': c.class_id, 'score': c.score,
                        'confidence': c.confidence, 'terms': clean(c.terms), 'not_evaluated': list(c.not_evaluated),
                        'point': list(c.point), 'normal': list(c.normal), 'approach': list(c.approach),
                        'view_angle_deg': c.view_angle_deg, 'approach_angle_deg': c.approach_angle_deg,
                        'centre_estimator': c.centre_estimator, 'centre_by': c.centre_by,
                        'centre_shift_mm': finite(c.centre_shift_mm)}
                       for c in result.candidates],
        'rejected': [{'id': r.id, 'class_id': r.class_id, 'code': r.code, 'why': r.why, 'terms': clean(r.terms)}
                     for r in result.rejections],
        'ineligible': [{'id': str(e['id']), 'class_id': str(e['class_id']), 'code': str(e['code']),
                        'why': str(e['why']), 'inside_fraction': finite(e.get('inside_fraction'))}
                       for e in ineligible],
        'timing_ms': clean(result.timing_ms),
    }
    if segmentation is not None:
        out['segmentation'] = dict(segmentation)
    return out
