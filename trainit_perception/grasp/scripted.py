"""The SCRIPTED grasp provider (`tp.scripted`, vNext M4.5): candidates from a scenario file, for mock product tests.

ROS-free and pure (the node is `nodes/scripted_grasp_node.py`). In a mock cell the camera is a flat plane, so no
measured provider can propose anything meaningful (KNOWN_DEBT); the scripted provider stands in for one behind the
SAME runtime surface -- the capture Trigger, a `GraspCandidateArray` on `/perception/<name>/grasp_candidates`, a
diagnostics JSON -- so an application runs unchanged between mock and Isaac (M4_5_DESIGN.md §12). It measures
nothing and claims nothing: every value comes from the scenario.

perception.yaml:

    detectors:
      part_grasp:
        method: scripted_grasp
        params: {scenario_file: /abs/path/scenario.yaml, frame_id: camera_color_optical_frame,
                 after_end: empty}            # or repeat_last
        continuous: false                     # optional; a scripted provider is trigger-only

scenario.yaml -- ONE step per Trigger call, in order:

    steps:
      - candidates:
          - {position: [0.0, 0.0, 0.5], approach: [0, 0, 1], standoff_m: 0.05, eef: main, site: suction,
             score: 0.9, confidence: 0.8, object_class: capsule}      # orientation computed (+Z = approach)
          - {position: [...], approach: [...], orientation: [qx, qy, qz, qw], standoff_m: 0.05}
        stamp_offset_ms: 0        # the array is stamped now + offset on the node clock; negative = STALE
        delay_ms: 0               # published this long after the Trigger
      - {candidates: []}          # an empty answer: nothing to pick (and, by default, nothing present)
      - {candidates: [], objects: [[0.02, 0.01, 0.5]]}   # no candidate, but an object PRESENT (M4.5b)
      - {unavailable: true}       # the Trigger answers success=false and nothing is published

A candidate without `orientation` gets the SurfaceGrasp rule: the reference frame (+Z = `approach_cone_axis`, +X =
`reference_x_axis` projected) tilted minimally onto the approach, so its +Z IS the approach. `approach` and
`orientation` are normalised (the wire contract wants unit vectors). After the last step, `after_end: empty`
answers every further Trigger with an empty array, `repeat_last` repeats the last step.

PRESENCE (M4.5b): each published step also publishes its OBJECTS PRESENT (`/perception/<name>/objects`, the same
header as its candidate array): `objects: [[x, y, z], ...]` (positions in `frame_id`), or an int `n` (one object at
each of the step's first n candidates' positions; n above the step's candidates is refused), or -- without the key --
one object per candidate at its position. `objects: []` = nothing present. Past the end, `empty` presents nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from .surface_grasp import ConfigError, _vec3, minimal_rotation, quat_xyzw, reference_rotation

PROVIDER_ID = 'tp.scripted'
METHOD = 'scripted_grasp'
AFTER_END = ('empty', 'repeat_last')
#: a scripted candidate is for whatever end effector the scenario names: it proposes no contact model
GRASP_MODES = ('any',)

#: the provider's own parameters (exported by `vocabulary.export()` under `grasp_providers.scripted_grasp`)
PARAMETERS: Dict[str, Dict[str, object]] = {
    'scenario_file': {'type': 'str', 'required': True, 'since': '0.7.0'},
    'frame_id': {'type': 'str', 'required': True, 'since': '0.7.0'},
    'after_end': {'type': 'str', 'choices': list(AFTER_END), 'since': '0.7.0'},
    'approach_cone_axis': {'type': 'vec3', 'since': '0.7.0'},
    'reference_x_axis': {'type': 'vec3', 'since': '0.7.0'},
}

STEP_KEYS = ('candidates', 'objects', 'unavailable', 'stamp_offset_ms', 'delay_ms')
CANDIDATE_KEYS = ('id', 'position', 'approach', 'orientation', 'standoff_m', 'eef', 'site', 'score', 'confidence',
                  'object_class')


def defaults() -> Dict[str, object]:
    return {'scenario_file': '', 'frame_id': '', 'after_end': 'empty',
            'approach_cone_axis': [0.0, 0.0, 1.0], 'reference_x_axis': [1.0, 0.0, 0.0]}


@dataclass(frozen=True)
class ScriptedConfig:
    scenario_file: str
    frame_id: str
    after_end: str
    cone_axis: Tuple[float, float, float]
    R_ref: np.ndarray = field(compare=False, repr=False)

    @classmethod
    def from_params(cls, params: Mapping) -> 'ScriptedConfig':
        params = params or {}
        problems: List[str] = []
        unknown = sorted(set(params) - set(PARAMETERS))
        if unknown:
            problems.append(f'unknown parameter(s) {", ".join(unknown)} (have: {", ".join(PARAMETERS)})')
        p = dict(defaults())
        p.update(params)
        for key in ('scenario_file', 'frame_id'):
            if not isinstance(p[key], str) or not p[key]:
                problems.append(f'{key} is required (a non-empty string), got {p[key]!r}')
        if p['after_end'] not in AFTER_END:
            problems.append(f'after_end must be one of {", ".join(AFTER_END)}, got {p["after_end"]!r}')
        axis = _vec3(p['approach_cone_axis'], 'approach_cone_axis', problems)
        xref = _vec3(p['reference_x_axis'], 'reference_x_axis', problems)
        R_ref = None
        if axis is not None and xref is not None:
            R_ref = reference_rotation(axis, xref)
            if R_ref is None:
                problems.append('reference_x_axis is parallel to approach_cone_axis: the rotation about the '
                                'approach would be undefined')
        if problems:
            raise ConfigError('scripted grasp configuration refused: ' + '; '.join(problems))
        return cls(scenario_file=p['scenario_file'], frame_id=p['frame_id'], after_end=p['after_end'],
                   cone_axis=tuple(float(x) for x in axis), R_ref=R_ref)


@dataclass(frozen=True)
class ScriptedCandidate:
    id: str
    position: Tuple[float, float, float]
    approach: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]
    standoff_m: float
    eef: str
    site: str
    score: float
    confidence: float
    object_class: str


@dataclass(frozen=True)
class Step:
    candidates: Tuple[ScriptedCandidate, ...]
    unavailable: bool = False
    stamp_offset_ms: float = 0.0
    delay_ms: float = 0.0
    #: the objects PRESENT (M4.5b), positions in `frame_id`; None = one per candidate, at its position
    objects: Optional[Tuple[Tuple[float, float, float], ...]] = None

    def present(self) -> Tuple[Tuple[float, float, float], ...]:
        """The positions of the objects this step presents."""
        if self.objects is not None:
            return self.objects
        return tuple(c.position for c in self.candidates)


def _finite(v) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _vector(v, n: int) -> Optional[np.ndarray]:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    vals = [_finite(x) for x in v]
    if any(x is None for x in vals):
        return None
    return np.asarray(vals, float)


def _candidate(raw, where: str, cfg: ScriptedConfig, problems: List[str]) -> Optional[ScriptedCandidate]:
    if not isinstance(raw, Mapping):
        problems.append(f'{where}: a candidate is a mapping, got {raw!r}')
        return None
    stray = sorted(set(raw) - set(CANDIDATE_KEYS))
    if stray:
        problems.append(f'{where}: unknown key(s) {", ".join(map(str, stray))} (have: {", ".join(CANDIDATE_KEYS)})')
    n0 = len(problems)
    pos = _vector(raw.get('position'), 3)
    if pos is None:
        problems.append(f'{where}: position must be three finite numbers [x, y, z], got {raw.get("position")!r}')
    app = _vector(raw.get('approach'), 3)
    if app is None or np.linalg.norm(app) < 1e-9:
        problems.append(f'{where}: approach must be a finite, non-zero 3-vector, got {raw.get("approach")!r}')
        app = None
    else:
        app = app / np.linalg.norm(app)
    quat = None
    if raw.get('orientation') is not None:
        quat = _vector(raw['orientation'], 4)
        if quat is None or np.linalg.norm(quat) < 1e-9:
            problems.append(f'{where}: orientation must be a finite, non-zero quaternion [qx, qy, qz, qw], got '
                            f'{raw["orientation"]!r}')
            quat = None
        else:
            quat = quat / np.linalg.norm(quat)
    elif app is not None:
        M = minimal_rotation(np.asarray(cfg.cone_axis), app)
        if M is None:
            problems.append(f'{where}: the approach is antiparallel to approach_cone_axis, so no orientation can be '
                            f'computed: give `orientation` explicitly')
        else:
            quat = np.asarray(quat_xyzw(M @ cfg.R_ref))
    standoff = _finite(raw.get('standoff_m'))
    if standoff is None or standoff < 0:
        problems.append(f'{where}: standoff_m must be a finite number >= 0, got {raw.get("standoff_m")!r}')
    nums = {}
    for key in ('score', 'confidence'):
        v = _finite(raw.get(key, 1.0))
        if v is None or not 0.0 <= v <= 1.0:
            problems.append(f'{where}: {key} must be in [0, 1], got {raw.get(key)!r}')
        nums[key] = v
    texts = {}
    for key in ('id', 'eef', 'site', 'object_class'):
        v = raw.get(key, '')
        if not isinstance(v, str):
            problems.append(f'{where}: {key} must be a string, got {v!r}')
        texts[key] = v if isinstance(v, str) else ''
    if len(problems) != n0:
        return None
    return ScriptedCandidate(id=texts['id'], position=tuple(float(x) for x in pos),
                             approach=tuple(float(x) for x in app),
                             orientation_xyzw=tuple(float(x) for x in quat), standoff_m=float(standoff),
                             eef=texts['eef'], site=texts['site'], score=float(nums['score']),
                             confidence=float(nums['confidence']), object_class=texts['object_class'])


def parse_scenario(doc, cfg: ScriptedConfig, where: str = 'scenario') -> List[Step]:
    """A scenario document (already YAML-loaded) -> its steps. Raises `ConfigError` naming EVERY problem: a
    scenario is validated in full at start-up, never discovered broken at its seventh Trigger."""
    if not isinstance(doc, Mapping) or set(doc) != {'steps'}:
        raise ConfigError(f'{where}: a scenario is a mapping with exactly one key, `steps` (a list), got '
                          f'{sorted(doc) if isinstance(doc, Mapping) else type(doc).__name__}')
    raw_steps = doc['steps']
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ConfigError(f'{where}: steps must be a non-empty list')
    problems: List[str] = []
    steps: List[Step] = []
    for i, raw in enumerate(raw_steps):
        at = f'{where}: steps[{i}]'
        if not isinstance(raw, Mapping):
            problems.append(f'{at}: a step is a mapping, got {raw!r}')
            continue
        stray = sorted(set(raw) - set(STEP_KEYS))
        if stray:
            problems.append(f'{at}: unknown key(s) {", ".join(map(str, stray))} (have: {", ".join(STEP_KEYS)})')
        unavailable = raw.get('unavailable', False)
        if not isinstance(unavailable, bool):
            problems.append(f'{at}: unavailable must be true or false, got {unavailable!r}')
            unavailable = False
        has_cands = 'candidates' in raw
        if unavailable == has_cands:
            problems.append(f'{at}: a step is EITHER `candidates: [...]` (possibly empty) OR `unavailable: true`')
        offset, delay = _finite(raw.get('stamp_offset_ms', 0)), _finite(raw.get('delay_ms', 0))
        if offset is None:
            problems.append(f'{at}: stamp_offset_ms must be a finite number, got {raw.get("stamp_offset_ms")!r}')
        if delay is None or delay < 0:
            problems.append(f'{at}: delay_ms must be a finite number >= 0, got {raw.get("delay_ms")!r}')
        cands: List[ScriptedCandidate] = []
        raw_c = raw.get('candidates') or []
        if not isinstance(raw_c, list):
            problems.append(f'{at}: candidates must be a list')
            raw_c = []
        for j, rc in enumerate(raw_c):
            c = _candidate(rc, f'{at}.candidates[{j}]', cfg, problems)
            if c is not None:
                cands.append(c if c.id else replace(c, id=f'scripted@{i}.{j}'))
        ids = [c.id for c in cands]
        dup = sorted({x for x in ids if ids.count(x) > 1})
        if dup:
            problems.append(f'{at}: candidate id(s) {dup} repeat: ids are unique within an array')
        objects = _objects(raw, at, cands, bool(unavailable), problems)
        steps.append(Step(candidates=tuple(cands), unavailable=bool(unavailable), stamp_offset_ms=offset or 0.0,
                          delay_ms=delay or 0.0, objects=objects))
    if problems:
        raise ConfigError('scripted scenario refused: ' + '; '.join(problems))
    return steps


def _objects(raw: Mapping, at: str, cands: Sequence[ScriptedCandidate], unavailable: bool,
             problems: List[str]) -> Optional[Tuple[Tuple[float, float, float], ...]]:
    """A step's `objects`: None (absent: one per candidate), positions, or an int count of candidate positions."""
    if 'objects' not in raw:
        return None
    value = raw['objects']
    if unavailable:
        problems.append(f'{at}: an unavailable step publishes nothing: it has no objects')
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value <= len(cands):
            problems.append(f'{at}: objects: {value} places one object at each of the step\'s first {value} '
                            f'candidates, and it has {len(cands)}: give positions [[x, y, z], ...] instead')
            return None
        return tuple(c.position for c in cands[:value])
    if not isinstance(value, list):
        problems.append(f'{at}: objects must be a list of positions [[x, y, z], ...] or a count, got {value!r}')
        return None
    out = []
    for k, p in enumerate(value):
        v = _vector(p, 3)
        if v is None:
            problems.append(f'{at}.objects[{k}]: a position is three finite numbers [x, y, z], got {p!r}')
            continue
        out.append(tuple(float(x) for x in v))
    return tuple(out)


def load_scenario(path: str, cfg: ScriptedConfig) -> List[Step]:
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f'scripted scenario refused: cannot read {path!r} ({exc.strerror or exc})') from None
    except yaml.YAMLError as exc:
        raise ConfigError(f'scripted scenario refused: {path!r} is not YAML ({str(exc).splitlines()[0]})') from None
    return parse_scenario(doc, cfg, where=path)


class Scenario:
    """The steps and the cursor: `next()` returns (index, step) and advances one step per call. Past the end:
    `empty` -> (index, an empty step), `repeat_last` -> the last step again."""

    def __init__(self, steps: Sequence[Step], after_end: str = 'empty'):
        if not steps:
            raise ConfigError('a scenario needs at least one step')
        if after_end not in AFTER_END:
            raise ConfigError(f'after_end must be one of {", ".join(AFTER_END)}')
        self.steps = list(steps)
        self.after_end = after_end
        self.calls = 0

    def next(self) -> Tuple[int, Step]:
        i = self.calls
        self.calls += 1
        if i < len(self.steps):
            return i, self.steps[i]
        if self.after_end == 'repeat_last':
            return i, self.steps[-1]
        return i, Step(candidates=(), objects=())


def object_id(k: int) -> str:
    """The id of a scripted step's k-th object present."""
    return f'scripted_object@{k}'


def diagnostics(step_index: int, step: Step, n_steps: int, frame_id: str, stamp_sec: int,
                stamp_nanosec: int) -> Dict[str, object]:
    """The minimal diagnostics record, in SurfaceGrasp's shape (provider, stamp, frame, summary, candidates,
    rejected, ineligible) plus the scenario position."""
    return {
        'provider': PROVIDER_ID,
        'stamp': {'sec': int(stamp_sec), 'nanosec': int(stamp_nanosec)},
        'frame_id': frame_id,
        'summary': {'regions': 0, 'valid': len(step.candidates), 'rejected': 0, 'rejections_by_code': {},
                    'ineligible': 0, 'objects': len(step.present()), 'objects_without_depth': 0},
        'camera': None,
        'candidates': [{'rank': k, 'id': c.id, 'class_id': c.object_class, 'score': c.score,
                        'confidence': c.confidence, 'point': list(c.position), 'approach': list(c.approach)}
                       for k, c in enumerate(step.candidates)],
        'rejected': [],
        'ineligible': [],
        'objects': {'published': [{'id': object_id(k), 'position': list(p)} for k, p in enumerate(step.present())],
                    'without_depth': [], 'skipped': []},
        'scenario': {'step': int(step_index), 'steps': int(n_steps), 'past_end': step_index >= n_steps},
    }
