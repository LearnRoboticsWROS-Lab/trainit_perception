"""trainit_perception's CONFIGURATION VOCABULARY, as one record (vNext M2.2, ADR-0021).

This package owns what a configurator may say about perception: the detector methods and their
kind, what each method's detections carry, the colour mask's own parameters, the candidate rules
and the terms they multiply, what makes a term computable, the grasp faces, the version floors.
They live in the package's BUILT-IN tables; this module copies none of them, and :func:`export`
reads them into one plain, deterministic record (schema :data:`SCHEMA_VERSION`). The Setup
Assistant commits a snapshot of that record, written by its explicit catalog refresh, and NEVER
imports this package while it generates: the same project gives the same bundle whether or not a
perception package is installed.

What crosses is a closed allow-list (ADR-0021, Decision 9): names, kinds, parameter schema,
requirements, version floors. Never a rule note, a measured statement, a formula, a complement, a
gate value, an extent term or a weight. Strings are identifiers or versions; numbers sit only in a
parameter's schema and the face size band. Sets are written sorted and everything else in the
order its table declares it, so the record -- and the snapshot made from it -- is byte-stable.

ROS-free. Importing this module imports nothing else of the package: the methods are read here and
the ranking half is asked of ``ranking.vocabulary``, both inside :func:`export`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Dict, List

SCHEMA_VERSION = 1

_TOKEN = re.compile(r'^[a-z][a-z0-9_]*$')
_VERSION = re.compile(r'^\d+\.\d+\.\d+$')


def export() -> Dict[str, object]:
    """The configuration vocabulary of THIS release, from the built-in tables.

    Raises ``ValueError`` naming every inconsistency it finds instead of returning a record a
    configurator would then have to distrust.
    """
    from . import __version__
    from .detectors.registry import BUILTIN_METHODS
    from .ranking.vocabulary import export_ranking

    ranking = export_ranking()
    methods: Dict[str, object] = {}
    method_floors: Dict[str, str] = {}
    gaps: List[str] = []
    for name, kind, cls in BUILTIN_METHODS:
        if cls is None:
            # An external node publishes the contract from its own process: what its detections
            # carry and which parameters it reads are its own business, so they are unknown here.
            methods[name] = {'kind': kind, 'provides': None, 'parameters': None}
            continue
        try:
            defaults = cls.defaults()
        except Exception as exc:                                     # noqa: BLE001
            gaps.append(f'method {name!r}: defaults() fails ({type(exc).__name__}: {exc})')
            defaults = {}
        parameters = {}
        declared = getattr(cls, 'PARAMETERS', None)
        if not isinstance(declared, Mapping):
            gaps.append(f'method {name!r} declares no PARAMETERS schema (a mapping) for its own '
                        f'parameters')
            declared = {}
        for key, schema in declared.items():
            if key not in defaults or not isinstance(schema, Mapping) or 'type' not in schema:
                gaps.append(f'method {name!r} parameter {key!r} has no default, or no schema '
                            f'with a type')
                continue
            row = {'type': schema['type'], 'default': defaults[key]}
            row.update({k: schema[k] for k in ('min', 'max', 'choices') if k in schema})
            parameters[key] = row
            if 'since' in schema:
                method_floors[key] = schema['since']
        provides = getattr(cls, 'PROVIDES', None)
        if not (isinstance(provides, (set, frozenset, list, tuple))
                and all(isinstance(t, str) for t in provides)):
            gaps.append(f'method {name!r} declares no PROVIDES (a set of output names)')
            provides = ()
        methods[name] = {'kind': kind, 'provides': sorted(provides), 'parameters': parameters}

    floors = dict(ranking['param_since'])
    clashes = sorted(k for k in method_floors if k in floors and floors[k] != method_floors[k])
    floors.update(method_floors)
    record = {
        'schema_version': SCHEMA_VERSION,
        'trainit_perception': __version__,
        'provides': ranking['provides'],
        'detector_methods': methods,
        'auto': ranking['auto'],
        'rules': ranking['rules'],
        'default_preference': ranking['default_preference'],
        'terms': ranking['terms'],
        'faces': ranking['faces'],
        'default_face': ranking['default_face'],
        'face_size_band_m': ranking['face_size_band_m'],
        'param_since': dict(sorted(floors.items())),
    }
    problems = gaps + [f'parameter {k!r} is dated {method_floors[k]} by its method and '
                       f'{ranking["param_since"][k]} by the ranking' for k in clashes]
    problems += consistency_problems(record)
    if problems:
        raise ValueError('trainit_perception\'s configuration vocabulary is inconsistent: '
                         + '; '.join(problems))
    return record


def consistency_problems(record: Dict[str, object]) -> List[str]:
    """Every way the record contradicts itself, as sentences. Empty = consistent."""
    out: List[str] = []
    provides = set(record['provides'])
    terms = record['terms']
    faces = record['faces']
    floors = record['param_since']
    for name, method in record['detector_methods'].items():
        for token in method['provides'] or ():
            if token not in provides:
                out.append(f'method {name!r} provides {token!r}, which is not a declared output')
    for name, factors in record['rules'].items():
        if name == record['auto']:
            out.append(f'rule {name!r} has the name of the token that means "resolve it"')
        for term in factors:
            if term not in terms:
                out.append(f'rule {name!r} multiplies {term!r}, which is not a declared term')
    for name in record['default_preference']:
        if name not in record['rules']:
            out.append(f'the default preference names {name!r}, which is not a rule')
    for name, req in terms.items():
        for token in req['provides']:
            if token not in provides:
                out.append(f'term {name!r} reads {token!r}, which is not a declared output')
        for param in req['positive_params']:
            if param not in floors:
                out.append(f'term {name!r} needs {param!r}, which no release is dated with')
    for name, face in faces.items():
        if not face['params']:
            out.append(f'face {name!r} is sized by no parameter')
        for param in face['params']:
            if param not in floors:
                out.append(f'face {name!r} reads {param!r}, which no release is dated with')
        for term in face['enables_terms']:
            if not terms.get(term, {}).get('face_sized'):
                out.append(f'face {name!r} enables {term!r}, which is not a face-sized term')
    if record['default_face'] not in faces:
        out.append(f'the default face {record["default_face"]!r} is not a registered face')
    band = record['face_size_band_m']
    if not (isinstance(band, list) and len(band) == 2
            and all(isinstance(b, (int, float)) for b in band) and 0.0 < band[0] < band[1]):
        out.append(f'the face size band {band} is not [min, max] with 0 < min < max')
    for key, version in floors.items():
        if not _TOKEN.match(key) or not _VERSION.match(str(version)):
            out.append(f'floor {key!r}: {version!r} is not a parameter name and a release')
    return out
