"""The ranking half of trainit_perception's configuration vocabulary (vNext M2.2, ADR-0021).

``trainit_perception.vocabulary.export()`` assembles the whole record; this module contributes
what the ranking DECLARES: the detection outputs a term can read, the rules a configurator may
select and the terms each multiplies (in order), the default preference, what makes each term
computable, the grasp faces and the parameters each is sized by (in the order the face declares
them, which is the order a generator writes them), the default face, the size band, and the
ranking parameters' version floors.

NAMES, KINDS AND REQUIREMENTS ONLY. No rule note, no complement, no formula, no gate value, no
extent term: the export is a closed allow-list (ADR-0021, Decision 9). And only the BUILT-IN tables
are read, so a rule or face registered from outside this package is not exported -- it reaches a
configurator through its own package's snapshot, never by accident through this one.
"""
from __future__ import annotations

from typing import Dict

from .faces import BUILTIN_FACES, DEFAULT_FACE, MAX_FACE_M, MIN_FACE_M
from .ranker import PARAM_SINCE
from .rules import (AUTO, BUILTIN_RULES, DEFAULT_PREFERENCE, PROVIDES_BOX, PROVIDES_FIT3D,
                    PROVIDES_MASK_AREA, PROVIDES_SCORE, TERM_REQUIRES)


def export_ranking() -> Dict[str, object]:
    """The ranking's configuration vocabulary, as plain data in a fixed order."""
    return {
        'provides': sorted({PROVIDES_SCORE, PROVIDES_BOX, PROVIDES_MASK_AREA, PROVIDES_FIT3D}),
        'auto': AUTO,
        'rules': {rule.name: list(rule.factors) for rule in BUILTIN_RULES},
        'default_preference': list(DEFAULT_PREFERENCE),
        'terms': {name: {'provides': sorted(req.provides),
                         'positive_params': list(req.positive_params),
                         'face_sized': bool(req.face_sized)}
                  for name, req in TERM_REQUIRES.items()},
        'faces': {cls.name: {'params': list(cls.PARAMS),
                             'enables_terms': list(cls.ENABLES_TERMS)}
                  for cls in BUILTIN_FACES},
        'default_face': DEFAULT_FACE,
        'face_size_band_m': [float(MIN_FACE_M), float(MAX_FACE_M)],
        'param_since': dict(PARAM_SINCE),
    }
