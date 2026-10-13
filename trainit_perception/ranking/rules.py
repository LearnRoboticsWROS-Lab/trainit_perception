"""The RULE: which term decides the order, and the refusal that keeps it meaningful.

ADR-0016 Decision 1: the detector publishes `Detection3DArray` **best first**, and that order IS
the ranking. It is the one channel that carries a rank across the perception contract without
either side learning the other's vocabulary -- a detector knows nothing about robots and a runtime
knows nothing about masks, and "best first" needs neither. So a rule is a configuration choice
(ADR-0014) whose only effect is the permutation applied before the array goes out, and `rank_order`
below is that step: one permutation, computed once, applied to everything a frame publishes.

ADR-0017 Decision 3: a rule whose terms are not computable **stops the node**. Since the published
order is the ranking, a rule nobody computes cannot be allowed to mean "whatever order the model
returned": a configuration typo would then leave the cell picking at random and looking like a
perception fault, which is the expensive kind of mistake. The asymmetry this replaces was
confirmed by execution -- an unset face radius turned `candidate_x_visible` into `candidate_2d3d`
SILENTLY (both 0,7000 on the same frame) and dropped the `radius_ratio` gate with it, while an
unset tool radius was loudly refused.

A rule is a PRODUCT of named terms, declared rather than coded, so a new term arrives as a row in
`RULES` and not as a branch in a node -- the same registry shape as `detectors/registry.py`,
`fit3d.METHODS` and `faces.FACES`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .faces import available_faces, face_params, make_face

# --- what a DETECTOR provides per detection: the vocabulary `computable_terms` reads ----------
# Declared by a Detector subclass as `PROVIDES`. It says what each `Detection` carries, which is
# what decides which terms can be computed at all -- and therefore which rules are configurable.
PROVIDES_SCORE = 'score'               # every detection has one, by the contract
PROVIDES_BOX = 'box_px'                # a pixel bounding box -> box_overlap
PROVIDES_MASK_AREA = 'mask_area_px'    # a mask's pixel count -> visible_fraction
PROVIDES_FIT3D = 'fit3d'               # a GraspFit over the depth -> pose_quality, clearance


def _face_params_sentence() -> str:
    """Every registered face and the parameters it reads, for a refusal message.

    DERIVED FROM THE REGISTRY, never written out here (ADR-0018 Decision 6). The message used to
    say "(for `disc`: expected_radius_m)", which stopped being the whole truth the moment a
    second face shipped -- and a refusal that names the wrong parameter sends the reader to the
    wrong field.
    """
    return '; '.join(f'{name}: {", ".join(face_params(name)) or "no parameters"}'
                     for name in available_faces())


#: term -> the configuration that makes it computable, for the refusal message. READ IT THROUGH
#: `term_needs(term)`, not directly: `{faces}` is filled in from the face registry when the
#: message is built, so a face registered from outside this package names its own parameters
#: without this table being edited (ADR-0018 Decision 6).
TERM_NEEDS: Dict[str, str] = {
    'score': 'every detection carries one',
    'pose_quality': 'a detector that runs a 3D fit over the depth inside the mask',
    'box_overlap': 'a detector that reports a pixel bounding box',
    'visible_fraction': ('a grasp face with a known size, on a detector that reports a mask '
                         'area -- set `grasp_face` and that face\'s own parameters ({faces})'),
    'clearance': 'tool_radius_m > 0 AND tool_reach_mm > 0, on a detector that runs a 3D fit',
    'edge_on': ('tool_max_tilt_deg > 0, on a detector whose 3D fit reports a TILT -- '
                '`plane_ransac` does, `centroid` claims no orientation and reports none'),
}


@dataclass(frozen=True)
class TermRequires:
    """What makes ONE term computable, as data (vNext M2.2, ADR-0021).

    ``provides``         the detection outputs it reads (``Detector.PROVIDES`` tokens)
    ``positive_params``  the parameters that must be set above 0
    ``face_sized``       it comes from the configured grasp FACE: computable when that face lists
                         it in ``ENABLES_TERMS`` and every one of the face's parameters is set

    A configurator reads this table, through ``trainit_perception.vocabulary.export()``, instead of
    re-implementing :func:`computable_terms` -- which stays the runtime's answer and is held to this
    table over every configuration by test_vocabulary.py. Names and requirements only: HOW a term
    is computed is not in it.
    """
    provides: Tuple[str, ...] = ()
    positive_params: Tuple[str, ...] = ()
    face_sized: bool = False


TERM_REQUIRES: Dict[str, TermRequires] = {
    'score': TermRequires(),
    'box_overlap': TermRequires(provides=(PROVIDES_BOX,)),
    'pose_quality': TermRequires(provides=(PROVIDES_FIT3D,)),
    'visible_fraction': TermRequires(provides=(PROVIDES_MASK_AREA,), face_sized=True),
    'clearance': TermRequires(provides=(PROVIDES_FIT3D,),
                              positive_params=('tool_radius_m', 'tool_reach_mm')),
    'edge_on': TermRequires(provides=(PROVIDES_FIT3D,), positive_params=('tool_max_tilt_deg',)),
}


def term_needs(term: str) -> str:
    """The sentence for `term`, with anything registry-derived filled in at the time of asking."""
    return TERM_NEEDS.get(term, 'a term this detector does not publish').replace(
        '{faces}', _face_params_sentence())


@dataclass(frozen=True)
class Rule:
    """A candidate rule: the product of its factors, some of them complemented.

    `complement` holds the factors that enter as `(1 - term)`, which is how an OCCLUSION measure
    becomes a GOODNESS measure without publishing the same number under two names.
    """
    name: str
    factors: Tuple[str, ...] = ()
    complement: FrozenSet[str] = field(default_factory=frozenset)
    note: str = ''

    def requires(self) -> FrozenSet[str]:
        return frozenset(self.factors)

    def value(self, terms: Mapping[str, float]) -> float:
        """The rule's number for one detection. KeyError if a factor is not there -- never 0."""
        v = 1.0
        for f in self.factors:
            t = float(terms[f])
            v *= (1.0 - t) if f in self.complement else t
        return v


def _rule(name, factors=(), complement=(), note=''):
    return Rule(name, tuple(factors), frozenset(complement), note)


# The shipped rules. `candidate_*` are the four a real bin-picking cell ran head to head; the `score_*` family is
# the same idea for a detector that has no 3D fit, so a 2D-only detector can still rank.
RULES: Dict[str, Rule] = {r.name: r for r in (
    _rule('detector_order', (),
          note="no ranking: publish in the detector's own order. An EXPLICIT choice, which is "
               "what distinguishes it from a rule that silently could not be computed."),
    _rule('score', ('score',),
          note="the detector's own confidence alone"),
    _rule('candidate_2d3d', ('score', 'pose_quality'),
          note='P(detected) x P(pose well formed). MEASURED to predict the POSE error (9,38 mm '
               'worst quintile against 1,68 mm best) and MEASURED NOT to predict the pick '
               '(correlation -0,17). Says nothing about occlusion.'),
    _rule('candidate_x_visible', ('score', 'pose_quality', 'visible_fraction'),
          note='... x how much of the face the mask actually shows. THE MEASURED WINNER: over 40 '
               'pours on two independent seeds, 39 of 40 lifted against 35, 34 brought exactly '
               'one against 30, 55 % success at a 1,44 mm hold tolerance against 40 % -- WITHOUT '
               'refusing more candidates (7,6 against 7,4) or leaving fewer plans (14,4 against '
               '14,7), so it wins by choosing better among the same candidates.'),
    _rule('candidate_x_nonoverlap', ('score', 'pose_quality', 'box_overlap'), {'box_overlap'},
          note='... x (1 - how much of its BOX another box covers). Measured: roughly a wash as a '
               'RANKING, and as a 20 % refusal it halves the rate -- which is why no refusal '
               'ships.'),
    _rule('candidate_x_clearance', ('score', 'pose_quality', 'clearance'),
          note='... x how much room the tool has, from the depth. Needs a tool radius and a grip '
               'window. Measured 0,00 on every candidate the picker chose, so it ships as a term '
               'and not as a default.'),
    _rule('candidate_x_square', ('score', 'pose_quality', 'edge_on'), {'edge_on'},
          note='... x (1 - how edge-on the face is presented), against the tilt the TOOL can '
               'still hold. The owner\'s second signal (ADR-0018 Decision 5). Ships as a rule '
               'and NOT as a default: the term has not been scored against external labels on '
               'a real cell, and `clearance` is the precedent for what an unmeasured factor '
               'does at the top of a ranking.'),
    _rule('candidate_x_visible_x_square',
          ('score', 'pose_quality', 'visible_fraction', 'edge_on'), {'edge_on'},
          note='THE OWNER\'S SENTENCE, both halves: well visible in the region I mean to grasp '
               'AND not presented edge-on. The two factors are kept apart on purpose -- folding '
               'the tilt into the visible fraction is the cos(tilt) correction that saturated it '
               'at 1,00 on two thirds of detections (ADR-0018 Decision 5).'),
    _rule('score_x_visible', ('score', 'visible_fraction'),
          note='the same occlusion idea for a detector with a mask but no 3D fit'),
    _rule('score_x_nonoverlap', ('score', 'box_overlap'), {'box_overlap'},
          note='boxes only: the fallback for a detector with no mask and no depth'),
    _rule('score_x_clearance', ('score', 'clearance'),
          note='a detector with depth and a tool, but no pose quality worth gating on'),
)}

#: The rules THIS package ships, captured before `register_rule` can add one: what
#: `trainit_perception.vocabulary.export()` reads (ADR-0021, vNext M2.2).
BUILTIN_RULES: Tuple[Rule, ...] = tuple(RULES.values())

#: Which rule is chosen when none is configured, best first. ADR-0017 Decision 4: visible wherever
#: a face area is available, box overlap as the fallback for a detector with no mask and no depth.
DEFAULT_PREFERENCE: Tuple[str, ...] = (
    'candidate_x_visible',
    'score_x_visible',
    'candidate_2d3d',
    'score_x_nonoverlap',
    'score',
    'detector_order',
)

#: what a configurator may write to mean "resolve it from what this detector can compute"
AUTO = 'auto'


def available_rules() -> List[str]:
    return sorted(RULES)


def register_rule(rule: Rule) -> Rule:
    """Register a rule from outside this package, for a term a detector publishes itself."""
    if not rule.name:
        raise ValueError('a candidate rule needs a non-empty name')
    RULES[rule.name] = rule
    return rule


def computable_terms(params: Mapping, provides: Iterable[str] = (PROVIDES_SCORE,)) -> Set[str]:
    """Which terms THIS detector, configured THIS way, can actually compute.

    Derived from the configured parameters and from what the detector declares it produces per
    detection (`Detector.PROVIDES`) -- the two things a configurator knows before a frame arrives.
    This is `check_rule`'s `computable` argument, and the reason an unset face radius is now as
    loud as an unset tool radius.
    """
    have = set(provides)
    out = {'score'}
    if PROVIDES_BOX in have:
        out.add('box_overlap')
    if PROVIDES_FIT3D in have:
        out.add('pose_quality')
    if PROVIDES_MASK_AREA in have:
        # WHICH terms a configured face unlocks is the FACE's declaration, asked for here rather
        # than written out (ADR-0018 Decision 6). Every shipped face unlocks `visible_fraction`,
        # whose denominator IS the face; a face that measured something else would say so, and
        # neither this function nor the Setup Assistant would need editing for it.
        face = make_face(str(params.get('grasp_face', 'disc') or 'disc'), params)
        if face.has_known_area:
            out.update(face.ENABLES_TERMS)
    if (PROVIDES_FIT3D in have and float(params.get('tool_radius_m', 0.0) or 0.0) > 0.0
            and float(params.get('tool_reach_mm', 0.0) or 0.0) > 0.0):
        out.add('clearance')
    # The tilt comes from the same 3D fit, and the limit it is measured against is the TOOL's.
    # What this static check cannot see is WHICH 3D method is configured: `centroid` returns the
    # camera axis for every fit and reports no tilt, so a rule needing `edge_on` on a
    # centroid-only detector is refused at the first frame by `rank_order` (the term is absent,
    # loudly) rather than here at start-up. `PROVIDES` has no token for "my fit has an
    # orientation" and inventing one is a two-repo change ADR-0018 does not take.
    if PROVIDES_FIT3D in have and float(params.get('tool_max_tilt_deg', 0.0) or 0.0) > 0.0:
        out.add('edge_on')
    return out


def computable_rules(computable: Iterable[str]) -> List[str]:
    """Every rule whose factors this detector can compute -- what a configurator offers."""
    have = frozenset(computable)
    return sorted(n for n, r in RULES.items() if r.requires() <= have)


def check_rule(rule: str, *, computable: Iterable[str]) -> str:
    """The configured rule back, or `ValueError` saying what is missing and what would supply it.

    Called once, where the configuration is read, and loud on purpose: since the published ORDER
    of the detections is the ranking (ADR-0016 Decision 1), a rule nobody computes cannot be
    allowed to mean "whatever order the model returned". A name is checked where it is READ, not
    where it is used, so the node stops at start-up rather than mis-picking for a shift.
    """
    have = frozenset(computable)
    if rule not in RULES:
        raise ValueError(
            f'candidate_rule {rule!r} is not a rule this package knows, so nothing could rank by '
            f'it -- and the published order IS the ranking (ADR-0016), so an unknown rule cannot '
            f'simply be ignored. Accepted: {", ".join(available_rules())}')
    missing = sorted(RULES[rule].requires() - have)
    if missing:
        needs = '; '.join(f'{m} needs {term_needs(m)}' for m in missing)
        raise ValueError(
            f'candidate_rule {rule!r} needs the term(s) {", ".join(missing)}, which this detector '
            f'cannot compute as configured -- {needs}. Refusing to publish an order that is not '
            f'the ranking that was asked for (ADR-0017 Decision 3). Computable terms: '
            f'{", ".join(sorted(have))}. Rules available here: '
            f'{", ".join(computable_rules(have))}')
    return rule


def default_rule(*, computable: Iterable[str]) -> str:
    """The best rule this detector can compute, in the measured preference order.

    This is what an unset (or `auto`) `candidate_rule` resolves to, and it is NOT the silent
    degeneration Decision 3 closes: nothing was configured, so nothing is being betrayed -- and
    the caller logs what it resolved to.
    """
    have = frozenset(computable)
    for name in DEFAULT_PREFERENCE:
        if RULES[name].requires() <= have:
            return name
    return 'detector_order'                      # pragma: no cover - 'score' is always computable


def resolve_rule(rule, *, computable: Iterable[str]) -> str:
    """`auto`, empty or unset -> `default_rule`; anything else -> `check_rule`."""
    if rule is None or str(rule).strip() in ('', AUTO):
        return default_rule(computable=computable)
    return check_rule(str(rule).strip(), computable=computable)


def rank_order(terms: Sequence[Mapping[str, float]], rule: str) -> List[int]:
    """The indices of `terms`, BEST FIRST by `rule`. Equal values keep the detector's own order.

    Returns the PERMUTATION rather than the sorted terms, because every list a frame publishes
    beside the detections has to be reordered by the same one (ADR-0016 Decision 1): the
    detections, the quality sidecar, an overlay's annotations. Three lists that must stay in step
    and a fourth that may drift is how the next edit here introduces a bug.

    Raises `ValueError` when a detection carries no such term, or when the value is not finite,
    because the one thing this must never do is hand back the order it was given as though it were
    a ranking. (A NaN sort key does exactly that, silently.)
    """
    if rule not in RULES:
        raise ValueError(f'unknown candidate_rule {rule!r} '
                         f'(accepted: {", ".join(available_rules())})')
    r = RULES[rule]
    values: List[float] = []
    for i, t in enumerate(terms):
        missing = sorted(r.requires() - set(t))
        if missing:
            raise ValueError(
                f'detection {i} of {len(terms)} carries no {", ".join(missing)} term (it has: '
                f'{", ".join(sorted(t))}) -- refusing to publish an order that is not a ranking')
        v = r.value(t)
        if not np.isfinite(v):
            raise ValueError(
                f'detection {i} of {len(terms)} scores {v} under {rule!r}: a non-finite key would '
                f'sort arbitrarily and the order IS the ranking (terms: '
                f'{ {k: t[k] for k in sorted(r.requires())} })')
        values.append(v)
    # `sorted` is stable and `reverse=True` does not reverse equal elements, so a tie keeps the
    # order the detector produced -- which is what makes `detector_order` (no factors, so every
    # value 1.0) mean exactly "leave it alone".
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)
