"""The rule, the refusal and the permutation (ADR-0016 Decision 1, ADR-0017 Decisions 3 and 4).

The published ORDER of the detections is the ranking, so a rule nobody can compute must stop the
node rather than quietly mean "whatever order the detector returned".
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import R_FACE                            # noqa: E402

from trainit_perception.detectors import ColorMaskDetector, make_detector  # noqa: E402
from trainit_perception.ranking import (PROVIDES_BOX, PROVIDES_FIT3D,  # noqa: E402
                                        PROVIDES_MASK_AREA, PROVIDES_SCORE, Rule, available_rules,
                                        check_rule, computable_rules, computable_terms,
                                        default_rule, rank_order, register_rule, resolve_rule)

FULL = {PROVIDES_SCORE, PROVIDES_BOX, PROVIDES_MASK_AREA, PROVIDES_FIT3D}


# --- what is computable, derived from the CONFIGURED parameters ------------------------------

def test_computable_terms_comes_from_the_parameters_and_from_what_the_detector_provides():
    assert computable_terms({}, {PROVIDES_SCORE}) == {'score'}
    assert computable_terms({}, {PROVIDES_SCORE, PROVIDES_BOX}) == {'score', 'box_overlap'}
    # a mask alone is not enough: visible_fraction needs the FACE's size to divide by
    assert computable_terms({}, {PROVIDES_SCORE, PROVIDES_MASK_AREA}) == {'score'}
    assert computable_terms({'expected_radius_m': R_FACE},
                            {PROVIDES_SCORE, PROVIDES_MASK_AREA}) == {'score', 'visible_fraction'}
    # clearance needs BOTH tool numbers: a radius with no grip window can only ever answer 1,0
    assert 'clearance' not in computable_terms({'tool_radius_m': 0.01}, FULL)
    assert 'clearance' not in computable_terms({'tool_reach_mm': 8.0}, FULL)
    assert 'clearance' in computable_terms({'tool_radius_m': 0.01, 'tool_reach_mm': 8.0}, FULL)


def test_THE_SILENT_DEGENERATION_IS_NOW_A_REFUSAL():
    """THE REGRESSION THIS WHOLE DECISION EXISTS FOR (ADR-0017 Decision 3).

    VERIFIED on the pre-ADR-0017 code, on one synthetic frame: with the face radius unset,
    `visible_fraction` returned exactly 1.0 so `candidate_x_visible` and `candidate_2d3d` both
    read 0,7000 -- the configured rule had silently become a different rule -- and the
    `radius_ratio` gate vanished from the quality dict, which RAISED `pose_quality` from 0,8913 to
    1,0000 on the same fit. Meanwhile the same omission on the TOOL radius was loudly refused.
    Both are now refusals.
    """
    with pytest.raises(ValueError) as e:
        check_rule('candidate_x_visible', computable=computable_terms({}, FULL))
    assert 'visible_fraction' in str(e.value)
    assert 'expected_radius_m' in str(e.value), 'the message has to say what would supply it'
    with pytest.raises(ValueError) as e2:
        check_rule('candidate_x_clearance', computable=computable_terms({}, FULL))
    assert 'tool_radius_m' in str(e2.value) and 'tool_reach_mm' in str(e2.value)
    # and with the face configured, the same rule is accepted
    ok = computable_terms({'expected_radius_m': R_FACE}, FULL)
    assert check_rule('candidate_x_visible', computable=ok) == 'candidate_x_visible'


def test_an_unknown_rule_is_refused_where_the_configuration_is_READ():
    with pytest.raises(ValueError) as e:
        check_rule('candidate_x_vibes', computable=computable_terms({}, FULL))
    assert 'candidate_2d3d' in str(e.value), 'the message has to say what IS available'
    assert 'candidate_x_vibes' in str(e.value)


def test_the_refusal_names_the_rules_that_WOULD_work_here():
    with pytest.raises(ValueError) as e:
        check_rule('candidate_2d3d', computable={'score', 'box_overlap'})
    msg = str(e.value)
    assert 'score_x_nonoverlap' in msg and 'pose_quality' in msg


# --- the default: measured, and it never silently replaces a configured rule -----------------

def test_the_default_is_VISIBLE_wherever_a_face_area_is_available():
    """ADR-0017 Decision 4: visible wins and replicates on two seeds; box overlap is the fallback
    for a detector with no mask and no depth."""
    full_face = computable_terms({'expected_radius_m': R_FACE, 'tool_radius_m': 0.01,
                                  'tool_reach_mm': 8.0}, FULL)
    assert default_rule(computable=full_face) == 'candidate_x_visible'
    # a mask and a face but no 3D fit: the same idea without the pose factor
    assert default_rule(computable=computable_terms(
        {'expected_radius_m': R_FACE}, {PROVIDES_SCORE, PROVIDES_MASK_AREA})) == 'score_x_visible'
    # a 3D fit but no face size: the pose factor alone
    assert default_rule(computable=computable_terms({}, FULL)) == 'candidate_2d3d'
    # no mask and no depth: boxes only
    assert default_rule(computable={'score', 'box_overlap'}) == 'score_x_nonoverlap'
    assert default_rule(computable={'score'}) == 'score'


def test_auto_and_empty_resolve_but_a_NAMED_rule_is_checked():
    have = {'score', 'box_overlap'}
    assert resolve_rule('auto', computable=have) == 'score_x_nonoverlap'
    assert resolve_rule('', computable=have) == 'score_x_nonoverlap'
    assert resolve_rule(None, computable=have) == 'score_x_nonoverlap'
    assert resolve_rule('score', computable=have) == 'score'
    with pytest.raises(ValueError):
        resolve_rule('candidate_2d3d', computable=have)


def test_THE_20_PERCENT_OVERLAP_REFUSAL_IS_NOT_SHIPPED():
    """MEASURED: as a ranking, box overlap is a wash; as a 20 % REFUSAL it throws away 18,1 of
    22,1 candidates, leaves 4,0 against the five the cell needs, and halves the pick rate. No
    rule here filters anything -- a rule is a permutation (ADR-0017 Decision 4)."""
    assert all(not n.startswith('max_overlap') for n in available_rules())
    for name in available_rules():
        from trainit_perception.ranking import RULES
        assert RULES[name].requires() <= {'score', 'pose_quality', 'box_overlap',
                                          'visible_fraction', 'clearance', 'edge_on'}
    # ranking by (1 - overlap) keeps every candidate: 4 in, 4 out, merely reordered
    terms = [{'score': 0.9, 'box_overlap': ov} for ov in (0.0, 0.28, 0.56, 0.95)]
    assert sorted(rank_order(terms, 'score_x_nonoverlap')) == [0, 1, 2, 3]


# --- the permutation -------------------------------------------------------------------------

def test_best_first_and_it_returns_the_PERMUTATION():
    terms = [{'score': 0.5, 'pose_quality': 0.4}, {'score': 0.9, 'pose_quality': 0.9},
             {'score': 0.7, 'pose_quality': 0.8}]
    assert rank_order(terms, 'candidate_2d3d') == [1, 2, 0]
    assert rank_order([], 'candidate_2d3d') == []
    assert rank_order(terms[:1], 'candidate_2d3d') == [0]


def test_a_tie_keeps_the_detectors_own_order():
    terms = [{'score': 0.5, 'pose_quality': 0.5} for _ in range(4)]
    assert rank_order(terms, 'candidate_2d3d') == [0, 1, 2, 3]
    terms[2] = {'score': 1.0, 'pose_quality': 1.0}
    assert rank_order(terms, 'candidate_2d3d') == [2, 0, 1, 3]


def test_detector_order_IS_the_identity_and_is_an_EXPLICIT_choice():
    """Not the same thing as a rule that could not be computed: this one was asked for."""
    terms = [{'score': s} for s in (0.1, 0.9, 0.5)]
    assert rank_order(terms, 'detector_order') == [0, 1, 2]
    assert check_rule('detector_order', computable={'score'}) == 'detector_order'


def test_THE_RULE_DECIDES_THE_ORDER_and_two_rules_disagree():
    """The case the rank exists for: the higher-scoring candidate is the covered one."""
    terms = [
        {'score': 0.90, 'pose_quality': 0.80, 'visible_fraction': 0.40, 'box_overlap': 0.70},
        {'score': 0.70, 'pose_quality': 0.80, 'visible_fraction': 0.95, 'box_overlap': 0.10},
    ]
    assert rank_order(terms, 'candidate_2d3d') == [0, 1]
    assert rank_order(terms, 'candidate_x_visible') == [1, 0]
    assert rank_order(terms, 'candidate_x_nonoverlap') == [1, 0]


def test_a_term_that_is_not_there_is_REFUSED_rather_than_ignored():
    terms = [{'score': 0.9, 'pose_quality': 0.9, 'clearance': 1.0},
             {'score': 0.5, 'pose_quality': 0.5}]
    with pytest.raises(ValueError) as e:
        rank_order(terms, 'candidate_x_clearance')
    assert 'clearance' in str(e.value)
    assert 'not a ranking' in str(e.value)


def test_a_NON_FINITE_key_is_refused_because_a_NaN_sorts_ARBITRARILY():
    terms = [{'score': 0.9, 'pose_quality': 0.9}, {'score': float('nan'), 'pose_quality': 0.5}]
    with pytest.raises(ValueError) as e:
        rank_order(terms, 'candidate_2d3d')
    assert 'nan' in str(e.value).lower()


def test_an_unknown_rule_cannot_sneak_in_through_rank_order():
    with pytest.raises(ValueError):
        rank_order([{'score': 1.0}], 'candidate_x_vibes')


# --- the registry shape ----------------------------------------------------------------------

def test_a_rule_FROM_OUTSIDE_the_package_plugs_in():
    try:
        register_rule(Rule('test_score_x_clear_face',
                           ('score', 'clearance', 'visible_fraction'), frozenset()))
        assert 'test_score_x_clear_face' in available_rules()
        have = {'score', 'clearance', 'visible_fraction'}
        assert 'test_score_x_clear_face' in computable_rules(have)
        assert rank_order([{'score': 1.0, 'clearance': 0.5, 'visible_fraction': 1.0},
                           {'score': 1.0, 'clearance': 1.0, 'visible_fraction': 1.0}],
                          'test_score_x_clear_face') == [1, 0]
    finally:
        from trainit_perception.ranking import RULES
        RULES.pop('test_score_x_clear_face', None)


# --- and the whole thing as a DETECTOR parameter (Decision 1) --------------------------------

def test_the_rule_is_a_DETECTOR_PARAMETER_and_a_bad_one_stops_the_CONSTRUCTOR():
    """`make_detector` is called in the node's constructor, so this is the node stopping."""
    with pytest.raises(ValueError) as e:
        make_detector('color_mask', {'candidate_rule': 'candidate_x_visible'})
    assert 'pose_quality' in str(e.value), 'a colour mask runs no 3D fit'
    with pytest.raises(ValueError):
        make_detector('color_mask', {'candidate_rule': 'score_x_visible'})   # no face radius
    det = make_detector('color_mask', {'candidate_rule': 'score_x_visible',
                                       'expected_radius_m': R_FACE})
    assert det.ranker.rule == 'score_x_visible'
    # and a typo in the FACE stops it just as loudly: nothing here guesses
    with pytest.raises(KeyError):
        make_detector('color_mask', {'grasp_face': 'hexagon'})


def test_the_shipped_default_resolves_from_what_the_colour_mask_can_compute():
    assert ColorMaskDetector.defaults()['candidate_rule'] == 'auto'
    assert ColorMaskDetector({}).ranker.rule == 'score_x_nonoverlap'
    assert ColorMaskDetector({'expected_radius_m': R_FACE}).ranker.rule == 'score_x_visible'
    assert ColorMaskDetector({'candidate_rule': 'detector_order'}).ranker.rule == 'detector_order'
    assert 'score_x_nonoverlap' in ColorMaskDetector({}).ranker.describe()
