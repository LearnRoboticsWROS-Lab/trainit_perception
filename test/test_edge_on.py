"""The EDGE-ON TERM (ADR-0018 Decision 5), and the thing it must never become again.

The owner's second signal: *"se l'oggetto e ben visibile nella zona in cui lo voglio prendere,
allora e un buon candidato; se e ben visibile ma messo di taglio allora non e un buon candidato"*.

It was first tried as a CORRECTION -- dividing `visible_fraction` by `cos(tilt)` -- and that
destroyed the only term that measures: the fraction saturated at 1,00 on 65 % of detections where
the labelled truth said 28 %. So it comes back as a TERM, a factor of its own, with no cosine
anywhere in it. These tests check the arithmetic, the registry wiring and -- the part no unit test
could do on its own -- that the area term is still untouched. `scripts/score_term.py --term
edge_on` is the half that scores it against external labels (ADR-0017 Decision 6).
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K, R_FACE, synth_disc               # noqa: E402

from trainit_perception.detectors import ColorMaskDetector, Detection  # noqa: E402
from trainit_perception.ranking import (DEFAULT_PREFERENCE, PROVIDES_BOX,  # noqa: E402
                                        PROVIDES_FIT3D, PROVIDES_MASK_AREA, PROVIDES_SCORE, RULES,
                                        Ranker, centroid, check_rule, computable_rules,
                                        computable_terms, default_rule, edge_on, make_face,
                                        plane_ransac, rank_order, ranking_defaults,
                                        visible_fraction)

FULL = {PROVIDES_SCORE, PROVIDES_BOX, PROVIDES_MASK_AREA, PROVIDES_FIT3D}
FACE = make_face('disc', {'expected_radius_m': R_FACE})


class _FittingDetector(ColorMaskDetector):
    """A stand-in for a detector that runs a 3D fit, so the term can be exercised end to end."""

    method = 'test_color_mask_3d_tilt'
    PROVIDES = frozenset({PROVIDES_SCORE, PROVIDES_MASK_AREA, PROVIDES_FIT3D})

    @classmethod
    def defaults(cls):
        return {**ColorMaskDetector.defaults(), **cls.ranking_defaults()}


def tilted_fit(tilt_deg, *, seed=0):
    """A disc presented at `tilt_deg`, fitted. `plane_ransac` recovers the angle exactly on an
    analytic raster, so the term is being fed the truth and nothing else."""
    nx, nz = math.sin(math.radians(tilt_deg)), -math.cos(math.radians(tilt_deg))
    mask, depth, _c, _n = synth_disc((0.0, 0.0, 0.600), (nx, 0.0, nz), seed=seed)
    fit = plane_ransac(mask, depth, K, face=FACE)
    assert fit.quality['tilt_deg'] == pytest.approx(tilt_deg, abs=1.0)
    return fit, int(mask.sum())


# --- the function ----------------------------------------------------------------------------

def test_it_is_0_square_on_and_1_at_the_TOOLS_limit_and_LINEAR_in_the_ANGLE():
    assert edge_on(0.0, max_tilt_deg=45.0) == 0.0
    assert edge_on(45.0, max_tilt_deg=45.0) == 1.0
    assert edge_on(90.0, max_tilt_deg=45.0) == 1.0           # clipped, never above 1
    assert edge_on(22.5, max_tilt_deg=45.0) == pytest.approx(0.5)
    assert edge_on(15.0, max_tilt_deg=45.0) == pytest.approx(1 / 3)
    assert edge_on(-5.0, max_tilt_deg=45.0) == 0.0           # clipped the other way too
    # LINEAR IN THE ANGLE, and there is no cosine in it anywhere -- which is the whole decision
    assert edge_on(60.0, max_tilt_deg=90.0) == pytest.approx(2 / 3)
    assert edge_on(60.0, max_tilt_deg=90.0) != pytest.approx(1 - math.cos(math.radians(60)))


def test_the_LIMIT_IS_THE_TOOLS_and_0_switches_the_term_OFF():
    """A compliant cup seals up to some tilt and a two-finger gripper to another, so the
    saturation point is a tool property -- the same reasoning that made `approach_clearance`
    require the grip window instead of defaulting it."""
    assert edge_on(30.0, max_tilt_deg=20.0) == 1.0           # past a strict tool's limit
    assert edge_on(30.0, max_tilt_deg=60.0) == pytest.approx(0.5)
    assert edge_on(30.0, max_tilt_deg=0.0) == 0.0            # switched off = no penalty
    with pytest.raises(TypeError):
        edge_on(30.0)                                        # never a default


def test_a_TILT_THAT_COULD_NOT_BE_MEASURED_IS_1_AND_NOT_0():
    """The same asymmetry as a NaN obstruction, in the same direction: the answer to a question
    that could not be asked is the pessimistic one. A NaN here must not make a candidate look
    square to the lens."""
    assert edge_on(float('nan'), max_tilt_deg=45.0) == 1.0
    assert edge_on(float('inf'), max_tilt_deg=45.0) == 1.0


# --- IT IS A TERM AND NOT A CORRECTION -------------------------------------------------------

def test_THE_AREA_TERM_IS_UNTOUCHED_which_is_the_whole_of_DECISION_5():
    """A correction claims to know how much area the tilt costs; a term only says how tilted it
    is. MEASURED on the raster: a disc at 60 degrees really does show cos(60) of its pixels, and
    `visible_fraction` reports that -- while `edge_on` reports the angle, separately, and neither
    number is folded into the other."""
    square_on = FACE.expected_area_px(0.600, K)
    for tilt in (0, 30, 60):
        fit, area = tilted_fit(tilt)
        vf = visible_fraction(area, square_on)
        assert vf == pytest.approx(math.cos(math.radians(tilt)), abs=0.03)
        # the term is computed from the SAME fit and does not change that number by a hair
        assert visible_fraction(area, square_on, tilt) == vf
        assert edge_on(fit.tilt_deg, max_tilt_deg=90.0) == pytest.approx(tilt / 90.0, abs=0.02)
    # and the historical correction is what it would have cost: a saturated area term
    _fit, area60 = tilted_fit(60)
    corrected = min(1.0, visible_fraction(area60, square_on) / math.cos(math.radians(60)))
    assert corrected == pytest.approx(1.0, abs=0.03), 'the cos version read "fully visible"'


def test_WELL_VISIBLE_BUT_EDGE_ON_IS_RANKED_BELOW_WELL_VISIBLE_AND_SQUARE_ON():
    """The owner's sentence, as a permutation. Two candidates the area term cannot separate --
    both masks measured against their own tilted expectation -- and the tilt decides."""
    terms = [
        {'score': 0.90, 'pose_quality': 0.80, 'visible_fraction': 0.95, 'edge_on': 0.90},
        {'score': 0.90, 'pose_quality': 0.80, 'visible_fraction': 0.95, 'edge_on': 0.05},
    ]
    assert rank_order(terms, 'candidate_x_visible') == [0, 1], 'visible alone cannot tell'
    assert rank_order(terms, 'candidate_x_visible_x_square') == [1, 0]
    assert rank_order(terms, 'candidate_x_square') == [1, 0]
    # and it does NOT overrule the area: a square-on candidate that is half buried still loses
    buried = [{'score': 0.9, 'pose_quality': 0.8, 'visible_fraction': 0.30, 'edge_on': 0.05},
              {'score': 0.9, 'pose_quality': 0.8, 'visible_fraction': 0.95, 'edge_on': 0.40}]
    assert rank_order(buried, 'candidate_x_visible_x_square') == [1, 0]


def test_it_enters_the_rule_COMPLEMENTED_like_an_occlusion_measure():
    for name in ('candidate_x_square', 'candidate_x_visible_x_square'):
        assert RULES[name].complement == frozenset({'edge_on'})
    assert RULES['candidate_x_square'].value(
        {'score': 1.0, 'pose_quality': 1.0, 'edge_on': 0.25}) == pytest.approx(0.75)


def test_IT_IS_NOT_A_DEFAULT_because_it_has_not_been_SCORED_on_a_real_cell():
    """`clearance` is the precedent (ADR-0017 Decision 4): it ships as a published term and not
    as a ranking default, because a factor that reads the same on every candidate the picker
    chooses turns a product rule into a constant -- and ties publish the detector's own order."""
    assert 'candidate_x_square' not in DEFAULT_PREFERENCE
    assert 'candidate_x_visible_x_square' not in DEFAULT_PREFERENCE
    everything = computable_terms({'expected_radius_m': R_FACE, 'tool_radius_m': 0.01,
                                   'tool_reach_mm': 8.0, 'tool_max_tilt_deg': 45.0}, FULL)
    assert 'edge_on' in everything
    assert default_rule(computable=everything) == 'candidate_x_visible'
    # the failure the precedent names: too strict a limit makes every candidate read 1,00
    strict = [{'score': 0.5, 'pose_quality': 0.9, 'edge_on': edge_on(t, max_tilt_deg=20.0)}
              for t in (33.0, 45.0, 63.0)]
    assert [t['edge_on'] for t in strict] == [1.0, 1.0, 1.0]
    assert rank_order(strict, 'candidate_x_square') == [0, 1, 2], 'a tie IS the detector order'


# --- the registry wiring ---------------------------------------------------------------------

def test_the_term_is_computable_only_with_a_FIT_and_the_TOOLS_limit():
    assert 'edge_on' not in computable_terms({}, FULL)
    assert 'edge_on' not in computable_terms({'tool_max_tilt_deg': 45.0},
                                             {PROVIDES_SCORE, PROVIDES_MASK_AREA})
    assert 'edge_on' in computable_terms({'tool_max_tilt_deg': 45.0}, FULL)
    assert 'candidate_x_square' in computable_rules(computable_terms(
        {'tool_max_tilt_deg': 45.0}, FULL))


def test_a_rule_that_needs_it_is_REFUSED_and_the_message_says_WHERE_TO_SET_IT():
    with pytest.raises(ValueError) as e:
        check_rule('candidate_x_square', computable=computable_terms({}, FULL))
    msg = str(e.value)
    assert 'edge_on' in msg and 'tool_max_tilt_deg' in msg
    assert 'plane_ransac' in msg and 'centroid' in msg, 'it must name the method that reports it'
    ok = computable_terms({'tool_max_tilt_deg': 45.0}, FULL)
    assert check_rule('candidate_x_square', computable=ok) == 'candidate_x_square'


def test_the_tools_limit_is_a_DETECTOR_PARAMETER_beside_the_other_two():
    defaults = ranking_defaults(FULL)
    assert defaults['tool_max_tilt_deg'] == 0.0              # 0 = the term is switched off
    assert {'tool_radius_m', 'tool_reach_mm', 'tool_max_tilt_deg'} <= set(defaults)
    # a detector with no 3D fit has no use for it, so it is not rendered
    assert 'tool_max_tilt_deg' not in ranking_defaults({PROVIDES_SCORE, PROVIDES_MASK_AREA})
    with pytest.raises(ValueError):
        _FittingDetector({'candidate_rule': 'candidate_x_square'})     # no limit set
    det = _FittingDetector({'candidate_rule': 'candidate_x_square', 'tool_max_tilt_deg': 45.0})
    assert det.ranker.rule == 'candidate_x_square'
    assert det.ranker.tool_max_tilt_deg == 45.0


# --- through the Ranker, on real fits --------------------------------------------------------

def test_the_term_rides_on_the_SIDECAR_and_orders_real_fits():
    ranker = Ranker.from_params({'candidate_rule': 'candidate_x_visible_x_square',
                                 'expected_radius_m': R_FACE, 'tool_max_tilt_deg': 45.0},
                                provides=_FittingDetector.PROVIDES)
    dets = []
    for tilt in (60, 10):
        fit, area = tilted_fit(tilt, seed=tilt)
        dets.append(Detection('cap', 0.9, fit.point, normal=fit.normal, area_px=area, fit=fit))
    out = ranker.apply(dets, K)
    assert len(out) == 2
    assert out[0].extra['edge_on'] < out[1].extra['edge_on'], 'the square-on one comes first'
    assert out[0].extra['edge_on'] == pytest.approx(10 / 45, abs=0.05)
    assert out[1].extra['edge_on'] == pytest.approx(1.0), '60 deg is past a 45 deg tool'
    # every computable rule travels too, so a tuner can see what another one would have said
    assert {'candidate_x_square', 'candidate_x_visible_x_square'} <= set(out[0].extra)


def test_A_METHOD_THAT_CLAIMS_NO_ORIENTATION_PUBLISHES_NO_TILT_AND_IS_REFUSED_LOUDLY():
    """THE FAILURE THIS TERM COULD HAVE HAD. `centroid` returns the camera axis for every fit, so
    reading a tilt off it would hand every candidate a 0 -- a rule multiplying by a constant,
    wearing the name of a measurement (ADR-0017 Decision 6). It reports no `tilt_deg`, the Ranker
    therefore computes no `edge_on`, and `rank_order` refuses rather than inventing one."""
    mask, depth, _c, _n = synth_disc((0.0, 0.0, 0.600), (0.5, 0.0, -0.866))      # 30 deg of tilt
    flat = centroid(mask, depth, K, face=FACE)
    assert 'tilt_deg' not in flat.quality
    assert flat.tilt_deg == 0.0, 'the property reads 0 by construction -- which is the trap'
    det = Detection('cap', 0.9, flat.point, normal=None, area_px=int(mask.sum()), fit=flat)
    ranker = Ranker.from_params({'candidate_rule': 'candidate_x_square', 'tool_max_tilt_deg': 45.0,
                                 'expected_radius_m': R_FACE},
                                provides=_FittingDetector.PROVIDES)
    terms = ranker.terms([det], K)
    assert 'edge_on' not in terms[0]
    with pytest.raises(ValueError) as e:
        ranker.order(terms)
    assert 'edge_on' in str(e.value) and 'not a ranking' in str(e.value)


def test_a_fit_that_FAILED_is_dropped_before_the_term_is_ever_asked_for():
    empty = np.zeros((224, 256), bool)
    depth = np.full((224, 256), np.nan, np.float32)
    bad = plane_ransac(empty, depth, K, face=FACE)
    assert not bad.ok
    det = Detection('cap', 0.9, (0, 0, 0.6), fit=bad)
    ranker = Ranker.from_params({'candidate_rule': 'candidate_x_square', 'tool_max_tilt_deg': 45.0,
                                 'expected_radius_m': R_FACE},
                                provides=_FittingDetector.PROVIDES)
    assert ranker.apply([det], K) == []
