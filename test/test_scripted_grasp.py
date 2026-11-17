"""The scripted grasp provider's scenario (vNext M4.5, §12): parsed and validated in full, ROS-free.

Pinned: a candidate without an orientation gets SurfaceGrasp's rule (its +Z is the approach); approach and
orientation are normalised; ids default per step; `unavailable` and `candidates` are exclusive; every problem of a
scenario is named at once; the cursor advances one step per call and `after_end` decides what follows the end.
"""
import numpy as np
import pytest

from trainit_perception.grasp import ConfigError
from trainit_perception.grasp.scripted import (Scenario, ScriptedConfig, Step, diagnostics, load_scenario,
                                               parse_scenario)

CFG = ScriptedConfig.from_params({'scenario_file': 'unused.yaml', 'frame_id': 'cam_optical'})
C1 = {'position': [0.0, 0.0, 0.5], 'approach': [0, 0, 2], 'standoff_m': 0.05, 'eef': 'main', 'site': 'suction',
      'score': 0.9, 'confidence': 0.8, 'object_class': 'capsule'}


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_a_scenario_parses_with_the_orientation_computed_from_the_approach():
    tilted = dict(C1, approach=[0.3, 0.0, 1.0], id='b')
    steps = parse_scenario({'steps': [{'candidates': [C1, tilted], 'stamp_offset_ms': -500, 'delay_ms': 20},
                                      {'candidates': []}, {'unavailable': True}]}, CFG)
    assert len(steps) == 3
    a, b = steps[0].candidates
    assert a.id == 'scripted@0.0' and b.id == 'b'
    assert a.approach == (0.0, 0.0, 1.0) and a.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    R = rot(b.orientation_xyzw)
    assert np.allclose(R[:, 2], b.approach) and abs(np.linalg.norm(b.approach) - 1) < 1e-12
    assert (steps[0].stamp_offset_ms, steps[0].delay_ms) == (-500.0, 20.0)
    assert steps[1].candidates == () and not steps[1].unavailable and steps[2].unavailable


def test_an_explicit_orientation_is_kept_normalised_and_the_defaults_apply():
    c = parse_scenario({'steps': [{'candidates': [{'position': [1, 2, 3], 'approach': [1, 0, 0],
                                                   'orientation': [0, 0, 0, 2], 'standoff_m': 0}]}]}, CFG)[0]
    cand = c.candidates[0]
    assert cand.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert (cand.eef, cand.site, cand.score, cand.confidence, cand.object_class) == ('', '', 1.0, 1.0, '')


def test_every_problem_of_a_scenario_is_named_at_once():
    bad = {'steps': [
        {'candidates': [{'position': [0, 0], 'approach': [0, 0, 0], 'standoff_m': -1, 'score': 2, 'colour': 1}]},
        {'candidates': [], 'unavailable': True},
        {'delay_ms': -5},
        {'candidates': [dict(C1, id='x'), dict(C1, id='x')]},
        {'candidates': [dict(C1, approach=[0, 0, -1])]},
    ]}
    with pytest.raises(ConfigError) as err:
        parse_scenario(bad, CFG)
    text = str(err.value)
    for says in ('steps[0].candidates[0]: unknown key(s) colour', 'position must be three', 'approach must be',
                 'standoff_m must be', 'score must be in [0, 1]', 'steps[1]: a step is EITHER',
                 'steps[2]: a step is EITHER', 'delay_ms must be', "candidate id(s) ['x'] repeat",
                 'antiparallel'):
        assert says in text, says


@pytest.mark.parametrize('doc, says', [(None, 'exactly one key'), ({'steps': []}, 'non-empty'),
                                       ({'steps': [{}], 'extra': 1}, 'exactly one key'),
                                       ({'steps': ['x']}, 'a step is a mapping')])
def test_a_malformed_scenario_is_refused(doc, says):
    with pytest.raises(ConfigError, match=says):
        parse_scenario(doc, CFG)


def test_load_scenario_names_a_missing_or_broken_file(tmp_path):
    with pytest.raises(ConfigError, match='cannot read'):
        load_scenario(str(tmp_path / 'nope.yaml'), CFG)
    p = tmp_path / 'bad.yaml'
    p.write_text('steps: [\n')
    with pytest.raises(ConfigError, match='not YAML'):
        load_scenario(str(p), CFG)


@pytest.mark.parametrize('params, says', [({}, 'scenario_file is required'),
                                          ({'scenario_file': 'a'}, 'frame_id is required'),
                                          ({'scenario_file': 'a', 'frame_id': 'f', 'after_end': 'loop'}, 'after_end'),
                                          ({'scenario_file': 'a', 'frame_id': 'f', 'reference_x_axis': [0, 0, 1]},
                                           'parallel'),
                                          ({'scenario_file': 'a', 'frame_id': 'f', 'seed': 1}, 'unknown')])
def test_a_bad_configuration_is_refused(params, says):
    with pytest.raises(ConfigError, match=says):
        ScriptedConfig.from_params(params)


def test_the_cursor_and_after_end():
    s1, s2 = Step(candidates=()), Step(candidates=(), unavailable=True)
    sc = Scenario([s1, s2], 'empty')
    assert [sc.next()[0] for _ in range(2)] == [0, 1]
    i, past = sc.next()
    assert i == 2 and past.candidates == () and not past.unavailable
    sc = Scenario([s1, s2], 'repeat_last')
    sc.next(), sc.next()
    assert sc.next()[1] is s2


def test_the_diagnostics_have_surface_graspS_minimal_shape():
    step = parse_scenario({'steps': [{'candidates': [C1]}]}, CFG)[0]
    d = diagnostics(0, step, 1, 'cam_optical', 3, 4)
    assert d['provider'] == 'tp.scripted' and d['stamp'] == {'sec': 3, 'nanosec': 4}
    assert d['summary'] == {'regions': 0, 'valid': 1, 'rejected': 0, 'rejections_by_code': {}, 'ineligible': 0,
                            'objects': 1, 'objects_without_depth': 0}
    assert d['objects'] == {'published': [{'id': 'scripted_object@0', 'position': [0.0, 0.0, 0.5]}],
                            'without_depth': [], 'skipped': []}
    assert d['candidates'][0]['class_id'] == 'capsule' and d['rejected'] == [] and d['ineligible'] == []
    assert d['scenario'] == {'step': 0, 'steps': 1, 'past_end': False}


# --- M4.5b: the objects PRESENT -----------------------------------------------------------------------------------

C2 = dict(C1, position=[0.1, 0.0, 0.5])


def test_objects_default_to_the_candidates_positions():
    steps = parse_scenario({'steps': [{'candidates': [C1, C2]}, {'candidates': []}]}, CFG)
    assert steps[0].objects is None and steps[0].present() == ((0.0, 0.0, 0.5), (0.1, 0.0, 0.5))
    assert steps[1].present() == ()                                  # no candidate, no object: an empty bin


def test_explicit_objects_remain_when_no_candidate_does():
    step = parse_scenario({'steps': [{'candidates': [], 'objects': [[0.02, 0.01, 0.5], [0, 0, 0.6]]}]}, CFG)[0]
    assert step.candidates == () and step.present() == ((0.02, 0.01, 0.5), (0.0, 0.0, 0.6))


def test_an_explicit_empty_list_presents_nothing_and_a_count_takes_the_first_candidates():
    s_empty, s_count = parse_scenario({'steps': [{'candidates': [C1], 'objects': []},
                                                 {'candidates': [C1, C2], 'objects': 1}]}, CFG)
    assert s_empty.present() == () and len(s_empty.candidates) == 1
    assert s_count.present() == ((0.0, 0.0, 0.5),)


@pytest.mark.parametrize('objects, says', [
    (3, "first 3 candidates, and it has 2"),
    ([[0, 0]], 'objects[0]: a position is three finite numbers'),
    ([[0, 0, float('nan')]], 'objects[0]: a position is three finite numbers'),
    ('many', 'objects must be a list of positions'),
    (True, 'objects must be a list of positions'),
])
def test_bad_objects_are_refused(objects, says):
    with pytest.raises(ConfigError) as e:
        parse_scenario({'steps': [{'candidates': [C1, C2], 'objects': objects}]}, CFG)
    assert says in str(e.value)


def test_an_unavailable_step_has_no_objects():
    with pytest.raises(ConfigError) as e:
        parse_scenario({'steps': [{'unavailable': True, 'objects': [[0, 0, 0.5]]}]}, CFG)
    assert 'an unavailable step publishes nothing' in str(e.value)


def test_past_the_end_empty_presents_nothing_and_repeat_last_repeats_the_objects():
    steps = parse_scenario({'steps': [{'candidates': [], 'objects': [[0, 0, 0.5]]}]}, CFG)
    sc = Scenario(steps, 'empty')
    sc.next()
    assert sc.next()[1].present() == ()
    sc = Scenario(steps, 'repeat_last')
    sc.next()
    assert sc.next()[1].present() == ((0.0, 0.0, 0.5),)
