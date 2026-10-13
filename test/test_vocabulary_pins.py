"""TP's configuration vocabulary as TP 0.4.0 answers it -- pinned BEFORE vNext M2.2 adds the export.

ADR-0021, amendment M2.2: TP registers its configuration vocabulary once, in its built-in tables,
and ``trainit_perception.vocabulary.export()`` reads them for the Setup Assistant's committed
snapshot. The export adds DATA beside the code -- what each term needs beside ``computable_terms``,
the default face beside its ``'disc'`` literals, the colour mask's parameter schema beside its
defaults -- and must not change one answer the runtime gives. This file freezes those answers:
written and run on 0.4.0 before the export existed, and not edited by the implementation that
follows.

What is pinned (``test/data/vocabulary_pins_m2_2.txt``):

- the detector registry: the in-process methods, the external ones, the colour mask's ``PROVIDES``
  and ``defaults()``;
- the rules: every name in declaration order with its factors in order, ``AUTO``,
  ``DEFAULT_PREFERENCE``;
- the faces: ``faces_schema()``, the size band, ``all_face_params()``; ``PARAM_SINCE``;
- ``computable_terms`` and ``default_rule`` over EVERY combination a configurator can produce: each
  subset of the four ``PROVIDES`` tokens x each of the tool's three numbers set or not x the face
  (unset, and every subset of every registered face's parameters, and a face nobody registers).

Regenerate the table only deliberately, never because this test failed::

    python3 test/test_vocabulary_pins.py > test/data/vocabulary_pins_m2_2.txt
"""
import itertools
import json
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                    'vocabulary_pins_m2_2.txt')

# What a configurator can set, at values inside every band: a face size of 2 cm, a tool of
# 12 mm radius with a 4 mm grip window, a 30 degree tilt limit.
SIZE_M = 0.02
TOOLS = (('tool_radius_m', 0.012), ('tool_reach_mm', 4.0), ('tool_max_tilt_deg', 30.0))
UNKNOWN_FACE = 'blob'


def _faces_built_in():
    """The faces 0.4.0 ships, in declaration order (a test elsewhere may register one for a moment:
    it is removed in a finally, and this table is about what ships)."""
    from trainit_perception.ranking import faces as F
    return [name for name in F.FACES if name in ('disc', 'rect', 'cylinder_side')]


def _face_configs():
    from trainit_perception.ranking import faces as F
    yield '-', {}
    yield '-', {'expected_radius_m': SIZE_M}            # an unset face is the default one
    for name in _faces_built_in():
        params = F.FACES[name].PARAMS
        for k in range(len(params) + 1):
            for subset in itertools.combinations(params, k):
                yield name, {p: SIZE_M for p in subset}
    yield UNKNOWN_FACE, {}


def table():
    """Every line of the pin, as TP answers it now."""
    from trainit_perception.detectors import registry as dreg
    from trainit_perception.detectors.color_mask import ColorMaskDetector
    from trainit_perception.ranking import faces as F
    from trainit_perception.ranking import ranker as RK
    from trainit_perception.ranking import rules as R

    out = ['[registry]',
           f'in_process {json.dumps(dreg.available_methods())}',
           f'external {json.dumps(sorted(dreg.EXTERNAL_METHODS))}',
           f'color_mask.PROVIDES {json.dumps(sorted(ColorMaskDetector.PROVIDES))}',
           f'color_mask.defaults {json.dumps(ColorMaskDetector.defaults())}',
           '[rules]',
           f'AUTO {R.AUTO}',
           f'DEFAULT_PREFERENCE {json.dumps(list(R.DEFAULT_PREFERENCE))}']
    builtin_rules = [n for n in R.RULES if not n.startswith('test_')]
    out += [f'rule {name} {json.dumps(list(R.RULES[name].factors))}' for name in builtin_rules]
    out += ['[faces]',
            f'band {json.dumps([F.MIN_FACE_M, F.MAX_FACE_M])}',
            f'all_face_params {json.dumps(list(F.all_face_params()))}']
    for name in _faces_built_in():
        s = F.face_schema(name)
        out.append(f'face {name} params={json.dumps(list(s["params"]))} '
                   f'extent={json.dumps(list(s["extent_term_keys"]))} '
                   f'enables={json.dumps(list(s["enables_terms"]))}')
    out += ['[param_since]']
    out += [f'{k} {v}' for k, v in sorted(RK.PARAM_SINCE.items())]
    out += ['[computable]']
    provides_all = (R.PROVIDES_SCORE, R.PROVIDES_BOX, R.PROVIDES_MASK_AREA, R.PROVIDES_FIT3D)
    for k in range(len(provides_all) + 1):
        for provides in itertools.combinations(provides_all, k):
            for tools in itertools.product((0, 1), repeat=len(TOOLS)):
                for face, sized in _face_configs():
                    params = dict(sized)
                    if face != '-':
                        params['grasp_face'] = face
                    params.update({name: value for (name, value), on in zip(TOOLS, tools) if on})
                    key = (f'{"+".join(sorted(provides)) or "-"} | '
                           f'tools={"".join(map(str, tools))} | face={face} '
                           f'{",".join(sorted(sized)) or "-"}')
                    try:
                        terms = R.computable_terms(params, provides)
                        answer = (f'terms={",".join(sorted(terms))} | '
                                  f'default={R.default_rule(computable=terms)}')
                    except Exception as exc:                      # noqa: BLE001
                        answer = f'raises {type(exc).__name__}'
                    out.append(f'{key} | {answer}')
    return out


def test_tp_answers_exactly_what_it_answered_before_the_export():
    with open(DATA) as fh:
        want = fh.read().splitlines()
    got = table()
    if got != want:
        changed = [f'  pinned: {w}\n  now:    {g}' for w, g in zip(want, got) if w != g][:20]
        extra = len(got) - len(want)
        raise AssertionError(
            'TP\'s configuration answers moved (vNext M2.2 must add data, not change answers):\n'
            + '\n'.join(changed)
            + (f'\n  ... and {extra:+d} lines' if extra else ''))


if __name__ == '__main__':                                       # pragma: no cover
    sys.stdout.write('\n'.join(table()) + '\n')
