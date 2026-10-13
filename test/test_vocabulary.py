"""The configuration vocabulary export: what crosses to the Setup Assistant (vNext M2.2, ADR-0021).

``trainit_perception.vocabulary.export()`` is what the Setup Assistant commits a snapshot of and
reads instead of this package. What this file holds it to:

1. the SCHEMA -- the exact keys at every level and closed kinds and types, so an incompatible change
   fails here before it reaches a configurator's reader;
2. the BOUNDARY (ADR-0021, Decision 9) -- every string an identifier or a version, numbers only in
   a parameter's schema and the face size band, no rule note or complement, no gate or tool
   default, nothing registered from outside the package;
3. DETERMINISM -- the same bytes under different hash seeds (``PROVIDES`` is a frozenset);
4. COVERAGE and ORDER -- every built-in method, rule and face, in the order its table declares
   it; a rule's terms are its factors in order, a face's parameters its ``PARAMS`` in order;
5. ``TERM_REQUIRES`` agrees with ``computable_terms`` over every configuration, partly sized faces
   included; ``DEFAULT_FACE`` with the runtime's ``'disc'`` literals;
6. the colour mask's ``PARAMETERS``, ``defaults()`` and the profile YAML agree; its noise-filter
   floors sit beside its defaults and the ranking's ``PARAM_SINCE`` stays ranking-only;
7. one version number; and ``consistency_problems`` names a broken record.
"""
import copy
import itertools
import os
import re
import subprocess
import sys

import pytest
import yaml

from trainit_perception import __version__, vocabulary
from trainit_perception.detectors import registry as dreg
from trainit_perception.detectors.color_mask import ColorMaskDetector
from trainit_perception.ranking import faces as F
from trainit_perception.ranking import ranker as RK
from trainit_perception.ranking import rules as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = ['schema_version', 'trainit_perception', 'provides', 'detector_methods', 'auto', 'rules',
       'default_preference', 'terms', 'faces', 'default_face', 'face_size_band_m', 'param_since']
TOKEN = re.compile(r'^[a-z][a-z0-9_]*$')
VERSION = re.compile(r'^\d+\.\d+\.\d+$')
PROVIDES_ALL = (R.PROVIDES_SCORE, R.PROVIDES_BOX, R.PROVIDES_MASK_AREA, R.PROVIDES_FIT3D)
TOOLS = (('tool_radius_m', 0.012), ('tool_reach_mm', 4.0), ('tool_max_tilt_deg', 30.0))


@pytest.fixture
def record():
    return vocabulary.export()


# --- 1. the schema ------------------------------------------------------------------------------

def test_the_record_is_schema_1_with_exactly_these_keys(record):
    assert list(record) == TOP
    assert record['schema_version'] == vocabulary.SCHEMA_VERSION == 1
    for name, method in record['detector_methods'].items():
        assert list(method) == ['kind', 'provides', 'parameters'], name
        assert method['kind'] in ('pure_python', 'external_node'), name
        if method['kind'] == 'external_node':
            assert method['provides'] is None and method['parameters'] is None, name
            continue
        for key, row in method['parameters'].items():
            assert {'type', 'default'} <= set(row) <= {'type', 'default', 'min', 'max',
                                                       'choices'}, (name, key)
            assert row['type'] in ('str', 'int', 'float', 'range'), (name, key)
    for name, req in record['terms'].items():
        assert list(req) == ['provides', 'positive_params', 'face_sized'], name
    for name, face in record['faces'].items():
        assert list(face) == ['params', 'enables_terms'], name


# --- 2. the boundary ----------------------------------------------------------------------------

def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path + (k,), k                  # every KEY is a leaf too: names cross, prose not
            yield from _leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, path + (i,))
    else:
        yield path, node


def _number_allowed(path):
    if path in (('schema_version',),) or path[:1] == ('face_size_band_m',):
        return True
    return (len(path) >= 5 and path[0] == 'detector_methods' and path[2] == 'parameters'
            and path[4] in ('default', 'min', 'max'))


def test_every_string_is_a_name_or_a_release_and_numbers_sit_only_in_a_schema(record):
    for path, value in _leaves(record):
        if isinstance(value, bool):
            assert path[0] == 'terms' and path[-1] == 'face_sized', path
        elif isinstance(value, (int, float)):
            assert _number_allowed(path), (path, value)
        elif isinstance(value, str):
            assert TOKEN.match(value) or VERSION.match(value), (path, value)
        else:
            assert value is None and path[0] == 'detector_methods', (path, value)


def test_no_prose_no_complement_and_no_gate_or_tool_default_crosses(record):
    keys = {k for path, k in _leaves(record) if isinstance(k, str) and path[-1] == k}
    assert not keys & {'note', 'notes', 'label', 'description', 'complement', 'value',
                       'extent_term_keys', 'weight'}
    for name, method in record['detector_methods'].items():
        for key in method['parameters'] or ():
            assert not re.match(r'^(pose_quality_|min_pose_quality|tool_)', key), (name, key)
    for rule in R.BUILTIN_RULES:
        assert record['rules'][rule.name] == list(rule.factors)        # names only, in order


def test_a_rule_or_face_registered_from_outside_is_not_exported(record):
    class _Outside(F.DiscFace):
        name = 'test_outside_face'

    try:
        R.register_rule(R.Rule('test_outside_rule', ('score',)))
        F.register_face(_Outside)
        assert vocabulary.export() == record
    finally:
        R.RULES.pop('test_outside_rule', None)
        F.FACES.pop('test_outside_face', None)


# --- 3. determinism -----------------------------------------------------------------------------

def test_the_record_is_the_same_bytes_under_any_hash_seed():
    code = ('import yaml, trainit_perception.vocabulary as v; '
            'print(yaml.safe_dump(v.export(), sort_keys=False, width=100), end="")')
    outs = set()
    for seed in ('0', '1', '4242'):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1


# --- 4. coverage and order ----------------------------------------------------------------------

def test_every_built_in_entry_is_exported_in_declared_order(record):
    assert list(record['detector_methods']) == [m for m, _, _ in dreg.BUILTIN_METHODS]
    assert [m for m, k, _ in dreg.BUILTIN_METHODS if k == 'pure_python'] == list(dreg._REGISTRY)
    assert [m for m, k, _ in dreg.BUILTIN_METHODS if k == 'external_node'] == list(
        dreg.EXTERNAL_METHODS)
    assert record['detector_methods']['color_mask']['provides'] == sorted(
        ColorMaskDetector.PROVIDES)
    assert list(record['rules']) == [r.name for r in R.BUILTIN_RULES] == list(R.RULES)
    assert record['auto'] == R.AUTO
    assert record['default_preference'] == list(R.DEFAULT_PREFERENCE)
    assert list(record['faces']) == [c.name for c in F.BUILTIN_FACES] == list(F.FACES)
    for cls in F.BUILTIN_FACES:
        assert record['faces'][cls.name] == {'params': list(cls.PARAMS),
                                             'enables_terms': list(cls.ENABLES_TERMS)}
    assert record['default_face'] == F.DEFAULT_FACE
    assert record['face_size_band_m'] == [F.MIN_FACE_M, F.MAX_FACE_M]
    assert set(record['provides']) == set(PROVIDES_ALL)
    assert list(record['terms']) == list(R.TERM_REQUIRES)
    assert {k: v for k, v in record['param_since'].items() if k in RK.PARAM_SINCE} == dict(
        RK.PARAM_SINCE)


# --- 5. the data beside the code agrees with the code -------------------------------------------

def _face_configs():
    yield {}
    yield {'expected_radius_m': 0.02}                         # an unset face is the default one
    for cls in F.BUILTIN_FACES:
        for k in range(len(cls.PARAMS) + 1):
            for subset in itertools.combinations(cls.PARAMS, k):
                yield dict({p: 0.02 for p in subset}, grasp_face=cls.name)


def _terms_from_the_record(record, params, provides):
    """What a configurator computes from the record alone -- the Setup Assistant's reading."""
    have, out = set(provides), set()
    face = record['faces'].get(str(params.get('grasp_face') or record['default_face']))
    for term, req in record['terms'].items():
        if not set(req['provides']) <= have:
            continue
        if not all(float(params.get(p, 0.0) or 0.0) > 0.0 for p in req['positive_params']):
            continue
        if req['face_sized'] and (face is None or term not in face['enables_terms'] or not all(
                float(params.get(p, 0.0) or 0.0) > 0.0 for p in face['params'])):
            continue
        out.add(term)
    return out


def test_term_requires_is_what_computable_terms_computes_in_every_configuration(record):
    checked = 0
    for k in range(len(PROVIDES_ALL) + 1):
        for provides in itertools.combinations(PROVIDES_ALL, k):
            for tools in itertools.product((0, 1), repeat=len(TOOLS)):
                for face in _face_configs():
                    params = dict(face)
                    params.update({n: v for (n, v), on in zip(TOOLS, tools) if on})
                    assert _terms_from_the_record(record, params, provides) == \
                        R.computable_terms(params, provides), (provides, params)
                    checked += 1
    assert checked == 16 * 8 * 12


def test_the_default_face_is_the_runtimes():
    mask = {R.PROVIDES_SCORE, R.PROVIDES_MASK_AREA}
    assert RK.ranking_defaults(mask)['grasp_face'] == F.DEFAULT_FACE
    assert RK.Ranker.from_params({}, provides=mask).face.name == F.DEFAULT_FACE
    sized = {p: 0.02 for p in F.face_params(F.DEFAULT_FACE)}
    assert 'visible_fraction' in R.computable_terms(sized, mask)       # no grasp_face key at all
    assert 'visible_fraction' not in R.computable_terms({}, mask)


# --- 6. the colour mask -------------------------------------------------------------------------

def test_the_colour_mask_schema_defaults_and_profile_agree():
    defaults = ColorMaskDetector.defaults()
    ranking = ColorMaskDetector.ranking_defaults()
    assert list(ColorMaskDetector.PARAMETERS) == [k for k in defaults if k not in ranking]
    with open(os.path.join(ROOT, 'profiles', 'color_mask.yaml')) as fh:
        profile = yaml.safe_load(fh)
    assert profile['method'] == ColorMaskDetector.method
    params = profile['params']
    assert {k: params[k]['default'] for k in defaults} == defaults
    assert set(params) - set(defaults) == set(F.all_face_params()) - set(
        F.face_params(F.DEFAULT_FACE))
    for key, schema in ColorMaskDetector.PARAMETERS.items():
        for field in ('type', 'min', 'max', 'choices'):
            assert params[key].get(field) == schema.get(field), (key, field)


def test_the_noise_filter_floors_sit_beside_the_colour_masks_defaults(record):
    dated = {k: s['since'] for k, s in ColorMaskDetector.PARAMETERS.items() if 'since' in s}
    assert dated == {'blur_px': '0.2.0', 'depth_min_m': '0.2.0', 'depth_max_m': '0.2.0'}
    assert not set(dated) & set(RK.PARAM_SINCE)         # the ranking's table stays ranking-only
    assert {k: record['param_since'][k] for k in dated} == dated
    assert list(record['param_since']) == sorted(record['param_since'])


# --- 7. one version, and a broken record says why -----------------------------------------------

def test_one_version_number():
    with open(os.path.join(ROOT, 'package.xml')) as fh:
        xml_version = re.search(r'<version>([^<]+)</version>', fh.read()).group(1)
    with open(os.path.join(ROOT, 'setup.py')) as fh:
        setup_version = re.search(r"version='([^']+)'", fh.read()).group(1)
    assert __version__ == xml_version == setup_version
    assert vocabulary.export()['trainit_perception'] == __version__


@pytest.mark.parametrize('break_it, says', [
    (lambda r: r['rules'].update(test_rule=['score', 'nope']), "multiplies 'nope'"),
    (lambda r: r['rules'].update(auto=[]), 'the token that means "resolve it"'),
    (lambda r: r['default_preference'].append('nope'), "names 'nope', which is not a rule"),
    (lambda r: r['faces']['disc']['enables_terms'].append('score'), 'not a face-sized term'),
    (lambda r: r['faces']['rect']['params'].append('expected_nothing_m'),
     "reads 'expected_nothing_m', which no release is dated with"),
    (lambda r: r.update(default_face='blob'), "'blob' is not a registered face"),
    (lambda r: r.update(face_size_band_m=[1.0, 0.5]), 'with 0 < min < max'),
    (lambda r: r['param_since'].update(blur_px='soon'), 'is not a parameter name and a release'),
    (lambda r: r['detector_methods']['color_mask']['provides'].append('normal'),
     "provides 'normal', which is not a declared output"),
], ids=['undeclared_term', 'auto_as_a_rule', 'preference', 'face_term', 'undated_face_param',
        'default_face', 'band', 'floor', 'method_output'])
def test_a_broken_record_is_named(record, break_it, says):
    broken = copy.deepcopy(record)
    break_it(broken)
    problems = vocabulary.consistency_problems(broken)
    assert any(says in p for p in problems), problems
    assert vocabulary.consistency_problems(record) == []


@pytest.mark.parametrize('breakage, says', [
    (lambda c, m: m.delattr(c, 'PARAMETERS'), 'declares no PARAMETERS schema'),
    (lambda c, m: m.setattr(c, 'PARAMETERS', tuple(c.PARAMETERS)), 'declares no PARAMETERS schema'),
    (lambda c, m: m.setitem(c.PARAMETERS, 'gamma', {'type': 'float'}),
     "parameter 'gamma' has no default, or no schema with a type"),
    (lambda c, m: m.setitem(c.PARAMETERS, 'h', None),
     "parameter 'h' has no default, or no schema with a type"),
    (lambda c, m: m.setattr(c, 'PROVIDES', None), 'declares no PROVIDES'),
    (lambda c, m: m.setattr(c, 'defaults', classmethod(lambda cls: {}['h'])),
     r"defaults\(\) fails \(KeyError"),
], ids=['no_schema', 'a_tuple', 'no_default', 'a_none_row', 'no_provides', 'defaults_fails'])
def test_a_method_whose_schema_is_broken_stops_the_export(monkeypatch, breakage, says):
    breakage(ColorMaskDetector, monkeypatch)
    with pytest.raises(ValueError, match=says):
        vocabulary.export()


def test_a_read_only_schema_is_still_a_schema(monkeypatch, record):
    from types import MappingProxyType
    monkeypatch.setattr(ColorMaskDetector, 'PARAMETERS',
                        MappingProxyType(dict(ColorMaskDetector.PARAMETERS)))
    assert vocabulary.export() == record


def test_a_size_band_of_another_shape_is_named(record):
    broken = copy.deepcopy(record)
    broken['face_size_band_m'] = [0.0001, 0.5, 1.0]
    assert any('is not [min, max]' in p for p in vocabulary.consistency_problems(broken))


def test_importing_the_vocabulary_module_imports_nothing_else_of_the_package():
    """The Community module carries no module-level import of the ranking (or of anything else):
    the methods and the ranking half are asked for inside export()."""
    code = ('import sys, trainit_perception.vocabulary; '
            'print(sorted(n for n in sys.modules if n.startswith("trainit_perception")))')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "['trainit_perception', 'trainit_perception.vocabulary']"


def test_a_floor_two_tables_disagree_on_stops_the_export(monkeypatch):
    monkeypatch.setitem(RK.PARAM_SINCE, 'blur_px', '0.1.0')
    with pytest.raises(ValueError, match=r"'blur_px' is dated 0\.2\.0 by its method and 0\.1\.0"):
        vocabulary.export()
