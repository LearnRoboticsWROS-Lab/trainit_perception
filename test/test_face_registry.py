"""A FACE DECLARES EVERYTHING NEEDED TO CONFIGURE IT (ADR-0018 Decision 6).

The asymmetry this closes was verified: a hand-written `rect` face was ACCEPTED by this runtime
and REFUSED by the Setup Assistant, because the Assistant's ranking-parameter list was a fixed
four-tuple that never wrote `grasp_face` -- the key this package already read -- and its
computable-terms mirror hardcoded `expected_radius_m`. Keeping a second copy of the answer makes
every new face family a two-repo change, and the registry then buys nothing.

So these tests are written against the REGISTRY rather than against the shipped faces: they loop
over whatever is registered, including a face registered from outside the package, and check that
the three declarations are true of it. A new face passes them or it is not finished.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from ranking_scenes import K                                   # noqa: E402

from trainit_perception.ranking import (PROVIDES_FIT3D, PROVIDES_MASK_AREA,  # noqa: E402
                                        PROVIDES_SCORE, FACES, GraspFace, RULES, all_face_params,
                                        available_faces, computable_terms, face_enables_terms,
                                        face_extent_term_keys, face_params, face_schema,
                                        faces_schema, make_face, p90_extent, ranking_defaults,
                                        register_face)

SHIPPED = ('cylinder_side', 'disc', 'rect')
PLAUSIBLE = 0.020                      # a size every face parameter can take, in METRES


def configured(name):
    """The face, with every parameter it declares set to a plausible size."""
    return make_face(name, {key: PLAUSIBLE for key in face_params(name)})


# --- the three declarations, in one call ------------------------------------------------------

def test_face_schema_hands_over_all_three_declarations_and_nothing_else():
    assert face_schema('disc') == {
        'name': 'disc',
        'params': ('expected_radius_m',),
        'extent_term_keys': ('radius_ratio',),
        'enables_terms': ('visible_fraction',)}
    assert sorted(faces_schema()) == available_faces() == list(SHIPPED)


@pytest.mark.parametrize('name', SHIPPED)
def test_the_schema_agrees_with_the_class_and_with_the_single_accessors(name):
    schema = face_schema(name)
    cls = FACES[name]
    assert schema['name'] == name
    assert schema['params'] == tuple(cls.PARAMS) == face_params(name)
    assert schema['extent_term_keys'] == tuple(cls.extent_term_keys) == \
        face_extent_term_keys(name)
    assert schema['enables_terms'] == tuple(cls.ENABLES_TERMS) == face_enables_terms(name)
    # plain strings, so a configurator can serialise them with nothing imported from here
    for key in ('params', 'extent_term_keys', 'enables_terms'):
        assert all(isinstance(s, str) for s in schema[key]), key


@pytest.mark.parametrize('name', SHIPPED)
def test_every_parameter_a_face_DECLARES_is_a_parameter_it_READS(name):
    """A declared parameter that the face ignores is a form field that does nothing, and a read
    parameter that is not declared is a field a configurator never renders."""
    face = configured(name)
    assert face.has_known_area, f'{name} did not accept {PLAUSIBLE} m for each of its parameters'
    assert make_face(name, {}).has_known_area is False
    for key in face_params(name):
        # drop one at a time: each declared parameter must matter on its own
        partial = {k: PLAUSIBLE for k in face_params(name) if k != key}
        assert make_face(name, partial).has_known_area is False, f'{name} ignores {key}'
        assert key.endswith('_m'), 'the UNIT is part of the name (ADR-0018 Decision 4)'


@pytest.mark.parametrize('name', SHIPPED)
def test_extent_term_keys_is_EXACTLY_what_extent_terms_produces(name):
    """What `pose_quality` gates on. A key declared and not produced would silently drop a gate;
    a key produced and not declared would never be gated on at all."""
    face = configured(name)
    assert set(face.extent_terms((PLAUSIBLE, PLAUSIBLE))) == set(face.extent_term_keys)
    # and with no size, nothing is produced while the keys stay DECLARED -- which is what makes
    # the rule refusable instead of silently degenerate (ADR-0017 Decision 3)
    unset = make_face(name, {})
    assert unset.extent_terms((PLAUSIBLE, PLAUSIBLE)) == {}
    assert unset.expected_area_px(0.6, K) is None
    assert tuple(unset.extent_term_keys) == face_extent_term_keys(name)


@pytest.mark.parametrize('name', SHIPPED)
def test_enables_terms_names_terms_some_RULE_can_actually_use(name):
    """A face that unlocked a term no rule multiplies would be a dead end."""
    known = {t for rule in RULES.values() for t in rule.requires()}
    assert set(face_enables_terms(name)) <= known
    # and the declaration is what `computable_terms` actually acts on, for THIS face
    params = {'grasp_face': name, **{key: PLAUSIBLE for key in face_params(name)}}
    assert set(computable_terms(params, {PROVIDES_SCORE, PROVIDES_MASK_AREA})) - {'score'} == \
        set(face_enables_terms(name))
    unset = {'grasp_face': name}
    assert computable_terms(unset, {PROVIDES_SCORE, PROVIDES_MASK_AREA}) == {'score'}


@pytest.mark.parametrize('name', SHIPPED)
def test_the_face_refuses_a_depth_it_cannot_answer_at(name):
    face = configured(name)
    assert face.expected_area_px(0.6, K) > 0.0
    for z in (0.0, -0.3, float('nan'), float('inf')):
        assert face.expected_area_px(z, K) is None, z


# --- what a generator needs: the full parameter set, and `grasp_face` itself -------------------

def test_the_RANKING_PARAMETERS_come_from_the_registry_and_INCLUDE_grasp_face():
    """The key the fixed four-tuple never wrote, which is why a hand-written face was accepted by
    the runtime and refused by the generator."""
    for name in SHIPPED:
        params = ranking_defaults({PROVIDES_SCORE, PROVIDES_MASK_AREA}, grasp_face=name)
        assert params['grasp_face'] == name
        assert set(face_params(name)) <= set(params)
        assert all(params[key] == 0.0 for key in face_params(name)), '0 = unknown, refusably so'
    # the tool's three numbers appear only for a detector that runs a 3D fit
    fit3d = ranking_defaults({PROVIDES_SCORE, PROVIDES_MASK_AREA, PROVIDES_FIT3D})
    assert {'tool_radius_m', 'tool_reach_mm', 'tool_max_tilt_deg'} <= set(fit3d)
    assert 'candidate_rule' in fit3d


def test_the_PROFILE_documents_every_face_parameter_the_registry_declares():
    """The profile YAML is what a form is rendered from today, so a face whose parameters are
    missing from it is a face a configurator cannot fill in -- which is the refusal half of the
    asymmetry Decision 6 closes. The registry is the source; this check is the guard."""
    import yaml
    from trainit_perception.detectors import ColorMaskDetector
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'profiles', 'color_mask.yaml')
    with open(path) as fh:
        declared = set(yaml.safe_load(fh)['params'])
    assert 'grasp_face' in declared, 'the key the runtime reads and a fixed list never wrote'
    assert set(all_face_params()) <= declared
    # and nothing is documented that no face and no detector reads
    assert declared <= set(ColorMaskDetector.defaults()) | set(all_face_params())


def test_all_face_params_is_the_UNION_and_is_what_the_FIT_refuses():
    """One list, derived, so a face parameter handed to a fit by mistake is refused whatever face
    introduced it -- and a generator knows the full set of keys it may have to clear."""
    assert all_face_params() == tuple(sorted(
        {k for cls in FACES.values() for k in cls.PARAMS}))
    assert set(all_face_params()) >= {'expected_radius_m', 'expected_half_a_m',
                                      'expected_half_b_m', 'expected_height_m'}
    from trainit_perception.ranking import plane_ransac
    for key in all_face_params():
        with pytest.raises(TypeError) as e:
            plane_ransac(np.zeros((4, 4), bool), np.zeros((4, 4), np.float32), K, **{key: 0.02})
        assert key in str(e.value)


@pytest.mark.parametrize('accessor', [face_params, face_extent_term_keys, face_enables_terms,
                                      face_schema])
def test_every_accessor_refuses_an_unknown_name_and_SAYS_WHAT_IS_AVAILABLE(accessor):
    with pytest.raises(KeyError) as e:
        accessor('hexagon')
    assert 'disc' in str(e.value) and 'hexagon' in str(e.value)


# --- the point of a registry: a face from OUTSIDE the package is configurable too --------------

def test_A_FACE_FROM_OUTSIDE_DECLARES_ITS_OWN_TERMS_AND_NOTHING_HERE_IS_EDITED():
    """The whole of Decision 6 in one test: an outside face that unlocks a DIFFERENT term is
    believed, because `computable_terms` asks the face instead of naming `visible_fraction`."""

    class _AnnulusFace(GraspFace):
        name = 'test_annulus'
        PARAMS = ('expected_outer_radius_m', 'expected_inner_radius_m')
        extent_term_keys = ('annulus_ratio',)
        ENABLES_TERMS = ('visible_fraction', 'test_rim_width')

        def __init__(self, outer=0.0, inner=0.0):
            self.outer, self.inner = float(outer), float(inner)

        @classmethod
        def from_params(cls, params):
            return cls(params.get('expected_outer_radius_m', 0.0),
                       params.get('expected_inner_radius_m', 0.0))

        @property
        def has_known_area(self):
            return self.outer > self.inner > 0.0

        def half_extents(self, offsets, estimator):
            offsets = np.asarray(offsets, float)
            if offsets.size == 0:
                return (0.0, 0.0)
            r = estimator(np.hypot(offsets[:, 0], offsets[:, 1]))
            return (r, r)

        def expected_area_px(self, z_m, K_):
            if not self.has_known_area or not z_m > 0:
                return None
            f = 0.5 * (K_[0, 0] + K_[1, 1])
            return np.pi * ((f * self.outer / z_m) ** 2 - (f * self.inner / z_m) ** 2)

        def extent_terms(self, half_extents_m):
            if not self.has_known_area:
                return {}
            return {'annulus_ratio': 0.5 * sum(half_extents_m) / self.outer}

    try:
        register_face(_AnnulusFace)
        assert 'test_annulus' in available_faces() and 'test_annulus' in faces_schema()
        assert face_schema('test_annulus') == {
            'name': 'test_annulus',
            'params': ('expected_outer_radius_m', 'expected_inner_radius_m'),
            'extent_term_keys': ('annulus_ratio',),
            'enables_terms': ('visible_fraction', 'test_rim_width')}
        # the parameters reach `ranking_defaults`, so a generator writes them without being edited
        params = ranking_defaults({PROVIDES_SCORE, PROVIDES_MASK_AREA},
                                  grasp_face='test_annulus')
        assert params['grasp_face'] == 'test_annulus'
        assert 'expected_inner_radius_m' in params
        # and its OWN term is what becomes computable -- not a term named in this package
        configured_params = {'grasp_face': 'test_annulus', 'expected_outer_radius_m': 0.03,
                             'expected_inner_radius_m': 0.01}
        terms = computable_terms(configured_params, {PROVIDES_SCORE, PROVIDES_MASK_AREA})
        assert terms == {'score', 'visible_fraction', 'test_rim_width'}
        # the fit refuses its parameters as fit parameters, from the registry and not a list
        assert 'expected_inner_radius_m' in all_face_params()
        face = make_face('test_annulus', configured_params)
        assert face.half_extents(np.zeros((0, 2)), p90_extent) == (0.0, 0.0)
        assert face.expected_area_px(0.6, K) > 0.0
    finally:
        FACES.pop('test_annulus', None)


def test_register_face_refuses_a_face_with_no_name():
    class _Nameless(GraspFace):
        @classmethod
        def from_params(cls, params):
            return cls()

        @property
        def has_known_area(self):
            return False

        def half_extents(self, offsets, estimator):
            return (0.0, 0.0)

        def expected_area_px(self, z_m, K_):
            return None

        def extent_terms(self, half_extents_m):
            return {}

    with pytest.raises(ValueError):
        register_face(_Nameless)


# --- PARAM_SINCE: the pin a configurator needs ------------------------------------------------
def test_param_since_covers_every_ranking_parameter():
    """Every parameter `ranking_defaults` can emit must say which release understands it.

    This is the test that fails when a release adds a knob and forgets to date it -- which is the
    whole failure mode, because an undated parameter written into a bundle merges SILENTLY into an
    older runtime and does nothing.
    """
    from trainit_perception.ranking import PARAM_SINCE, all_face_params, ranking_defaults
    from trainit_perception.ranking.rules import (PROVIDES_BOX, PROVIDES_FIT3D, PROVIDES_MASK_AREA,
                                                  PROVIDES_SCORE)
    emitted = set(ranking_defaults((PROVIDES_SCORE, PROVIDES_BOX, PROVIDES_MASK_AREA,
                                    PROVIDES_FIT3D)))
    emitted |= set(all_face_params())
    missing = sorted(emitted - set(PARAM_SINCE))
    assert not missing, f'undated ranking parameters: {missing}'


def test_param_since_is_ordered_sensibly():
    from trainit_perception.ranking import param_since
    assert param_since('tool_radius_m') == '0.3.0'       # ADR-0017
    assert param_since('tool_max_tilt_deg') == '0.4.0'   # ADR-0018, the edge-on term
    assert param_since('nothing_like_this') == ''        # unknown is empty, never a guess
