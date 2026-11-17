"""Workspace eligibility (vNext M4.5, PA45-4): a pixel polygon + a depth band, applied BEFORE the provider.

Pinned: a region inside is eligible; one with EXACTLY `min_inside` of its pixels inside is eligible (inclusive),
one pixel fewer is not; a region outside is listed OUTSIDE_WORKSPACE with its inside fraction and never reaches
`propose`; a pixel ON the polygon's edge counts as inside; the depth band uses the region's median depth; a
polygon that is short, degenerate, self-intersecting or off the image is refused by name; the 15 geometry codes
are untouched.
"""
import json

import numpy as np
import pytest

from trainit_perception.grasp import CHECK_ORDER, ConfigError, Region, region_id
from trainit_perception.grasp import surface_grasp as SG
from trainit_perception.grasp.eligibility import (OUTSIDE_WORKSPACE, EligibilityConfig, polygon_mask,
                                                  polygon_problems, split_eligible)

H, W = 40, 60
SQUARE = [[10, 10], [30, 10], [30, 30], [10, 30]]          # pixel centres 10..30 inclusive


def region(cells, class_id='part', score=0.9):
    m = np.zeros((H, W), bool)
    for v, u in cells:
        m[v, u] = True
    return Region(class_id=class_id, score=score, mask=m)


def block(v0, v1, u0, u1):
    return [(v, u) for v in range(v0, v1) for u in range(u0, u1)]


def cfg(**kw):
    return EligibilityConfig.from_params(dict({'roi_px': SQUARE, 'min_inside': 0.9}, **kw))


def test_a_region_inside_is_eligible_and_keeps_its_order():
    a, b = region(block(12, 15, 12, 15)), region(block(20, 25, 20, 25), class_id='other')
    ok, out = split_eligible([a, b], None, cfg())
    assert ok == [a, b] and out == []


def test_exactly_min_inside_is_eligible_and_one_pixel_less_is_not():
    # 10 pixels in one row; the polygon's right edge is u = 30 (inclusive): u 21..30 inside, 31.. outside
    nine_in = region([(20, u) for u in range(22, 32)])            # u 22..30 inside (9), 31 outside (1)
    eight_in = region([(20, u) for u in range(23, 33)])           # 8 inside, 2 outside
    ok, out = split_eligible([nine_in, eight_in], None, cfg(min_inside=0.9))
    assert ok == [nine_in]
    assert len(out) == 1 and out[0]['code'] == OUTSIDE_WORKSPACE
    assert out[0]['inside_fraction'] == pytest.approx(0.8)
    assert set(out[0]) == {'id', 'class_id', 'code', 'why', 'inside_fraction'}


def test_a_pixel_on_the_polygon_edge_is_inside():
    inside = polygon_mask([(10, 10), (30, 10), (30, 30), (10, 30)], H, W)
    assert inside[10, 10] and inside[30, 30] and inside[10, 20] and inside[20, 30]
    assert not inside[9, 20] and not inside[20, 31]
    tri = polygon_mask([(0, 0), (20, 0), (0, 20)], H, W)             # a diagonal edge u + v = 20
    assert tri[10, 10] and tri[0, 20] and not tri[11, 10]


def test_a_region_outside_is_listed_and_never_enters_propose():
    far = region(block(32, 38, 40, 50))
    ok, out = split_eligible([far], None, cfg())
    assert ok == []
    assert out == [{'id': out[0]['id'], 'class_id': 'part', 'code': OUTSIDE_WORKSPACE, 'why': out[0]['why'],
                    'inside_fraction': 0.0}]
    assert out[0]['id'].startswith('part@') and 'min_inside' in out[0]['why']


def test_the_depth_band_uses_the_median_depth():
    depth = np.full((H, W), 0.5, np.float32)
    near = region(block(12, 15, 12, 15))
    depth[12:15, 12:15] = 0.2
    depth[12, 12] = np.nan                                         # invalid depth is ignored by the median
    deep = region(block(20, 25, 20, 25))
    ok, out = split_eligible([near, deep], depth, cfg(z_min_m=0.3, z_max_m=0.6))
    assert ok == [deep]
    assert out[0]['code'] == OUTSIDE_WORKSPACE and 'median depth 0.200' in out[0]['why']
    # inclusive bounds; and a region without valid depth is left to the provider (INSUFFICIENT_DEPTH)
    ok, _ = split_eligible([deep], depth, cfg(z_min_m=0.5, z_max_m=0.5000001))
    assert ok == [deep]
    depth[20:25, 20:25] = 0.0
    ok, _ = split_eligible([deep], depth, cfg(z_min_m=0.3, z_max_m=0.4))
    assert ok == [deep]


def test_no_eligibility_and_an_empty_region_pass_through():
    a, empty = region(block(32, 38, 40, 50)), region([])
    assert split_eligible([a], None, None) == ([a], [])
    ok, out = split_eligible([empty], None, cfg())
    assert ok == [empty] and out == []                             # REGION_EMPTY stays the provider's code


@pytest.mark.parametrize('params, says', [
    ({}, 'roi_px is required'),
    ({'roi_px': [[0, 0], [10, 0]]}, '>= 3'),
    ({'roi_px': [[0, 0], [10, 0], ['a', 3]]}, 'vertex 2'),
    ({'roi_px': [[0, 0], [10, 0], [20, 0]]}, 'degenerate'),
    ({'roi_px': [[0, 0], [10, 10], [10, 0], [0, 10]]}, 'must be simple'),
    ({'roi_px': []}, '>= 3'),
    ({'roi_px': [[0, 0]]}, '>= 3'),
    ({'roi_px': [[0, 0], [10, 0], [0, 0]]}, 'coincide'),
    ({'roi_px': [[0, 0], [20, 0], [10, 0], [10, 20]]}, 'spike'),                   # adjacent edges 0, 1 overlap
    ({'roi_px': [[0, 0], [20, 0], [20, 20], [20, 30], [20, 25], [0, 20]]}, 'spike'),  # a zero-width spike
    ({'roi_px': [[0, 0], [20, 0], [30, 0], [25, 0], [20, 20]]}, 'spike'),
    ({'roi_px': [[0, 0], [0, 0], [0, 0]]}, 'coincide'),
    ({'roi_px': SQUARE, 'min_inside': 0.0}, 'min_inside must be in (0, 1]'),
    ({'roi_px': SQUARE, 'min_inside': 1.5}, 'min_inside must be in (0, 1]'),
    ({'roi_px': SQUARE, 'z_min_m': 0.8, 'z_max_m': 0.4}, 'depth band'),
    ({'roi_px': SQUARE, 'colour': 1}, 'unknown key'),
])
def test_a_bad_eligibility_is_refused_by_name(params, says):
    with pytest.raises(ConfigError, match=says.replace('(', r'\(').replace(']', r'\]')):
        EligibilityConfig.from_params(params)


def test_the_polygon_must_lie_on_the_image_once_its_size_is_known():
    c = cfg()
    assert c.image_problems(W, H) == []
    assert 'outside the 20x20' in c.image_problems(20, 20)[0]
    assert polygon_problems([[0, 0], [W, 0], [W, H], [0, H]], W, H) == []
    with pytest.raises(ConfigError, match='outside'):
        EligibilityConfig.from_params({'roi_px': [[0, 0], [70, 0], [70, 10]]}, W, H)


def test_the_geometry_codes_are_unchanged_and_the_diagnostics_list_the_ineligible():
    # 15 geometry codes through 0.7.0; 0.8.0 (M4.5d, ADR-0031 amendment) adds FACE_FIT_FAILED (known_shape_fit)
    assert len(CHECK_ORDER) == 16 and OUTSIDE_WORKSPACE not in CHECK_ORDER and 'FACE_FIT_FAILED' in CHECK_ORDER
    res = SG.GraspResult(candidates=[], rejections=[], n_regions=0)
    rec = {'id': 'part@45,35', 'class_id': 'part', 'code': OUTSIDE_WORKSPACE, 'why': 'x', 'inside_fraction': 0.0}
    d = json.loads(json.dumps(SG.diagnostics(res, 'cam', 1, 2, ineligible=[rec],
                                             segmentation={'method': 'color_mask'}), allow_nan=False))
    assert d['summary']['ineligible'] == 1 and d['ineligible'] == [rec] and d['rejected'] == []
    assert d['segmentation'] == {'method': 'color_mask'}
    assert SG.diagnostics(res, 'cam', 1, 2)['ineligible'] == []


def test_a_collinear_spike_is_named_and_a_straight_collinear_vertex_is_not_one():
    from trainit_perception.grasp.eligibility import _spike
    assert _spike([(0, 0), (20, 0), (10, 0), (10, 20)]) == 1           # edges 0 and 1 fold back at vertex 1
    assert _spike([(0, 0), (10, 0), (20, 0), (20, 20), (0, 20)]) is None    # a redundant vertex on a straight edge
    assert polygon_problems([[0, 0], [10, 0], [20, 0], [20, 20], [0, 20]]) == []


def test_min_inside_one_needs_every_pixel_inside_and_zero_is_refused():
    whole = region(block(12, 15, 12, 15))                              # 9 px, all inside
    edge = region([(20, u) for u in range(21, 31)])                    # u 21..30: on the right edge, inside
    one_out = region([(20, u) for u in range(22, 32)])                 # u 31 lies outside: 0.9 < 1.0
    ok, out = split_eligible([whole, edge, one_out], None, cfg(min_inside=1.0))
    assert ok == [whole, edge]
    assert len(out) == 1 and out[0]['inside_fraction'] == pytest.approx(0.9)
    for refused in (0, 0.0, -0.1, float('nan'), True):
        with pytest.raises(ConfigError, match=r'min_inside must be in \(0, 1\]'):
            cfg(min_inside=refused)


def test_the_ineligible_id_is_the_providers_region_id():
    far = region(block(32, 38, 40, 50))
    _, out = split_eligible([far, far], None, cfg())
    rid, (u, v) = region_id('part', far.mask)
    assert rid == f'part@{u},{v}' and [e['id'] for e in out] == [rid, rid + '#1']
    assert region_id('part', np.zeros((H, W), bool)) == ('part@empty', (-1, -1))
