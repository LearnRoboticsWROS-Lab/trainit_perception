"""The objects PRESENT (`grasp/objects.py`, vNext M4.5b PB-3): ROS-free, on exact ray-cast scenes.

Pinned: one object per region (whatever the geometry would say of its grasp), in region order, under the SAME id as
`propose` gives it; the position is the centroid of the valid depth points (the provider's band) back-projected
through the camera model -- a disc's true centre on a flat disc; the size is the robust metric extent; a region with
NO valid depth is placed on its pixel ray at the frame's median depth (`estimated`, size 0) and counted; when the
frame has no valid depth at all the result is UNMEASURABLE (fail closed: never an empty answer, the provider
publishes nothing); no eligible region is a genuine empty answer; a region without pixels is no object.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from surface_grasp_scenes import BASE, Disc, distorted, regions, render, tilted  # noqa: E402

from trainit_perception.grasp import Region, SurfaceGraspConfig, propose  # noqa: E402
from trainit_perception.grasp.objects import ESTIMATED, MEASURED, present_objects  # noqa: E402

CFG = SurfaceGraspConfig.from_params(dict(BASE))
CAM = distorted()
DISCS = [Disc((-0.07, 0.0, 0.5), (0, 0, -1), 0.021), Disc((0.07, 0.0, 0.5), tilted(30.0), 0.021),
         Disc((0.0, 0.07, 0.5), tilted(60.0, 'x'), 0.021)]


def _scene():
    depth, masks = render(DISCS, CAM)
    return depth, regions(DISCS, masks)


def test_every_region_is_present_whatever_its_grasp_verdict_with_the_providers_ids():
    depth, regs = _scene()
    grasp = propose(regs, depth, CAM, CFG)
    pres = present_objects(regs, depth, CAM, CFG.z_min_m, CFG.z_max_m)
    assert len(grasp.candidates) == 2 and [r.code for r in grasp.rejections] == ['APPROACH_OUTSIDE_CONE']
    assert len(pres.objects) == 3                                    # the 60 deg disc has no grasp, it IS there
    ids = {c.id for c in grasp.candidates} | {r.id for r in grasp.rejections}
    assert {o.id for o in pres.objects} == ids
    assert all(o.depth == MEASURED for o in pres.objects) and pres.without_depth == [] and pres.skipped == []
    assert pres.summary() == {'objects': 3, 'objects_without_depth': 0, 'objects_skipped': 0}


def test_the_position_is_the_depth_centroid_and_the_size_its_extent():
    depth, regs = _scene()
    o = present_objects(regs, depth, CAM, CFG.z_min_m, CFG.z_max_m).objects[0]
    assert np.linalg.norm(np.asarray(o.position) - DISCS[0].centre) < 1.5e-3   # a square-on disc: its centre
    sx, sy, sz = o.size
    assert 0.030 < sx < 0.042 and 0.030 < sy < 0.042 and sz < 1e-3          # 5..95 % of a 42 mm disc, flat
    assert o.class_id == DISCS[0].class_id and 0.0 <= o.score <= 1.0


def test_a_region_without_valid_depth_is_estimated_on_its_ray_and_counted():
    depth, regs = _scene()
    holed = depth.copy()
    holed[regs[1].mask] = 0.0                                        # the tilted disc: no depth at all
    pres = present_objects(regs, holed, CAM, CFG.z_min_m, CFG.z_max_m)
    assert len(pres.objects) == 3 and pres.without_depth == [pres.objects[1].id] and pres.skipped == []
    est = pres.objects[1]
    valid = holed[np.isfinite(holed) & (holed > CFG.z_min_m) & (holed < CFG.z_max_m)]
    assert est.depth == ESTIMATED and est.size == (0.0, 0.0, 0.0)
    assert abs(est.position[2] - float(np.median(valid))) < 1e-9
    ray = np.asarray(est.position) / np.linalg.norm(est.position)              # on its pixel ray
    true = np.asarray(DISCS[1].centre) / np.linalg.norm(DISCS[1].centre)
    assert float(ray @ true) > np.cos(np.radians(1.0))


@pytest.mark.parametrize('blind', [0.0, np.nan])
def test_without_any_valid_depth_the_frame_is_unmeasurable_never_an_empty_answer(blind):
    depth, regs = _scene()
    pres = present_objects(regs, np.full_like(depth, blind), CAM, CFG.z_min_m, CFG.z_max_m)
    assert pres.objects == [] and len(pres.without_depth) == 3 and pres.skipped == pres.without_depth
    assert pres.summary() == {'objects': 0, 'objects_without_depth': 3, 'objects_skipped': 3}
    assert pres.unmeasurable                                         # fail closed: the provider publishes nothing


def test_no_eligible_region_is_a_genuine_empty_answer_not_unmeasurable():
    depth, regs = _scene()
    for d in (depth, np.zeros_like(depth)):                         # even on a blind frame: nothing to measure
        pres = present_objects([], d, CAM, CFG.z_min_m, CFG.z_max_m)
        assert pres.objects == [] and pres.skipped == [] and not pres.unmeasurable
    empty = Region(class_id='part', score=0.5, mask=np.zeros_like(regs[0].mask))   # only a region without pixels
    assert not present_objects([empty], np.zeros_like(depth), CAM, CFG.z_min_m, CFG.z_max_m).unmeasurable


def test_a_measurable_frame_is_not_unmeasurable():
    depth, regs = _scene()
    holed = depth.copy()
    holed[regs[1].mask] = 0.0                                        # one region estimated, still measurable
    assert not present_objects(regs, depth, CAM, CFG.z_min_m, CFG.z_max_m).unmeasurable
    assert not present_objects(regs, holed, CAM, CFG.z_min_m, CFG.z_max_m).unmeasurable


def test_no_region_no_object_and_an_empty_region_is_no_object():
    depth, regs = _scene()
    assert present_objects([], depth, CAM, CFG.z_min_m, CFG.z_max_m).objects == []
    empty = Region(class_id='part', score=0.5, mask=np.zeros_like(regs[0].mask))
    pres = present_objects([empty, regs[0]], depth, CAM, CFG.z_min_m, CFG.z_max_m)
    assert len(pres.objects) == 1 and pres.without_depth == []


def test_the_depth_band_is_the_providers():
    depth, regs = _scene()
    pres = present_objects(regs[:1], depth, CAM, 0.6, 0.9)           # the disc (0.5 m), the background: outside
    assert pres.objects == [] and pres.without_depth == pres.skipped and len(pres.skipped) == 1
    assert pres.unmeasurable                                         # a wrong band: blind, never an empty bin
