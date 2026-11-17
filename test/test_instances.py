"""The `instance_topic` segmentation's pure mapping (vNext M4.5, PA45-2/3): an instance image -> regions.

Pinned: mask = image == k, the class and score are the segmenter's (the score becomes the detection term), the
regions come out in instance-id order; the allow-list, min_score, min_area_px and max_objects drop instances and
count why; an unlabelled pixel value is counted, never invented into a region; a duplicate id or a non-numeric
id breaks the contract; the parameters are validated by name.
"""
import numpy as np
import pytest

from trainit_perception.grasp import ConfigError
from trainit_perception.grasp.instances import (InstanceConfig, InstanceContractError, InstanceInfo,
                                                parse_instance_id, regions_from_instance_image)


def image():
    img = np.zeros((20, 30), np.uint16)
    img[2:6, 2:6] = 3            # 16 px
    img[10:12, 10:20] = 7        # 20 px
    img[15:16, 25:27] = 9        # 2 px
    img[0, 29] = 42              # a value instance_info does not label
    return img


INFOS = [InstanceInfo(7, 'capsule', 0.8), InstanceInfo(3, 'capsule', 0.95), InstanceInfo(9, 'cap_lid', 0.6)]


def cfg(**kw):
    return InstanceConfig.from_params(dict({'source': 'yolo'}, **kw))


def test_each_instance_becomes_a_region_with_its_mask_class_and_score():
    regions, report = regions_from_instance_image(image(), INFOS, cfg())
    assert [(r.class_id, r.score) for r in regions] == [('capsule', 0.95), ('capsule', 0.8), ('cap_lid', 0.6)]
    assert np.array_equal(regions[0].mask, image() == 3)
    assert regions[1].box_px == (10.0, 10.0, 20.0, 12.0)
    assert report == {'method': 'instance_topic', 'source': 'yolo', 'instances': 3, 'kept': 3,
                      'dropped_by': {'class': 0, 'score': 0, 'area': 0, 'max_objects': 0}, 'unlabelled': 1}


def test_the_filters_drop_and_count():
    _, rep = regions_from_instance_image(image(), INFOS, cfg(classes=['capsule']))
    assert rep['kept'] == 2 and rep['dropped_by']['class'] == 1
    _, rep = regions_from_instance_image(image(), INFOS, cfg(min_score=0.8))
    assert rep['kept'] == 2 and rep['dropped_by']['score'] == 1             # 0.8 is kept (inclusive)
    _, rep = regions_from_instance_image(image(), INFOS, cfg(min_area_px=3))
    assert rep['kept'] == 2 and rep['dropped_by']['area'] == 1
    regions, rep = regions_from_instance_image(image(), INFOS, cfg(max_objects=2))
    assert [r.score for r in regions] == [0.95, 0.8] and rep['dropped_by']['max_objects'] == 1


def test_an_info_whose_id_has_no_pixels_is_dropped_by_area():
    regions, rep = regions_from_instance_image(image(), INFOS + [InstanceInfo(11, 'capsule', 0.9)], cfg())
    assert len(regions) == 3 and rep['dropped_by']['area'] == 1


def test_a_non_finite_score_is_zero_and_a_score_is_clipped():
    regions, _ = regions_from_instance_image(image(), [InstanceInfo(3, 'capsule', float('nan')),
                                                       InstanceInfo(7, 'capsule', 1.5)], cfg())
    assert [r.score for r in regions] == [0.0, 1.0]


def test_the_contract_breaks_are_named():
    with pytest.raises(InstanceContractError, match='twice'):
        regions_from_instance_image(image(), INFOS + [InstanceInfo(3, 'x', 0.5)], cfg())
    with pytest.raises(InstanceContractError, match='one channel'):
        regions_from_instance_image(np.zeros((4, 4, 3), np.uint16), [], cfg())
    assert parse_instance_id(' 12 ') == 12
    for bad in ('0', 'a', '-3', '70000', '', '\u00b2', '1\u00b2', '\u0663', '\uff11', '+3', '3.0'):
        with pytest.raises(InstanceContractError):           # Unicode digits ('²', Arabic-Indic, full-width) too:
            parse_instance_id(bad)                           # never a ValueError escaping as a non-contract error


@pytest.mark.parametrize('params, says', [
    ({}, 'source is required'),
    ({'source': '/yolo'}, 'source is required'),
    ({'source': 'yolo', 'classes': 'capsule'}, 'classes must be a list'),
    ({'source': 'yolo', 'min_score': 1.2}, 'min_score'),
    ({'source': 'yolo', 'max_objects': 0}, 'max_objects'),
    ({'source': 'yolo', 'min_area_px': 2.5}, 'min_area_px'),
    ({'source': 'yolo', 'conf': 0.5}, 'unknown parameter'),
])
def test_a_bad_configuration_is_refused_by_name(params, says):
    with pytest.raises(ConfigError, match=says):
        InstanceConfig.from_params(params)


def test_the_topics_follow_the_source():
    c = cfg()
    assert (c.instances_topic, c.info_topic) == ('/perception/yolo/instances', '/perception/yolo/instance_info')
