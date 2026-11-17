"""The `instance_topic` segmentation: an EXTERNAL segmenter's instance image -> provider regions (vNext M4.5).

ROS-free and pure. Any segmenter (a YOLO-seg node, Mask R-CNN, SAM, a vendor SDK) publishes, per colour frame:

    /perception/<source>/instances      sensor_msgs/Image, 16UC1 (or mono16): 0 = background, k = instance k;
                                        the colour image's stamp and frame; registered to colour, same size
    /perception/<source>/instance_info  vision_msgs/Detection2DArray, the SAME stamp: per instance `id` = str(k),
                                        results[0].hypothesis.class_id / .score, bbox

(D0's `INSTANCE_MASK` representation: a compact-id image plus a class per id.) This module maps one such pair
onto `Region`s: the mask is `image == k`, the class and score are the segmenter's, and the score becomes the
provider's `detection` term. The ROS side (pairing by stamp, the checks on size / frame / encoding) is the node's.

perception.yaml:

    segmentation:
      method: instance_topic
      params: {source: yolo, classes: [capsule], min_score: 0.5, max_objects: 20, min_area_px: 150}
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .surface_grasp import ConfigError, Region

METHOD = 'instance_topic'
#: the instance image encodings accepted (16-bit unsigned, one channel)
ENCODINGS = ('16UC1', 'mono16')

#: the method's vocabulary (exported under `grasp_providers.<name>.segmentation_parameters.instance_topic`)
PARAMETERS: Dict[str, Dict[str, object]] = {
    'source': {'type': 'str', 'required': True, 'since': '0.7.0'},
    'classes': {'type': 'str_list', 'since': '0.7.0'},
    'min_score': {'type': 'float', 'min': 0.0, 'max': 1.0, 'since': '0.7.0'},
    'max_objects': {'type': 'int', 'min': 1, 'max': 1000, 'since': '0.7.0'},
    'min_area_px': {'type': 'int', 'min': 1, 'max': 100000000, 'since': '0.7.0'},
}

_ROS_TOKEN = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')


def defaults() -> Dict[str, object]:
    """Every parameter with its default: every class (an empty allow-list), every score, 20 regions, 1 pixel."""
    return {'source': '', 'classes': [], 'min_score': 0.0, 'max_objects': 20, 'min_area_px': 1}


@dataclass(frozen=True)
class InstanceConfig:
    source: str
    classes: Tuple[str, ...]
    min_score: float
    max_objects: int
    min_area_px: int

    @property
    def instances_topic(self) -> str:
        return f'/perception/{self.source}/instances'

    @property
    def info_topic(self) -> str:
        return f'/perception/{self.source}/instance_info'

    @classmethod
    def from_params(cls, params: Mapping) -> 'InstanceConfig':
        """Merge onto `defaults()` and VALIDATE; raises `ConfigError` naming every problem."""
        params = params or {}
        problems: List[str] = []
        unknown = sorted(set(params) - set(PARAMETERS))
        if unknown:
            problems.append(f'unknown parameter(s) {", ".join(unknown)} (have: {", ".join(PARAMETERS)})')
        p = dict(defaults())
        p.update(params)
        source = p['source']
        if not isinstance(source, str) or not _ROS_TOKEN.match(source):
            problems.append(f'source is required: the segmenter\'s name, one ROS name token (it publishes '
                            f'/perception/<source>/instances), got {source!r}')
        classes = p['classes']
        if classes is None:
            classes = []
        if not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) and c for c in classes):
            problems.append(f'classes must be a list of class names (empty = every class), got {classes!r}')
            classes = []
        ms = p['min_score']
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(float(ms)) \
                or not 0.0 <= float(ms) <= 1.0:
            problems.append(f'min_score must be in [0, 1], got {ms!r}')
        for key in ('max_objects', 'min_area_px'):
            v = p[key]
            lo, hi = PARAMETERS[key]['min'], PARAMETERS[key]['max']
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= int(v) <= hi:
                problems.append(f'{key} must be a whole number in [{lo}, {hi}], got {v!r}')
        if problems:
            raise ConfigError('instance_topic segmentation refused: ' + '; '.join(problems))
        return cls(source=source, classes=tuple(classes), min_score=float(ms), max_objects=int(p['max_objects']),
                   min_area_px=int(p['min_area_px']))


@dataclass(frozen=True)
class InstanceInfo:
    """One segmenter instance: the pixel value k, its class, its score."""
    instance_id: int
    class_id: str
    score: float


class InstanceContractError(ValueError):
    """An instance image / info pair that breaks the instance contract (the frame is not processed)."""


def parse_instance_id(text: str) -> int:
    """`Detection2D.id` -> k. The contract: the decimal pixel value, 1..65535."""
    s = str(text).strip()
    # ASCII decimal digits only: str.isdigit() also accepts e.g. '²', which int() then refuses with a ValueError
    if not (s.isascii() and s.isdecimal()) or not 1 <= int(s) <= 65535:
        raise InstanceContractError(f'instance id {text!r} is not a pixel value 1..65535 (0 is the background)')
    return int(s)


def regions_from_instance_image(image: np.ndarray, infos: Sequence[InstanceInfo],
                                cfg: InstanceConfig) -> Tuple[List[Region], Dict[str, object]]:
    """An instance image + its per-instance labels -> (regions in instance-id order, a report).

    Kept: an instance whose class is allowed, whose score >= `min_score`, whose mask has >= `min_area_px` pixels;
    then the `max_objects` best by (score desc, area desc, id asc). The report counts every instance dropped and
    why (`class`, `score`, `area`, `max_objects`) and the pixel values the info does not label (`unlabelled`).
    Raises `InstanceContractError` on a duplicate id or an image that is not one 2-D channel."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim != 2:
        raise InstanceContractError(f'the instance image must be one channel, got shape {img.shape}')
    ids = [i.instance_id for i in infos]
    dup = sorted({k for k in ids if ids.count(k) > 1})
    if dup:
        raise InstanceContractError(f'instance id(s) {dup} appear twice in one instance_info: ids are unique per '
                                    f'frame')
    present, counts = np.unique(img, return_counts=True)
    area = {int(k): int(c) for k, c in zip(present, counts) if k != 0}
    dropped = {'class': 0, 'score': 0, 'area': 0, 'max_objects': 0}
    keep: List[Tuple[InstanceInfo, int, float]] = []
    allowed = set(cfg.classes)
    for info in infos:
        a = area.get(info.instance_id, 0)
        score = float(info.score) if math.isfinite(float(info.score)) else 0.0
        if allowed and info.class_id not in allowed:
            dropped['class'] += 1
        elif score < cfg.min_score:
            dropped['score'] += 1
        elif a < cfg.min_area_px:
            dropped['area'] += 1
        else:
            keep.append((info, a, score))
    keep.sort(key=lambda t: (-t[2], -t[1], t[0].instance_id))
    dropped['max_objects'] = max(0, len(keep) - cfg.max_objects)
    keep = sorted(keep[:cfg.max_objects], key=lambda t: t[0].instance_id)
    regions: List[Region] = []
    for info, _, score in keep:
        mask = img == info.instance_id
        v, u = np.nonzero(mask)
        box = (float(u.min()), float(v.min()), float(u.max() + 1), float(v.max() + 1))
        regions.append(Region(class_id=info.class_id, score=float(np.clip(score, 0.0, 1.0)), mask=mask,
                              box_px=box))
    labelled = set(ids)
    report = {'method': METHOD, 'source': cfg.source, 'instances': len(infos), 'kept': len(regions),
              'dropped_by': dropped, 'unlabelled': len([k for k in area if k not in labelled])}
    return regions, report
