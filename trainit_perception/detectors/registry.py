"""Detectors by ``method`` name. Adding a method = adding a class here.

Two KINDS of method exist (D-015):
- ``pure-python`` — a Detector subclass in this registry. Pure function, no ROS: the
  TSA live tuner and the runtime node run the SAME ``detect()``. OpenCV-level 2D work
  lives here.
- ``external-node`` — a method implemented by a separate node (C++/PCL, a learned
  model, a vendor SDK) that publishes the SAME contract
  (``vision_msgs/Detection3DArray`` on ``/perception/<name>/detections``). It has no
  entry in ``_REGISTRY``; ``EXTERNAL_METHODS`` names it so configurators can list it,
  and the consumer (TMR's ``DetectObject``) never knows the difference. This is the
  predisposition for PCL 3D processing without putting C++ behind ``make_detector``.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import Detector
from .color_mask import ColorMaskDetector

_REGISTRY: Dict[str, Type[Detector]] = {
    ColorMaskDetector.method: ColorMaskDetector,
}

# method name -> short description; reserved for nodes outside this package.
EXTERNAL_METHODS: Dict[str, str] = {
    'pcl_cluster': 'C++ PCL pointcloud clustering node (roadmap)',
}


def available_methods():
    return sorted(_REGISTRY)


def make_detector(method: str, params: dict | None = None) -> Detector:
    if method not in _REGISTRY:
        raise KeyError(f'unknown detector method {method!r} (available: {", ".join(available_methods())})')
    return _REGISTRY[method](params)
