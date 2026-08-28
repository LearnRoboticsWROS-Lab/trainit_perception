"""Detectors by ``method`` name. Adding a method = adding a class here."""
from __future__ import annotations

from typing import Dict, Type

from .base import Detector
from .color_mask import ColorMaskDetector

_REGISTRY: Dict[str, Type[Detector]] = {
    ColorMaskDetector.method: ColorMaskDetector,
}


def available_methods():
    return sorted(_REGISTRY)


def make_detector(method: str, params: dict | None = None) -> Detector:
    if method not in _REGISTRY:
        raise KeyError(f'unknown detector method {method!r} (available: {", ".join(available_methods())})')
    return _REGISTRY[method](params)
