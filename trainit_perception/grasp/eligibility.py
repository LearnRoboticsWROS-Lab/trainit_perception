"""Workspace ELIGIBILITY: which segmented regions may enter a grasp provider at all (vNext M4.5, PA45-4).

ROS-free and pure. A cell's workspace (a bin's opening, a tray, a conveyor window) is stated as a POLYGON in the
colour image's pixels plus an optional depth band; a region whose mask lies mostly outside the polygon, or whose
median depth lies outside the band, is INELIGIBLE: it never enters `propose`, and the provider's diagnostics list
it with the code `OUTSIDE_WORKSPACE` beside -- never among -- the geometry rejections (ADR-0031 Decision 3 keeps
its 15 codes). The provider's geometry never learns about bins, and no coordinate lives in TP code.

perception.yaml (an entry key of a `surface_grasp` provider, beside `segmentation`):

    eligibility:
      roi_px: [[u, v], [u, v], [u, v], ...]   # REQUIRED: >= 3 vertices, a simple polygon, in colour pixels
      min_inside: 0.9                         # the share of the region's mask inside the polygon, (0, 1]
      z_min_m: 0.30                           # optional: the depth band (observation frame, metres) the
      z_max_m: 0.90                           #   region's MEDIAN depth must lie in (inclusive)

Pixel semantics: a pixel (u, v) is inside when its CENTRE (u, v) is inside the polygon or ON its boundary; a
region is eligible when `inside_fraction >= min_inside` (inclusive). A region with no pixels, or the z band on a
region with no valid depth, is left to the provider (its `REGION_EMPTY` / `INSUFFICIENT_DEPTH` are geometry
codes, not workspace ones).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .surface_grasp import ConfigError, Region, region_id

#: the diagnostics code of a region kept out of the provider by the workspace filter
OUTSIDE_WORKSPACE = 'OUTSIDE_WORKSPACE'

#: the `eligibility` block's vocabulary (exported under `grasp_providers.<name>.eligibility`). `min_` and the
#: polygon cross WITHOUT a default: they are the cell's numbers, never the framework's.
PARAMETERS: Dict[str, Dict[str, object]] = {
    'roi_px': {'type': 'polygon_px', 'required': True, 'since': '0.7.0'},
    'min_inside': {'type': 'float', 'min': 0.0, 'max': 1.0, 'since': '0.7.0'},
    'z_min_m': {'type': 'float', 'min': 0.0, 'max': 20.0, 'since': '0.7.0'},
    'z_max_m': {'type': 'float', 'min': 0.0, 'max': 20.0, 'since': '0.7.0'},
}

_ON_EDGE_PX = 1e-9


def defaults() -> Dict[str, object]:
    """Every parameter with its default. The default depth band is the schema's whole range: no depth filter."""
    return {'roi_px': [], 'min_inside': 0.9, 'z_min_m': 0.0, 'z_max_m': 20.0}


def _num(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _segments_cross(p1, p2, q1, q2) -> bool:
    """Whether the closed segments p1p2 and q1q2 share a point."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
                and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on(p1, p2, q1)) or (o2 == 0 and on(p1, p2, q2))
            or (o3 == 0 and on(q1, q2, p1)) or (o4 == 0 and on(q1, q2, p2)))


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _spike(pts) -> Optional[int]:
    """The first vertex where the two ADJACENT edges meeting at it are collinear and fold back onto each other
    (a zero-width spike), or None. The non-adjacent edge check does not look at adjacent pairs."""
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(_cross(a, b, c)) < 1e-12 and dot < 0:
            return i
    return None


def polygon_problems(roi, width: Optional[int] = None, height: Optional[int] = None) -> List[str]:
    """Why `roi` is not a usable workspace polygon, as sentences (empty = it is). With the image size, every
    vertex must also lie on the image: [0, width] x [0, height] pixels."""
    if not isinstance(roi, (list, tuple)) or len(roi) < 3:
        return [f'roi_px must be a list of >= 3 [u, v] vertices, got {roi!r}']
    pts = []
    for i, p in enumerate(roi):
        if not isinstance(p, (list, tuple)) or len(p) != 2 or _num(p[0]) is None or _num(p[1]) is None:
            return [f'roi_px vertex {i} must be two finite numbers [u, v], got {p!r}']
        pts.append((float(p[0]), float(p[1])))
    out = []
    n = len(pts)
    coincide = next((i for i in range(n) if pts[i] == pts[(i + 1) % n]), None)
    collinear = all(abs(_cross(pts[0], pts[1], p)) < 1e-12 for p in pts[2:])
    spike = None if collinear else _spike(pts)
    if coincide is not None:
        out.append(f'roi_px vertices {coincide} and {(coincide + 1) % n} coincide')
    elif spike is not None:
        out.append(f'roi_px vertex {spike} is a spike: edges {(spike - 1) % n} and {spike} fold back onto each other '
                   f'(the polygon must be simple)')
    else:
        crossing = next(((i, j) for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)
                         and _segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n])), None)
        if crossing is not None:
            out.append(f'roi_px edges {crossing[0]} and {crossing[1]} intersect: the polygon must be simple')
        else:
            area = abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))) / 2
            if area < 1.0:
                out.append(f'roi_px encloses {area:g} px^2: a degenerate polygon (collinear vertices) admits no '
                           f'region')
    if width is not None and height is not None:
        bad = [i for i, (u, v) in enumerate(pts) if not (0.0 <= u <= width and 0.0 <= v <= height)]
        if bad:
            out.append(f'roi_px vertex(es) {bad} lie outside the {width}x{height} colour image')
    return out


@dataclass(frozen=True)
class EligibilityConfig:
    roi_px: Tuple[Tuple[float, float], ...]
    min_inside: float
    z_min_m: float
    z_max_m: float
    _cache: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_params(cls, params: Mapping, width: Optional[int] = None,
                    height: Optional[int] = None) -> 'EligibilityConfig':
        """Merge onto `defaults()` and VALIDATE; raises `ConfigError` naming every problem."""
        if not isinstance(params, Mapping):
            raise ConfigError(f'eligibility refused: a mapping {{roi_px, min_inside, z_min_m, z_max_m}} is '
                              f'expected, got {params!r}')
        problems: List[str] = []
        unknown = sorted(set(params) - set(PARAMETERS))
        if unknown:
            problems.append(f'unknown key(s) {", ".join(unknown)} (have: {", ".join(PARAMETERS)})')
        p = dict(defaults())
        p.update(params)
        if 'roi_px' not in params:
            problems.append('roi_px is required: the workspace polygon in colour pixels (no default: a workspace '
                            'is the cell\'s)')
        else:
            problems += polygon_problems(p['roi_px'], width, height)
        mi, zmin, zmax = _num(p['min_inside']), _num(p['z_min_m']), _num(p['z_max_m'])
        if mi is None or not (0.0 < mi <= 1.0):
            problems.append(f'min_inside must be in (0, 1], got {p["min_inside"]!r}')
        if zmin is None or zmax is None or not (0.0 <= zmin < zmax <= 20.0):
            problems.append(f'the depth band [z_min_m, z_max_m] = [{p["z_min_m"]!r}, {p["z_max_m"]!r}] must satisfy '
                            f'0 <= z_min_m < z_max_m <= 20')
        if problems:
            raise ConfigError('eligibility refused: ' + '; '.join(problems))
        return cls(roi_px=tuple((float(u), float(v)) for u, v in p['roi_px']), min_inside=mi, z_min_m=zmin,
                   z_max_m=zmax)

    def image_problems(self, width: int, height: int) -> List[str]:
        """The polygon against the camera's resolution, once it is known (the CameraInfo)."""
        return polygon_problems([list(p) for p in self.roi_px], width, height)

    def inside_mask(self, height: int, width: int) -> np.ndarray:
        """Pixel centres inside the polygon or on its boundary (cached per image size)."""
        key = (int(height), int(width))
        if key not in self._cache:
            self._cache[key] = polygon_mask(self.roi_px, height, width)
        return self._cache[key]


def polygon_mask(roi: Sequence[Tuple[float, float]], height: int, width: int) -> np.ndarray:
    """A boolean (height, width) image: True where the pixel centre (u, v) is inside `roi` or on its boundary
    (even-odd rule for the interior, an exact point-to-segment distance for the boundary)."""
    v, u = np.mgrid[0:height, 0:width]
    u = u.astype(float)
    v = v.astype(float)
    inside = np.zeros((height, width), bool)
    on_edge = np.zeros((height, width), bool)
    pts = [(float(a), float(b)) for a, b in roi]
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        if y2 != y1:                                     # a horizontal edge crosses no scanline
            crosses = (y1 > v) != (y2 > v)
            x_at = x1 + (v - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (u < x_at)
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = np.clip(((u - x1) * dx + (v - y1) * dy) / L2, 0.0, 1.0) if L2 > 0 else np.zeros_like(u)
        d2 = (u - (x1 + t * dx)) ** 2 + (v - (y1 + t * dy)) ** 2
        on_edge |= d2 <= _ON_EDGE_PX ** 2
    return inside | on_edge


def _median_depth(mask: np.ndarray, depth_m: Optional[np.ndarray]) -> Optional[float]:
    if depth_m is None:
        return None
    z = np.asarray(depth_m, float)
    if z.ndim == 3 and z.shape[2] == 1:
        z = z[:, :, 0]
    vals = z[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if vals.size else None


def split_eligible(regions: Sequence[Region], depth_m: Optional[np.ndarray],
                   cfg: Optional[EligibilityConfig]) -> Tuple[List[Region], List[Dict[str, object]]]:
    """Regions -> (the eligible ones, in their order; the ineligible ones as diagnostics records
    `{id, class_id, code: OUTSIDE_WORKSPACE, why, inside_fraction}`). `cfg` None = everything is eligible.
    Ids are the provider's region ids (`<class>@<u>,<v>`, a collision suffixed `#k` in region order)."""
    if cfg is None:
        return list(regions), []
    eligible: List[Region] = []
    ineligible: List[Dict[str, object]] = []
    seen: Dict[str, int] = {}
    for r in regions:
        mask = np.asarray(r.mask, bool)
        rid, _ = region_id(str(r.class_id), mask)
        k = seen.get(rid, 0)
        seen[rid] = k + 1
        rid = rid if k == 0 else f'{rid}#{k}'
        n = int(mask.sum())
        if n == 0:
            eligible.append(r)                           # REGION_EMPTY is the provider's geometry code
            continue
        inside = cfg.inside_mask(*mask.shape[:2])
        frac = float(np.count_nonzero(mask & inside)) / n
        if frac < cfg.min_inside:
            ineligible.append({'id': rid, 'class_id': str(r.class_id), 'code': OUTSIDE_WORKSPACE,
                               'why': f'{frac:.3f} of its {n} px inside the workspace polygon < min_inside '
                                      f'{cfg.min_inside:g}',
                               'inside_fraction': frac})
            continue
        z = _median_depth(mask, depth_m)
        if z is not None and not (cfg.z_min_m <= z <= cfg.z_max_m):
            ineligible.append({'id': rid, 'class_id': str(r.class_id), 'code': OUTSIDE_WORKSPACE,
                               'why': f'median depth {z:.3f} m outside the workspace band [{cfg.z_min_m:g}, '
                                      f'{cfg.z_max_m:g}] m',
                               'inside_fraction': frac})
            continue
        eligible.append(r)
    return eligible, ineligible
