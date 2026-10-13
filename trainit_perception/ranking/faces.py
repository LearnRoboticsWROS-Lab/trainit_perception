"""The GRASP FACE: the one pluggable thing in the ranking, and it is not the object.

Of the functions that make up a candidate ranking, exactly four lines carry a shape assumption --
"the thing being grasped presents a disc of known radius" -- plus one field that baked it into the
output type (ADR-0017 Decision 2):

    visible_fraction's denominator  pi * (f * r / z)^2        -> `expected_area_px`
    centroid's MAD radial spread                              -> `half_extents`
    plane_ransac's hypot-collapsed extent                     -> `half_extents`
    pose_quality's radius_ratio gate                          -> `extent_terms` / `extent_term_keys`
    GraspFit.radius_m, ONE scalar where the plane fit already -> `GraspFit.half_extents_m`
    computed two half-extents and threw one away

Everything else -- `box_overlap`, `rank_order`, `check_rule`, `approach_obstruction` (parameterised
by the TOOL, not the object), `approach_clearance`, the method registry -- is generic as written.
So the plug is here and nowhere else, and it is named after the FACE rather than the object
because a bottle picked from above is a disc of the CAP's radius and not of the body's. The same
bottle gripped round the waist is a different face of the same object.

A face answers three questions and nothing more:

    half_extents(offsets, estimator)  given each surface point's offset from the grasp point IN
                                      THE PLANE of the fit, how big is the face -- as two
                                      half-extents along the fit's own principal axes. A disc
                                      collapses them radially, so both come back equal; a
                                      rectangular face would measure each axis separately. WHICH
                                      robust estimator to use is the fit method's choice, not the
                                      face's, so it is passed in.
    expected_area_px(z, K)            how many pixels the face should cover at that depth if
                                      nothing were on it. The denominator of `visible_fraction`.
    extent_terms(half_extents)        the fit's size against the size the face is CONFIGURED to
                                      have, as named terms. `pose_quality` gates on these, so
                                      the names are declared in `extent_term_keys`.

A face with no configured size answers `None` / `{}`: it is honest about knowing nothing, which
is what makes a rule that needs `visible_fraction` REFUSABLE rather than silently degenerate
(ADR-0017 Decision 3).

Registry shape: the same one `fit3d.METHODS` and `detectors/registry.py` already use -- a name in
a configuration file, a class here, and `register_face` for a face that lives outside this
package.

A FACE DECLARES EVERYTHING A CONFIGURATOR NEEDS TO CONFIGURE IT (ADR-0018 Decision 6), because the
alternative was measured and it is a two-repo change per face: a hand-written `rect` face was
ACCEPTED by this runtime and REFUSED by the Setup Assistant, whose parameter list never wrote
`grasp_face` and whose computable-terms mirror hardcoded `expected_radius_m`. So three class
attributes are the whole contract, and `face_schema(name)` hands all three over in one call (a
configurator reads the first and the last through `trainit_perception.vocabulary.export()` and
its own committed snapshot of it, never by importing this package while it generates -- ADR-0021):

    PARAMS             the parameter keys the face reads -- what a form renders, and what a
                       generator must write into the detector's params
    extent_term_keys   the quality terms `extent_terms` produces -- what `pose_quality` gates on
    ENABLES_TERMS      the candidate TERMS that become computable once PARAMS are set -- what
                       decides which `candidate_rule` is offerable (`rules.computable_terms`)

THE UNIT IS PART OF THE NAME, and that is not a style rule (ADR-0018 Decision 4). This workspace
mixes metres and millimetres, STL carries no unit metadata, and the error does not raise: the
survey's own run reported a 51 m^2 face and a 4042 mm radius without a word. Every face parameter
therefore ends in `_m`, and a face refuses a size it cannot believe rather than writing an absurd
denominator into a bundle.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Mapping, Optional, Tuple, Type

import numpy as np

Estimator = Callable[[np.ndarray], float]


# --- the two robust estimators the shipped fit methods ask for ------------------------------
# They live here, beside the faces that apply them, so a new face gets the same two for free.

def mad_extent(values: np.ndarray) -> float:
    """Median absolute deviation, scaled to a standard deviation (1.4826). What `centroid` uses:
    a median survives the holes a depth image is full of, and the scale factor makes the number
    comparable with a real extent."""
    return float(np.median(np.abs(values)) * 1.4826)


def p90_extent(values: np.ndarray) -> float:
    """The 90th percentile. What `plane_ransac` uses: the outer edge of the surface, with the few
    pixels a mask always leaks onto the neighbour behind discarded."""
    return float(np.percentile(np.abs(values), 90))


# --- the unit guard (ADR-0018 Decision 4) ----------------------------------------------------
# A grasp face is something a tool can hold. Nothing a tool holds is a metre across and nothing
# is a tenth of a millimetre, so a number outside this band is not a small face -- it is a
# MILLIMETRE value written into a metres field, which is the error that does not raise: the
# survey's own run reported a 4042 mm radius and a 51 m^2 face without a word. Caught here it
# stops the node inside `make_detector`; uncaught it makes `visible_fraction` a constant 0 for
# every candidate, and a rule whose factor is 0 everywhere publishes ties -- which is the
# detector's own order wearing the name of a ranking.
MIN_FACE_M = 1e-4
MAX_FACE_M = 1.0


def face_size_m(params: Mapping, key: str, *, what: str = 'face') -> float:
    """Read one face size out of a flat params dict. 0 / missing = UNKNOWN; absurd = refused."""
    raw = params.get(key, 0.0)
    if raw is None or raw == '':
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{key} = {raw!r} is not a number of METRES, so the {what} has no size '
                         f'a term could divide by') from None
    if value == 0.0:
        return 0.0                              # the honest "unknown", and a refusable rule
    if value < 0.0:
        raise ValueError(f'{key} = {value!r} is negative, and a grasp face has no negative '
                         f'size. 0 is how a size is left UNKNOWN -- which makes the rule that '
                         f'needs it refusable rather than silently wrong (ADR-0017 Decision 3).')
    if not math.isfinite(value) or not (MIN_FACE_M <= value <= MAX_FACE_M):
        raise ValueError(
            f'{key} = {value!r} is not a plausible grasp {what} in METRES (expected between '
            f'{MIN_FACE_M} and {MAX_FACE_M}). A grasp face is something a tool can hold; this '
            f'reads like MILLIMETRES written into a metres field, which is the one unit error '
            f'that does not raise -- it just makes visible_fraction a constant and the published '
            f'order stop being a ranking (ADR-0018 Decision 4). Leave it at 0 for "unknown".')
    return value


def _focal_px(face: 'GraspFace', z_m: float, K: np.ndarray) -> Optional[float]:
    """The mean focal length, or None when this face cannot answer an area at this depth.

    One place, so every face refuses the same three things the same way: an unknown size, a
    non-positive depth and a NaN depth (a hole in the depth image reaching this far).
    """
    if not face.has_known_area or not (z_m > 0.0) or not math.isfinite(z_m):
        return None
    return 0.5 * (float(K[0, 0]) + float(K[1, 1]))


class GraspFace(ABC):
    """The shape the tool meets. Subclasses are registered by name below.

    The three class attributes are the whole configuration contract (ADR-0018 Decision 6): a
    configurator reads the parameters and the terms a face unlocks from this package's exported
    vocabulary (`vocabulary.export()`, ADR-0021) -- never the extent terms, which stay here --
    rather than keeping a hand-written copy: a new face family is this class plus, on the
    configurator's side, a catalog refresh, not a second implementation.
    """

    name: str = ''
    #: the parameter keys this face reads out of the detector's flat params dict
    PARAMS: Tuple[str, ...] = ()
    #: the quality terms `extent_terms` produces -- what `pose_quality` gates on
    extent_term_keys: Tuple[str, ...] = ()
    #: the candidate TERMS that become computable once `PARAMS` are set. Every face that can
    #: answer `expected_area_px` unlocks `visible_fraction`, which is the one term whose
    #: denominator IS the face -- so this is the default, and a face that measured something
    #: else would declare something else. `rules.computable_terms` reads it instead of
    #: hardcoding a term name, and so does the Setup Assistant (ADR-0018 Decision 6).
    ENABLES_TERMS: Tuple[str, ...] = ('visible_fraction',)

    @classmethod
    @abstractmethod
    def from_params(cls, params: Mapping) -> 'GraspFace':
        """Build the face from the detector's parameters. Missing size = unknown size."""

    @abstractmethod
    def half_extents(self, offsets: np.ndarray, estimator: Estimator) -> Tuple[float, float]:
        """(N, 2) in-plane offsets from the grasp point -> two half-extents, in metres."""

    @abstractmethod
    def expected_area_px(self, z_m: float, K: np.ndarray) -> Optional[float]:
        """Pixels the whole face should cover at depth ``z_m``. None when the size is unknown."""

    @abstractmethod
    def extent_terms(self, half_extents_m: Tuple[float, float]) -> Dict[str, float]:
        """The measured extent against the configured one. Empty when the size is unknown."""

    @property
    @abstractmethod
    def has_known_area(self) -> bool:
        """True when `expected_area_px` can answer -- i.e. when `visible_fraction` is computable."""


class DiscFace(GraspFace):
    """A flat circular face of a configured radius: a capsule's cap, a bottle's cap, a tin's lid.

    The first face, and the one every measured number in ADR-0014/0017 was taken with. Its
    radius is NOT a property of the object as a whole -- it is the radius of the face the tool
    comes down on, which is why the parameter is read off the grasp target's own shape.
    """

    name = 'disc'
    PARAMS = ('expected_radius_m',)
    extent_term_keys = ('radius_ratio',)

    def __init__(self, radius_m: float = 0.0):
        self.radius_m = float(radius_m)

    @classmethod
    def from_params(cls, params: Mapping) -> 'DiscFace':
        return cls(radius_m=face_size_m(params, 'expected_radius_m', what='face radius'))

    def __repr__(self) -> str:                                      # pragma: no cover - debug aid
        return (f'DiscFace(radius_m={self.radius_m})' if self.radius_m > 0
                else 'DiscFace(radius unknown)')

    @property
    def has_known_area(self) -> bool:
        return self.radius_m > 0.0

    def half_extents(self, offsets: np.ndarray, estimator: Estimator) -> Tuple[float, float]:
        """RADIALLY, because a disc has one extent and it is the same in every direction.

        This is the line that `plane_ransac` used to hold itself (`np.hypot(a, b)` collapsed into
        one scalar). It is here now, so a face that is NOT round measures its axes separately
        without the fit being touched.
        """
        offsets = np.asarray(offsets, float)
        if offsets.size == 0:
            return (0.0, 0.0)
        r = estimator(np.hypot(offsets[:, 0], offsets[:, 1]))
        return (r, r)

    def expected_area_px(self, z_m: float, K: np.ndarray) -> Optional[float]:
        """pi * (f * r / z)^2 -- the disc square to the lens.

        NO cos(tilt) term, and that is a measurement and not a simplification: see
        `terms.visible_fraction`, whose denominator this is.
        """
        f = _focal_px(self, z_m, K)
        if f is None:
            return None
        r_px = f * self.radius_m / z_m
        return math.pi * r_px * r_px

    def extent_terms(self, half_extents_m: Tuple[float, float]) -> Dict[str, float]:
        """`radius_ratio`: the fitted radius over the configured one. 1.0 is a perfect agreement.

        This is what catches a mask that leaked onto the bin's wall and fitted a 78 mm "capsule":
        `pose_quality` gates on |ratio - 1|, and an unset radius must therefore make the rule
        REFUSABLE rather than quietly drop the gate (ADR-0017 Decision 3).
        """
        if not self.has_known_area:
            return {}
        a, b = half_extents_m
        return {'radius_ratio': 0.5 * (float(a) + float(b)) / self.radius_m}


class RectFace(GraspFace):
    """A flat RECTANGULAR face of two configured half-extents: a label panel, a carton's end, a
    machined pad, the flat of a lid that is not round.

    The second face, and the one the extension point was measured on: written against this ABC
    and registered, it needed ZERO changes to `fit3d`, `terms`, `ranker` or `rules` -- its extent
    ratios flow into the fit's quality dict and gate `pose_quality` automatically, and its
    `expected_area_px` is the projected rectangle (ADR-0018, Context 2).

    `a` AND `b` ARE INTERCHANGEABLE, deliberately. A plane fit's two in-plane axes come out of an
    SVD and carry no labels: the first is merely the direction of most spread, which for a
    partly occluded face is not necessarily the face's own long axis. So `extent_terms` compares
    the LARGER measured half-extent with the LARGER configured one, and entering 15 and 10 mm or
    10 and 15 mm gives the same answer. The alternative -- trusting the fit's axis order -- is a
    silent mis-measurement on exactly the candidates that are hardest to see.

    MEASURED on an analytic raster (`test_rect_face`), 60 x 40 mm at 600 mm:

      * the projected area is `(2a*f/z) * (2b*f/z)` and does NOT depend on the rotation about the
        face's own normal -- 1,000 of the analytic area at 30 deg of roll, 1,002 at 73 deg, and
        1,04 in the axis-aligned case where every edge pixel falls in or out together;
      * `plane_ransac`'s p90 recovers 0,90 of each true half-extent (0,896 and 0,920), because
        the 90th percentile of a uniform fill is 0,9 of its half-width. So a correct rectangle
        reads an extent ratio of about 0,90, well inside the 0,25 gate tolerance.
      * `centroid`'s MAD recovers only 0,73 to 0,77 (the MAD of a uniform fill is 0,7408 of its
        half-width, against 1,048 of a DISC's radius -- the estimators are calibrated on the
        SHAPE, and which shape it is is the face's business). So `pose_quality` reads 0,64 on a
        correct rectangle through `plane_ransac` and 0,12 to 0,25 through `centroid`: a
        rectangular face wants the plane fit, and `pose_quality_radius_tol` is the configured
        knob that says how much slack to allow (ADR-0017 Decision 4).
      * `centroid` hands over the IMAGE axes rather than the face's, so the measured pair can
        arrive in either order: on this raster it read (14,69, 23,08) mm for a face configured
        30 x 20. That is not a bug to fix in the fit -- it is why the comparison sorts.
    """

    name = 'rect'
    PARAMS = ('expected_half_a_m', 'expected_half_b_m')
    extent_term_keys = ('extent_a_ratio', 'extent_b_ratio')

    def __init__(self, half_a_m: float = 0.0, half_b_m: float = 0.0):
        self.half_a_m = float(half_a_m)
        self.half_b_m = float(half_b_m)

    @classmethod
    def from_params(cls, params: Mapping) -> 'RectFace':
        return cls(half_a_m=face_size_m(params, 'expected_half_a_m', what='face half-extent'),
                   half_b_m=face_size_m(params, 'expected_half_b_m', what='face half-extent'))

    def __repr__(self) -> str:                                      # pragma: no cover - debug aid
        return (f'RectFace(half_a_m={self.half_a_m}, half_b_m={self.half_b_m})'
                if self.has_known_area else 'RectFace(size unknown)')

    @property
    def has_known_area(self) -> bool:
        """BOTH half-extents, because one side of a rectangle is not a size. A face with only one
        of the two set is as unknown as a face with neither -- and loudly so, since the rule that
        needs `visible_fraction` is then refused at start-up."""
        return self.half_a_m > 0.0 and self.half_b_m > 0.0

    def half_extents(self, offsets: np.ndarray, estimator: Estimator) -> Tuple[float, float]:
        """PER AXIS, which is the whole difference from a disc: each in-plane axis separately.

        The axes are the fit's own (`plane_ransac` hands over the two principal in-plane
        directions), and they are NOT sorted here -- `GraspFit.half_extents_m` means "along the
        fit's axes" and re-ordering it would make that false. The sorting happens in
        `extent_terms`, where the comparison is, because that is where the ambiguity bites.
        """
        offsets = np.asarray(offsets, float)
        if offsets.size == 0:
            return (0.0, 0.0)
        return (float(estimator(offsets[:, 0])), float(estimator(offsets[:, 1])))

    def expected_area_px(self, z_m: float, K: np.ndarray) -> Optional[float]:
        """(2a * f / z) * (2b * f / z) -- the rectangle square to the lens.

        Invariant to rotation about the face's own normal, because a rotation preserves area;
        NOT corrected for tilt, for the same measured reason the disc is not (ADR-0018
        Decision 5: the tilt is the `edge_on` TERM, never a correction folded in here).
        """
        f = _focal_px(self, z_m, K)
        if f is None:
            return None
        return (2.0 * self.half_a_m * f / z_m) * (2.0 * self.half_b_m * f / z_m)

    def extent_terms(self, half_extents_m: Tuple[float, float]) -> Dict[str, float]:
        """TWO ratios, the larger measured against the larger configured.

        `pose_quality` gates on each of them, so a rectangle that fitted as a square, or a mask
        that leaked along one axis onto its neighbour, is caught on the axis it happened on --
        which a single radius could not see, because it averages the two.
        """
        if not self.has_known_area:
            return {}
        meas = sorted((abs(float(half_extents_m[0])), abs(float(half_extents_m[1]))), reverse=True)
        want = sorted((self.half_a_m, self.half_b_m), reverse=True)
        return {'extent_a_ratio': meas[0] / want[0], 'extent_b_ratio': meas[1] / want[1]}


class CylinderSideFace(GraspFace):
    """The SIDE of a cylinder: the face a bottle picked from the side presents to the tool.

    It is NOT a `rect` face with different numbers, even though the two area formulas coincide,
    and the difference is the reason this class exists. Three things separate them, all measured
    on an analytic ray/cylinder raster (`test_cylinder_side_face`):

    1. WHAT THE AREA IS INVARIANT TO. A cylinder's silhouette is `2r` wide seen from ANY
       direction perpendicular to its axis, so the area survives the one rotation a bottle in a
       crate is guaranteed to have -- the roll about its own axis. A rectangle's does not: roll a
       rectangular face about the view axis and the area holds, roll it about its own long axis
       and the face disappears.
    2. WHAT A PLANE FIT CAN MEASURE ON IT: not the radius. RANSAC keeps the band where the curved
       surface stays inside its tolerance, whose half-width is about `sqrt(2 * tol * r)` -- 11,40
       mm for a 32,5 mm bottle at the shipped 2 mm tolerance. MEASURED on the raster: the fit
       recovers (26,86, 12,29) mm of half-extent for a 60 mm-long band of a 32,5 mm-radius
       cylinder, so the transverse "radius" reads 0,38 of the truth. A `rect` face configured
       with the honest numbers (32,5 x 30 mm) therefore scores that bottle `pose_quality`
       EXACTLY 0.000 -- a refusal -- on a perfectly good candidate.
    3. THE DEPTH. A disc and a rectangle are flat, so the depth the fit reports IS the face's
       depth. A cylinder spans `r` in depth, and the area formula is exact only as `r/z -> 0`.
       MEASURED against the fit's own reported depth, a fully visible silhouette reads 0,958 to
       0,975 across `r/z` from 0,008 to 0,054 -- a few per cent LOW, which is the safe direction:
       the term never saturates at 1,00 on a face that is in fact partly covered. (Fed the AXIS
       depth instead it would read 1,07 and clip, which is the saturation failure ADR-0017
       Decision 6 exists to catch.)

    SO IT SHIPS WITH NO EXTENT TERMS, AND THAT IS THE HONEST HALF. `extent_term_keys` is empty:
    the transverse extent a plane fit reports is a property of the RANSAC tolerance rather than
    of the face, and which of the fit's two principal axes is the cylinder's axis is not knowable
    without a cylinder FIT (the two swap over as soon as the band is wider than it is long). What
    is missing is a `cylinder_ransac` method in `fit3d.METHODS`, which would return the radius and
    the axis and make both extent terms real; ADR-0018 names a cylinder face as an obvious next
    registry entry and does not take that decision. Until then this face answers the question it
    can answer -- how many pixels the silhouette should cover -- and declines the one it cannot.

    The fit says so too, and that is the point of not hiding it: a plane through a cylinder reads
    `inlier_ratio` 0,45 and `pose_quality` 0,18 on that same 32,5 mm bottle. It is telling you,
    correctly, that it is fitting a plane to something that is not one.

    `expected_height_m` IS THE BAND THE MASK COVERS, not the bottle. The same rule the disc has
    always had -- the face is the thing the detector's mask is of -- only it is easier to get
    wrong here: a segmenter that returns the whole body wants the body's height, one that returns
    a label band wants the band's.
    """

    name = 'cylinder_side'
    PARAMS = ('expected_radius_m', 'expected_height_m')
    #: NONE, deliberately. See the class docstring: a plane fit measures the RANSAC band, not the
    #: cylinder, so an extent ratio here would refuse good candidates with a straight face.
    extent_term_keys = ()

    def __init__(self, radius_m: float = 0.0, height_m: float = 0.0):
        self.radius_m = float(radius_m)
        self.height_m = float(height_m)

    @classmethod
    def from_params(cls, params: Mapping) -> 'CylinderSideFace':
        return cls(radius_m=face_size_m(params, 'expected_radius_m', what='cylinder radius'),
                   height_m=face_size_m(params, 'expected_height_m', what='cylinder height'))

    def __repr__(self) -> str:                                      # pragma: no cover - debug aid
        return (f'CylinderSideFace(radius_m={self.radius_m}, height_m={self.height_m})'
                if self.has_known_area else 'CylinderSideFace(size unknown)')

    @property
    def has_known_area(self) -> bool:
        return self.radius_m > 0.0 and self.height_m > 0.0

    def half_extents(self, offsets: np.ndarray, estimator: Estimator) -> Tuple[float, float]:
        """(0, 0) -- "not estimated", which is what `GraspFit.half_extents_m` documents it as.

        Not an oversight and not laziness: the offsets handed in are IN THE PLANE the fit found,
        and on a cylinder that plane is the RANSAC band, whose half-width is set by the
        tolerance (11,40 mm for a 32,5 mm bottle at 2 mm) and not by the face. Reporting it as
        the face's extent is the silent mis-measurement this face was written to avoid; reporting
        it under a neutral name and then not gating on it would still put a wrong number on the
        quality sidecar, where a tuner reads it. So nothing is claimed at all until a cylinder
        FIT exists to claim it honestly.
        """
        return (0.0, 0.0)

    def expected_area_px(self, z_m: float, K: np.ndarray) -> Optional[float]:
        """(2r * f / z) * (h * f / z) -- the silhouette, a rectangle of 2r by h.

        `2r` and not `pi*r`: the camera sees the cylinder's OUTLINE, which is as wide as the
        diameter, not the curved surface unrolled. And no cos(tilt): the axis leaning toward the
        lens foreshortens `h` exactly as a disc's tilt foreshortens its area, and dividing that
        back out is the correction ADR-0018 Decision 5 removed.
        """
        f = _focal_px(self, z_m, K)
        if f is None:
            return None
        return (2.0 * self.radius_m * f / z_m) * (self.height_m * f / z_m)

    def extent_terms(self, half_extents_m: Tuple[float, float]) -> Dict[str, float]:
        """Empty, and `extent_term_keys` is empty to match, so `pose_quality` gates on neither."""
        return {}


FACES: Dict[str, Type[GraspFace]] = {
    DiscFace.name: DiscFace,
    RectFace.name: RectFace,
    CylinderSideFace.name: CylinderSideFace,
}

#: The faces THIS package ships, captured before `register_face` can add one: what
#: `trainit_perception.vocabulary.export()` reads (ADR-0021, vNext M2.2).
BUILTIN_FACES: Tuple[Type[GraspFace], ...] = tuple(FACES.values())

#: The face the runtime assumes when `grasp_face` is unset. The `'disc'` literals in
#: `rules.computable_terms`, `ranker.ranking_defaults` and `Ranker.from_params` say the same and
#: stay as they are; test_vocabulary.py holds them to this value, which is what a configurator
#: reads (vNext M2.2).
DEFAULT_FACE = DiscFace.name


def available_faces() -> list:
    return sorted(FACES)


def register_face(cls: Type[GraspFace]) -> Type[GraspFace]:
    """Register a face from outside this package. Same escape hatch as `EXTERNAL_METHODS`."""
    if not getattr(cls, 'name', ''):
        raise ValueError('a grasp face needs a non-empty `name`')
    FACES[cls.name] = cls
    return cls


def make_face(name: str, params: Mapping | None = None) -> GraspFace:
    if name not in FACES:
        raise KeyError(f'unknown grasp face {name!r} (available: {", ".join(available_faces())})')
    return FACES[name].from_params(params or {})


def _face_class(name: str) -> Type[GraspFace]:
    if name not in FACES:
        raise KeyError(f'unknown grasp face {name!r} (available: {", ".join(available_faces())})')
    return FACES[name]


def face_params(name: str) -> Tuple[str, ...]:
    """The parameter keys a face reads -- what a configurator renders a form from."""
    return tuple(_face_class(name).PARAMS)


def face_extent_term_keys(name: str) -> Tuple[str, ...]:
    """The quality terms this face produces -- what `pose_quality` gates on when it is chosen."""
    return tuple(_face_class(name).extent_term_keys)


def face_enables_terms(name: str) -> Tuple[str, ...]:
    """The candidate TERMS that become computable once this face's `PARAMS` are set.

    What a configurator needs to answer "if I fill this in, which `candidate_rule` can I then
    choose" -- and what it must NOT keep its own copy of (ADR-0018 Decision 6).
    """
    return tuple(_face_class(name).ENABLES_TERMS)


def face_schema(name: str) -> Dict[str, Tuple[str, ...]]:
    """EVERYTHING a configurator needs in order to configure this face, in one call.

    The three declarations and the name, as plain tuples of strings -- no classes, nothing to
    import, serialisable as it stands:

        {'name': 'rect',
         'params': ('expected_half_a_m', 'expected_half_b_m'),
         'extent_term_keys': ('extent_a_ratio', 'extent_b_ratio'),
         'enables_terms': ('visible_fraction',)}

    This function is the whole of ADR-0018 Decision 6 on this side of the boundary. The asymmetry
    it closes was verified: a hand-written `rect` face was ACCEPTED by this runtime and REFUSED by
    the Setup Assistant, because the Assistant's parameter list was a fixed four-tuple that never
    wrote `grasp_face` -- the key this package already read -- and its computable-terms mirror
    hardcoded `expected_radius_m`. Ask the registry; do not keep a copy of the answer.
    """
    cls = _face_class(name)
    return {'name': name, 'params': tuple(cls.PARAMS),
            'extent_term_keys': tuple(cls.extent_term_keys),
            'enables_terms': tuple(cls.ENABLES_TERMS)}


def faces_schema() -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """`face_schema` for every registered face, including one registered from outside."""
    return {name: face_schema(name) for name in available_faces()}


def all_face_params() -> Tuple[str, ...]:
    """Every face parameter any registered face reads, sorted.

    Two callers, both of which would otherwise hardcode a list that goes stale on the next face:
    `fit3d._refuse_face_params`, which refuses a face parameter passed to a FIT, and a generator
    that must know the full set of ranking keys it may have to write or clear.
    """
    return tuple(sorted({key for cls in FACES.values() for key in cls.PARAMS}))


# --- the plug, in the shape ADR-0017 Decision 2 names it -------------------------------------
# `fit` is anything carrying a 3-D point in the camera's optical frame: a `GraspFit` (`point`) or
# a `Detection` (`position`). Both are passed here -- a detector that ran a 3D fit has the first,
# a detector that only read the depth at a centroid has the second -- and the only thing either
# is asked for is the depth the face is seen at.

def _depth_of(fit) -> float:
    for attr in ('point', 'position'):
        p = getattr(fit, attr, None)
        if p is not None:
            return float(p[2])
    raise TypeError(f'{type(fit).__name__} carries neither `point` nor `position`')


def expected_area_px(face: GraspFace, fit, K: np.ndarray) -> Optional[float]:
    """Pixels the face should cover, at the depth `fit` was found at. None = the size is unknown."""
    return face.expected_area_px(_depth_of(fit), K)


def extent_terms(face: GraspFace, fit) -> Dict[str, float]:
    """The fitted extent against the configured face. Empty when the size is unknown."""
    return face.extent_terms(tuple(getattr(fit, 'half_extents_m', (0.0, 0.0))))
