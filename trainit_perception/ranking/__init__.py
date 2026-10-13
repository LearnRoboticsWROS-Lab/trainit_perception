"""The candidate ranking: which object to pick, and why (ADR-0014, ADR-0016, ADR-0017).

    terms.py    the TERMS as pure functions: box_overlap, visible_fraction, clearance, edge_on,
                pose_quality (+ its four gates as serialisable configuration)
    fit3d.py    the 3D fit the depth terms need: GraspFit, centroid, plane_ransac, plane_nearest,
                approach_obstruction
    faces.py    the GRASP FACE plug -- the only place a shape assumption lives, and the registry
                that says what a face needs (`face_schema`, ADR-0018 Decision 6); a configurator
                reads it through `trainit_perception.vocabulary.export()` and a committed
                snapshot, never by importing this package at generation (ADR-0021)
    rules.py    a rule is a product of named terms; `check_rule` refuses one that is not
                computable, because the published ORDER is the ranking
    ranker.py   the one place a frame's terms are assembled, refused and permuted
    vocabulary.py  the ranking half of the configuration vocabulary export (names, kinds and
                requirements only)

The LOOP is not here and never will be: the retry over candidates, the stall recovery and the
refill are the cell's business logic, they are a behaviour tree the Setup Assistant generates
(ADR-0016 Decision 4), and pulling them in is the one move that would turn this into custom
engineering wearing a framework's name.
"""
from .faces import (CylinderSideFace, DiscFace, FACES, GraspFace, MAX_FACE_M,  # noqa: F401
                    MIN_FACE_M, RectFace, all_face_params, available_faces, expected_area_px,
                    extent_terms, face_enables_terms, face_extent_term_keys, face_params,
                    face_schema, face_size_m, faces_schema, make_face, mad_extent, p90_extent,
                    register_face)
from .fit3d import (GIVES_ORIENTATION, GraspFit, METHODS, approach_obstruction,  # noqa: F401
                    available_methods_3d, centroid, fit, plane_nearest, plane_ransac)
from .ranker import PARAM_SINCE, Ranker, param_since, ranking_defaults  # noqa: F401
from .rules import (AUTO, DEFAULT_PREFERENCE, PROVIDES_BOX, PROVIDES_FIT3D,  # noqa: F401
                    PROVIDES_MASK_AREA, PROVIDES_SCORE, RULES, Rule, TERM_NEEDS, available_rules,
                    check_rule, computable_rules, computable_terms, default_rule, rank_order,
                    register_rule, resolve_rule, term_needs)
from .terms import (DEFAULT_EXTENT_KEYS, PoseQualityGates, approach_clearance,  # noqa: F401
                    box_overlap, edge_on, pose_quality, visible_fraction)
