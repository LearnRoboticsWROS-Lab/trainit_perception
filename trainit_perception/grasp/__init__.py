"""Grasp PROVIDERS: measured surface geometry -> canonical grasp candidates (vNext M4, ADR-0026).

    camera.py         the camera model a provider back-projects through (K refreshed, plumb_bob undistorted,
                      any other non-zero distortion refused)
    surface_grasp.py  SurfaceGrasp (`tp.surface_grasp`): regions + registered depth -> ranked, constrained
                      candidates and explicit rejection codes. Pure, ROS-free, deterministic
    overlay.py        the same geometry drawn on the RGB image, for headless evidence (optional)
    instances.py      the `instance_topic` segmentation: an external segmenter's instance image -> regions (0.7.0)
    eligibility.py    the workspace eligibility filter (a pixel polygon + a depth band), before `propose` (0.7.0)
    objects.py        the objects PRESENT: one per eligible region, for the presence capture (0.7.0, M4.5b)
    scripted.py       the SCRIPTED provider (`tp.scripted`): candidates from a scenario file, for mock tests (0.7.0)

The ROS side is `trainit_perception.grasp_msgs` (GraspCandidateArray, markers), `nodes/surface_grasp_node.py` and
`nodes/scripted_grasp_node.py`.
"""
from .camera import CameraModel, CameraModelError, SUPPORTED_DISTORTION  # noqa: F401
from .surface_grasp import (ALWAYS_NOT_EVALUATED, CHECK_ORDER, Candidate, ConfigError, GraspResult,  # noqa: F401
                            NORMAL_POLICY, PARAMETERS, PROVIDER_ID, Region, Rejection, SurfaceGraspConfig, TERMS,
                            defaults, diagnostics, grasp_orientation, minimal_rotation, orient_toward_observer,
                            propose, quat_xyzw, reference_rotation, region_id, region_ids,
                            regions_from_detections)
