"""Colour-mask detector: HSV threshold -> blobs -> centroid -> depth -> 3D point.

The v4 method for the LOCATE goal. Deliberately simple and fully inspectable, which is
what makes it teachable: every stage has a picture (the mask) and a number (the area).

What it can and cannot give. At this cell's camera height a 2 cm workpiece is ~18 px
wide -- plenty for a centroid, not enough for an orientation from the image. So the
detection carries a position and a SIZE, and an identity orientation. A top-down
suction grasp needs nothing more; a 6-DoF grasp needs a different method (fiducial,
learned model) through the same contract.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .base import Detection, Detector, back_project, depth_at, pixel_extent_to_metres


class ColorMaskDetector(Detector):
    method = 'color_mask'

    @classmethod
    def defaults(cls) -> dict:
        return {
            'class_id': 'object',
            # OpenCV hue is 0-179. A range with lo > hi WRAPS around 0, which is how a
            # RED is written: [170, 10] means 170..179 plus 0..10. Red straddles the
            # hue seam and a naive single range silently loses half of it.
            'h': [170, 10],
            # Saturation floor kept LOW on purpose. Measured on the Isaac cell under a
            # dome light, the red cube renders at S = 62-72 out of 255 (bright and
            # washed out); the theory default of 120 found NOTHING. A real painted
            # object is far more saturated, so 40 is a floor that admits both.
            's': [40, 255],
            'v': [60, 255],
            'min_area_px': 60,
            'max_objects': 1,
            'morph_kernel': 3,          # 0 = off. Removes speckle, closes small gaps.
            'depth_window_px': 5,
        }

    # --- the mask, exposed on its own so the tuner can show it ----------------------
    def mask(self, rgb: np.ndarray) -> np.ndarray:
        p = self.params
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)      # rgb8 in, hence RGB2HSV not BGR
        h_lo, h_hi = int(p['h'][0]), int(p['h'][1])
        s_lo, s_hi = int(p['s'][0]), int(p['s'][1])
        v_lo, v_hi = int(p['v'][0]), int(p['v'][1])
        if h_lo <= h_hi:
            m = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
        else:                                            # hue wrap (red)
            m = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (179, s_hi, v_hi)) | \
                cv2.inRange(hsv, (0, s_lo, v_lo), (h_hi, s_hi, v_hi))
        k = int(p['morph_kernel'])
        if k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return m

    def debug_mask(self, rgb: np.ndarray):
        return self.mask(rgb)

    def detect(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> List[Detection]:
        p = self.params
        m = self.mask(rgb)
        # Connected components, not contours: cv2.contourArea() measures the polygon
        # through pixel CENTRES and under-reads small blobs by a full ring of pixels
        # (an 18 px square comes out at 287, not 324 -- 11% low). At the sizes this
        # cell works with that error would bias min_area_px and the solidity score.
        # Counting mask pixels is exact.
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        blobs = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)]     # 0 = background
        blobs = [(a, i) for a, i in blobs if a >= int(p['min_area_px'])]
        blobs.sort(key=lambda t: -t[0])

        out: List[Detection] = []
        for area, i in blobs[: int(p['max_objects'])]:
            u, v = float(centroids[i][0]), float(centroids[i][1])
            z = depth_at(depth_m, u, v, int(p['depth_window_px']))
            if not np.isfinite(z) or z <= 0.0:
                continue                                  # a hole with no rescue: skip it
            w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            # score: how SOLID the blob is. A top-down cube is a filled square (~1.0);
            # speckle and partial occlusions score lower. Cheap and honest.
            fill = float(area) / float(max(1, w * h))
            fx, fy = K[0, 0], K[1, 1]
            size = (pixel_extent_to_metres(w, z, fx), pixel_extent_to_metres(h, z, fy), 0.0)
            out.append(Detection(
                class_id=str(p['class_id']), score=min(1.0, fill),
                position=back_project(u, v, z, K), size=size,
                pixel=(u, v), area_px=int(area),
                extra={'bbox_px': float(max(w, h))},
            ))
        return out
