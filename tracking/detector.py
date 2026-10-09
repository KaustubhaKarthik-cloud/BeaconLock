"""
tracking.detector -- find the optical beacon in a camera frame.

Pipeline (classic, explainable, fast -- ~1 ms on 640x480):

1. **Denoise** with a small Gaussian blur.  This suppresses per-pixel sensor
   noise and, importantly, attenuates point-like stars (sigma ~1 px) far more
   than the larger beacon spot (sigma ~4 px).
2. **Peak gate**: if the brightest blurred pixel is below ``min_peak`` there is
   no beacon in view -> ``None``.
3. **Adaptive threshold** at ``max(min_threshold, rel_threshold * peak)``.
4. **Contours** of the binary mask; reject blobs outside
   ``[min_area, max_area]``.
5. **Score** each blob by ``peak * sqrt(area)``; if a predicted position
   (``hint``) is supplied by the tracker, blobs far from it are penalised
   (soft gating) so a bright star cannot steal the lock.
6. **Centroid** = intensity-weighted centre of mass of the winning blob
   (sub-pixel accuracy).

Returns a :class:`Detection` in **camera pixel coordinates** or ``None``.
No web-framework imports.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

__all__ = ["Detection", "DetectorConfig", "BeaconDetector"]


@dataclass
class Detection:
    """A detected beacon blob (camera pixel coordinates)."""

    x: float                     # sub-pixel centroid, px
    y: float
    area: float                  # blob area in px^2
    peak: float                  # max (blurred) intensity in the blob, DN
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectorConfig:
    blur_sigma: float = 1.2       # denoise blur, px
    min_peak: float = 110.0       # below this nothing is accepted (stars peak < ~100 after blur)
    min_threshold: float = 70.0   # absolute floor for the binary threshold, DN
    rel_threshold: float = 0.5    # threshold as a fraction of the frame peak
    min_area: float = 4.0         # px^2
    max_area: float = 5000.0      # px^2
    gate_px: float = 120.0        # soft-gate radius around the tracker's prediction

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DetectorConfig":
        return cls(**dict(d or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BeaconDetector:
    """Threshold + contour beacon detector.  See module docstring."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        #: All candidate blobs from the last call (for overlays / debugging).
        self.candidates: List[Detection] = []

    def detect(self, frame: np.ndarray, hint: Optional[Tuple[float, float]] = None) -> Optional[Detection]:
        """Locate the beacon in ``frame`` (uint8 gray or BGR).

        Parameters
        ----------
        frame:
            Camera image.
        hint:
            Optional predicted beacon position (camera px) from the tracker.

        Returns
        -------
        Detection or None
            ``None`` when no sufficiently bright, beacon-sized blob is present.
        """
        c = self.config
        self.candidates = []
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(gray, (0, 0), c.blur_sigma) if c.blur_sigma > 0 else gray
        peak = float(g.max())
        if peak < c.min_peak:
            return None
        thr = max(c.min_threshold, c.rel_threshold * peak)
        _, mask = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[Detection] = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = g[y:y + h, x:x + w].astype(np.float32)
            roi_mask = mask[y:y + h, x:x + w] > 0
            area = float(roi_mask.sum())
            if area < c.min_area or area > c.max_area:
                continue
            blob_peak = float(roi[roi_mask].max())
            if blob_peak < c.min_peak:
                continue
            wts = (roi - thr) * roi_mask
            wts[wts < 0] = 0
            s = float(wts.sum())
            if s <= 0:
                continue
            yy, xx = np.mgrid[0:h, 0:w]
            cx = x + float((wts * xx).sum() / s)
            cy = y + float((wts * yy).sum() / s)
            score = blob_peak * math.sqrt(area)
            if hint is not None:
                dist = math.hypot(cx - hint[0], cy - hint[1])
                score *= 1.0 / (1.0 + (dist / c.gate_px) ** 2)
            det = Detection(cx, cy, area, blob_peak, (x, y, w, h), score)
            self.candidates.append(det)
            if best is None or det.score > best.score:
                best = det
        return best
