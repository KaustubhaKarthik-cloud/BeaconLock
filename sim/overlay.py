"""
sim.overlay -- UI-agnostic visualisation helpers (OpenCV drawing only).

Produces the images shown by every front-end, so the web dashboard, the
headless video writer and a future desktop GUI look identical:

* :func:`draw_camera_overlay` -- camera frame + crosshair, detection bounding
  box, Kalman estimate, optional ground truth, lock-status badge, HUD text.
* :class:`Minimap` -- downscaled world with the camera FOV rectangle, beacon
  ground truth, Kalman estimate and recent boresight trail.
* :func:`compose_dashboard_frame` -- side-by-side camera + minimap + stats
  (used for MP4/GIF export of headless runs).

No web-framework imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .engine import FrameResult, SimulationEngine

__all__ = ["STATE_COLORS", "ACCENT", "draw_camera_overlay", "Minimap", "compose_dashboard_frame"]

#: BGR colour per tracker state (badge, FOV rectangle, bounding box).
#: Restrained semantic palette: grey idle/search, amber acquiring, green locked, red lost.
STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "IDLE": (134, 119, 107),       # #6b7786
    "SEARCHING": (194, 183, 174),  # #aeb7c2
    "ACQUIRING": (61, 163, 232),   # #e8a33d
    "LOCKED": (119, 180, 79),      # #4fb477
    "LOST": (76, 87, 224),         # #e0574c
}
ACCENT = (61, 163, 232)            # phosphor amber #e8a33d

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_AA = cv2.LINE_AA


def _dim(c: Tuple[int, int, int], k: float) -> Tuple[int, int, int]:
    return tuple(int(v * k) for v in c)  # type: ignore[return-value]


def _etched(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.36) -> None:
    """Low-contrast 'etched' label: dark shadow + dim grey face."""
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), _FONT, scale, (8, 8, 8), 1, _AA)
    cv2.putText(img, text, org, _FONT, scale, (92, 92, 92), 1, _AA)


def _brackets(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], arm: int = 6) -> None:
    """Small corner brackets around a box."""
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * arm, cy), color, 1, _AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * arm), color, 1, _AA)


def _reticle(img: np.ndarray, cx: int, cy: int, lock_radius: float) -> None:
    """Fine crosshair with graduated tick marks and a dashed lock-radius ring."""
    hair, tick = (170, 170, 170), (96, 96, 96)
    for a, b in ((6, 28),):
        cv2.line(img, (cx - b, cy), (cx - a, cy), hair, 1, _AA)
        cv2.line(img, (cx + a, cy), (cx + b, cy), hair, 1, _AA)
        cv2.line(img, (cx, cy - b), (cx, cy - a), hair, 1, _AA)
        cv2.line(img, (cx, cy + a), (cx, cy + b), hair, 1, _AA)
    cv2.circle(img, (cx, cy), 0, hair, -1)
    for d in range(40, 161, 10):          # graduations every 10 px, major every 50 px
        L = 5 if d % 50 == 0 else 2
        cv2.line(img, (cx - d, cy - L), (cx - d, cy + L), tick, 1)
        cv2.line(img, (cx + d, cy - L), (cx + d, cy + L), tick, 1)
        if d <= 120:
            cv2.line(img, (cx - L, cy - d), (cx + L, cy - d), tick, 1)
            cv2.line(img, (cx - L, cy + d), (cx + L, cy + d), tick, 1)
    r = int(round(lock_radius))
    for a0 in range(0, 360, 15):          # dashed ring
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 6, (84, 84, 84), 1, _AA)


def draw_camera_overlay(engine: "SimulationEngine", result: Optional["FrameResult"] = None,
                        frame: Optional[np.ndarray] = None, show_truth: bool = False) -> np.ndarray:
    """Return a BGR copy of the camera frame with the tracking overlay drawn."""
    r = result if result is not None else engine.last
    img = frame if frame is not None else (r.frame if r is not None else engine.preview())
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = out.shape[:2]
    state = engine.state.value
    col = STATE_COLORS.get(state, (200, 200, 200))
    cam = engine.camera

    # Fine reticle at the boresight (frame centre).
    cx, cy = int(round(w / 2)), int(round(h / 2))
    _reticle(out, cx, cy, engine.tracker.config.lock_radius_px)

    if r is not None:
        x0, y0 = r.viewport[0], r.viewport[1]
        if r.detection is not None:
            d = r.detection
            bx, by, bw, bh = d.bbox
            pad = 7
            p0, p1 = (bx - pad, by - pad), (bx + bw + pad, by + bh + pad)
            cv2.rectangle(out, p0, p1, _dim(col, 0.55), 1, cv2.LINE_8)      # thin 1px tracking box
            _brackets(out, p0[0], p0[1], p1[0], p1[1], col, arm=5)          # crisp corner brackets
            dx, dy = int(round(d.x)), int(round(d.y))
            cv2.line(out, (dx - 3, dy), (dx + 3, dy), col, 1, _AA)
            cv2.line(out, (dx, dy - 3), (dx, dy + 3), col, 1, _AA)
        if r.track.estimate is not None and state != "SEARCHING":
            ex, ey = int(round(r.track.estimate[0] - x0)), int(round(r.track.estimate[1] - y0))
            if 0 <= ex < w and 0 <= ey < h:   # Kalman estimate: small grey diamond
                pts = np.array([(ex, ey - 9), (ex + 9, ey), (ex, ey + 9), (ex - 9, ey)], np.int32)
                cv2.polylines(out, [pts], True, (176, 176, 176), 1, _AA)
        if show_truth:
            gx, gy = r.truth[0] - x0, r.truth[1] - y0
            if 0 <= gx < w and 0 <= gy < h:
                cv2.drawMarker(out, (int(round(gx)), int(round(gy))), (240, 240, 240), cv2.MARKER_TILTED_CROSS, 10, 1)

    # State tag (small square + label, no filled pill).
    cv2.rectangle(out, (26, 22), (31, 27), col, -1)
    cv2.putText(out, state, (38, 29), _FONT, 0.42, col, 1, _AA)
    _etched(out, "SYNTHETIC", (w - 26 - 66, 29))
    hud_col = (150, 150, 150)
    if r is not None:
        hud = f"T+{r.t:7.2f} s   ERR {r.error_px:6.1f} px   PROC {r.proc_ms:4.1f} ms"
        cv2.putText(out, hud, (26, h - 22), _FONT, 0.38, hud_col, 1, _AA)
    bx_, by_ = cam.center
    cv2.putText(out, f"AZ {bx_:6.0f}  EL {by_:5.0f} px", (w - 26 - 150, h - 22), _FONT, 0.38, hud_col, 1, _AA)
    return out


class Minimap:
    """Cached, downscaled world view for drawing the FOV rectangle on."""

    def __init__(self, engine: "SimulationEngine", width: int = 480) -> None:
        self.engine = engine
        sc = engine.scene
        self.scale = width / sc.width
        self.size = (width, int(round(sc.height * self.scale)))
        layer = sc.static_layer
        # INTER_AREA would wash out stars; take a max-pool-like resize for visibility.
        small = cv2.resize(cv2.dilate(layer, np.ones((3, 3), np.uint8)), self.size, interpolation=cv2.INTER_AREA)
        small = np.clip(small * 1.6 + 8, 0, 255).astype(np.uint8)
        self.base = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        # Graticule: faint grid every 160 world px, labelled every 320 px.
        grid, lab = (48, 40, 32), (110, 100, 92)
        for gx in range(160, sc.width, 160):
            X = int(round(gx * self.scale))
            cv2.line(self.base, (X, 0), (X, self.size[1] - 1), grid, 1)
            if gx % 320 == 0:
                cv2.putText(self.base, str(gx), (X + 2, 9), _FONT, 0.28, lab, 1, _AA)
        for gy in range(160, sc.height, 160):
            Y = int(round(gy * self.scale))
            cv2.line(self.base, (0, Y), (self.size[0] - 1, Y), grid, 1)
            if gy % 320 == 0:
                cv2.putText(self.base, str(gy), (2, Y - 2), _FONT, 0.28, lab, 1, _AA)

    def _p(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def render(self) -> np.ndarray:
        e = self.engine
        img = self.base.copy()
        s = self.scale
        # Boresight trail.
        pts = [self._p(*p) for p in e.trail]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (96, 96, 96), 1, _AA)
        # Beacons (ground truth): rings.
        for i, b in enumerate(e.scene.beacons):
            c = (235, 235, 235) if i == e.target_index else (130, 130, 130)
            p = self._p(*b.position)
            cv2.circle(img, p, 4, c, 1, _AA)
            cv2.circle(img, p, 0, c, -1)
        # Kalman estimate: small cross.
        if e.tracker.kf.initialized and e.state.value != "SEARCHING":
            cv2.drawMarker(img, self._p(*e.tracker.kf.position), (190, 190, 190), cv2.MARKER_CROSS, 9, 1)
        # FOV rectangle (state colour) + corner brackets.
        x0, y0, w, h = e.camera.viewport
        col = STATE_COLORS.get(e.state.value, (200, 200, 200))
        p0, p1 = self._p(x0, y0), self._p(x0 + w, y0 + h)
        cv2.rectangle(img, p0, p1, _dim(col, 0.6), 1, cv2.LINE_8)
        _brackets(img, p0[0], p0[1], p1[0], p1[1], col, arm=6)
        cv2.circle(img, self._p(*e.camera.center), 1, col, -1)
        _etched(img, f"WORLD {e.scene.width}x{e.scene.height} / SYNTHETIC", (6, self.size[1] - 7), 0.3)
        return img


def compose_dashboard_frame(engine: "SimulationEngine", minimap: Minimap, show_truth: bool = True) -> np.ndarray:
    """Camera overlay (left) + minimap and live stats (right) as one BGR image."""
    cam = draw_camera_overlay(engine, show_truth=show_truth)
    mm = minimap.render()
    h = cam.shape[0]
    panel_w = mm.shape[1]
    right = np.full((h, panel_w, 3), 18, np.uint8)
    right[: mm.shape[0]] = mm[: h]
    s = engine.metrics.summary()
    lines = [
        "BeaconLock - coarse alignment (SYNTHETIC)",
        f"state      {engine.state.value}",
        f"acq time   {s.get('acquisition_time_s') if s.get('acquisition_time_s') is not None else '-'} s",
        f"lock ret.  {s.get('lock_retention_pct') if s.get('lock_retention_pct') is not None else '-'} %",
        f"err avg    {s.get('err_locked_avg_px') if s.get('err_locked_avg_px') is not None else '-'} px",
        "dist  " + "  ".join(f"{k[:4]} {v:.0f}" for k, v in engine.disturbances.intensities.items()),
    ]
    y = mm.shape[0] + 24
    for i, line in enumerate(lines):
        cv2.putText(right, line, (10, y + 22 * i), _FONT, 0.45, (200, 200, 200) if i else ACCENT, 1, _AA)
    return np.hstack([cam, np.full((h, 4, 3), 60, np.uint8), right])
