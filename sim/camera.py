"""
sim.camera -- virtual pan-tilt camera looking into the synthetic world.

The camera is modelled as a rectangular **viewport** (default 640 x 480) that
can be slewed across the larger world image produced by :class:`sim.scene.Scene`.
This is a 2-D, small-angle stand-in for a gimballed camera: one world pixel of
pan/tilt corresponds to one pixel of image shift (a unit IFOV), which is the
standard simplification for coarse-pointing studies.

Conventions
-----------
* ``(cx, cy)`` is the **boresight** (centre of the viewport) in world pixels.
  A perfectly aligned link has the beacon exactly at the boresight.
* ``pan(dx)`` moves the boresight right by ``dx`` px (negative = left);
  ``tilt(dy)`` moves it down by ``dy`` px (image convention, y down).
* The viewport is always **clamped inside the world** (mechanical limits of
  the gimbal / field of regard).
* Optional **slew-rate limit** ``max_slew`` (px/s per axis): call
  :meth:`tick` once per simulation step with ``dt``; subsequent pan/tilt calls
  in that step can move at most ``max_slew * dt`` pixels per axis in total.

No web-framework imports (UI-independent core).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .scene import Scene

__all__ = ["PanTiltCamera"]


class PanTiltCamera:
    """Movable viewport into a :class:`~sim.scene.Scene`.

    Parameters
    ----------
    scene:
        The world to look into.
    width, height:
        Sensor resolution in pixels (viewport size), default 640 x 480.
    center:
        Initial boresight ``(cx, cy)`` in world pixels (default: world centre).
    max_slew:
        Maximum slew rate in px/s per axis, or ``None`` for unlimited.
    """

    def __init__(
        self,
        scene: Scene,
        width: int = 640,
        height: int = 480,
        center: Optional[Tuple[float, float]] = None,
        max_slew: Optional[float] = None,
    ) -> None:
        if width > scene.width or height > scene.height:
            raise ValueError("Camera viewport must not be larger than the world")
        self.scene = scene
        self.width = int(width)
        self.height = int(height)
        self.max_slew = None if max_slew is None else float(max_slew)
        # Allowed boresight range so that the viewport stays inside the world.
        self._cmin = np.array([self.width / 2.0, self.height / 2.0])
        self._cmax = np.array([scene.width - self.width / 2.0, scene.height - self.height / 2.0])
        c = (scene.width / 2.0, scene.height / 2.0) if center is None else center
        self._center = self._clamp(np.array(c, dtype=np.float64))
        self._home = self._center.copy()
        self._budget = np.array([math.inf, math.inf])
        self._last_center = self._center.copy()
        self._dt = 0.0
        self.velocity = np.zeros(2)  # boresight velocity over the last step, px/s

    # ------------------------------------------------------------------ state
    def _clamp(self, c: np.ndarray) -> np.ndarray:
        return np.clip(c, self._cmin, self._cmax)

    @property
    def center(self) -> Tuple[float, float]:
        """Boresight position ``(cx, cy)`` in world pixels."""
        return float(self._center[0]), float(self._center[1])

    @property
    def origin(self) -> Tuple[int, int]:
        """Integer world coordinate of the viewport's top-left pixel."""
        x0 = int(round(self._center[0] - self.width / 2.0))
        y0 = int(round(self._center[1] - self.height / 2.0))
        x0 = min(max(x0, 0), self.scene.width - self.width)
        y0 = min(max(y0, 0), self.scene.height - self.height)
        return x0, y0

    @property
    def viewport(self) -> Tuple[int, int, int, int]:
        """``(x0, y0, width, height)`` of the field of view in world pixels."""
        x0, y0 = self.origin
        return x0, y0, self.width, self.height

    @property
    def frame_center(self) -> Tuple[float, float]:
        """Boresight in *camera* pixel coordinates (where the crosshair is)."""
        return self.world_to_cam(*self.center)

    def world_to_cam(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world pixel coordinates to camera pixel coordinates."""
        x0, y0 = self.origin
        return x - x0, y - y0

    def cam_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Convert camera pixel coordinates to world pixel coordinates."""
        x0, y0 = self.origin
        return u + x0, v + y0

    def in_view(self, x: float, y: float) -> bool:
        """True if world point ``(x, y)`` lies inside the field of view."""
        u, v = self.world_to_cam(x, y)
        return 0 <= u < self.width and 0 <= v < self.height

    # --------------------------------------------------------------- motion
    def tick(self, dt: float) -> None:
        """Start a new simulation step of length ``dt`` seconds.

        Refreshes the slew budget and updates :attr:`velocity` from the motion
        performed during the previous step.
        """
        if self._dt > 0:
            self.velocity = (self._center - self._last_center) / self._dt
        self._last_center = self._center.copy()
        self._dt = float(dt)
        lim = math.inf if self.max_slew is None else self.max_slew * dt
        self._budget = np.array([lim, lim])

    def _move(self, d: np.ndarray) -> Tuple[float, float]:
        d = np.clip(d, -self._budget, self._budget)
        new = self._clamp(self._center + d)
        actual = new - self._center
        self._budget = self._budget - np.abs(actual)
        self._center = new
        return float(actual[0]), float(actual[1])

    def pan(self, dx: float) -> float:
        """Pan by ``dx`` px (+ right). Returns the movement actually applied."""
        return self._move(np.array([dx, 0.0]))[0]

    def tilt(self, dy: float) -> float:
        """Tilt by ``dy`` px (+ down). Returns the movement actually applied."""
        return self._move(np.array([0.0, dy]))[1]

    def move(self, dx: float, dy: float) -> Tuple[float, float]:
        """Pan and tilt in one call (subject to slew limit and clamping)."""
        return self._move(np.array([dx, dy], dtype=np.float64))

    def point_at(self, x: float, y: float) -> Tuple[float, float]:
        """Slew towards world point ``(x, y)`` (rate-limited)."""
        return self._move(np.array([x, y]) - self._center)

    def set_center(self, x: float, y: float) -> None:
        """Teleport the boresight (ignores the slew limit; for setup/reset)."""
        self._center = self._clamp(np.array([x, y], dtype=np.float64))
        self._last_center = self._center.copy()
        self.velocity = np.zeros(2)

    def reset(self) -> None:
        """Return to the initial boresight."""
        self.set_center(*self._home)
        self._dt = 0.0
        self._budget = np.array([math.inf, math.inf])

    # -------------------------------------------------------------- imaging
    def get_frame(self, noise: bool = True) -> np.ndarray:
        """Render the current field of view: uint8 ``(height, width)``.

        Disturbances (turbulence, vibration, blur, extra noise) are applied
        separately by :class:`sim.disturbances.Disturbances`.
        """
        x0, y0 = self.origin
        return self.scene.render_region(x0, y0, self.width, self.height, noise=noise)

    def __repr__(self) -> str:
        return f"PanTiltCamera({self.width}x{self.height}, center=({self._center[0]:.1f}, {self._center[1]:.1f}))"
