"""
sim.trajectories -- beacon motion models for the BeaconLock synthetic world.

Every beacon in a :class:`sim.scene.Scene` owns one :class:`Trajectory`.
A trajectory is a small, *stateful* object that answers one question:

    "Where is the beacon (in world pixel coordinates) after another ``dt`` seconds?"

Design notes
------------
* Coordinates are **world pixels**, origin at the top-left corner, ``x`` to the
  right, ``y`` downwards (the OpenCV / image convention).
* Time is in **seconds**; velocities are in **pixels per second**.
* Trajectories are deterministic: any randomness (random waypoint) is drawn from
  a :class:`numpy.random.Generator` that is created from an integer seed, so
  ``reset()`` replays exactly the same path.
* Trajectories know the world bounds so they can bounce / wrap / pick waypoints
  inside the world.  The bounds are injected by the scene via :meth:`bind`.
* This module depends only on NumPy (no OpenCV, no web frameworks) so it can be
  frozen into a standalone executable together with the rest of ``sim/``.

All scenes produced with these models are SYNTHETIC. No real mission / ISRO data
is used or imitated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np

__all__ = [
    "Trajectory",
    "StaticTrajectory",
    "LinearTrajectory",
    "SinusoidalTrajectory",
    "RandomWaypointTrajectory",
    "TRAJECTORY_TYPES",
    "trajectory_from_dict",
]

Vec2 = Tuple[float, float]


class Trajectory:
    """Abstract base class for all beacon motion models.

    Subclasses implement :meth:`_reset_state` and :meth:`_advance`.  The base
    class keeps track of elapsed time, the current position and an
    instantaneous velocity estimate (finite difference over the last step).

    Parameters
    ----------
    start:
        Initial ``(x, y)`` position in world pixels.  If ``None`` the world
        centre is used once bounds are known.
    margin:
        Distance in pixels to keep from the world border (used by bouncing,
        wrapping and waypoint selection).  Either a scalar or ``[mx, my]``.
        Tip: with a pan-tilt camera clamped to the world, a margin of at least
        half the camera FOV (e.g. ``[330, 250]`` for 640x480) keeps the beacon
        centrable everywhere.
    """

    #: Name used in JSON configs (``"type": "<type_name>"``).
    type_name: str = "base"

    def __init__(self, start: Optional[Vec2] = None, margin: Union[float, Sequence[float]] = 20.0) -> None:
        self._start_cfg = None if start is None else (float(start[0]), float(start[1]))
        self.margin = np.broadcast_to(np.asarray(margin, dtype=np.float64), (2,)).copy()
        self.width: float = 0.0
        self.height: float = 0.0
        self._bound = False
        self.t: float = 0.0
        self.pos = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------ setup
    def bind(self, width: int, height: int) -> "Trajectory":
        """Attach world bounds (pixels) and reset the trajectory to ``t = 0``."""
        self.width = float(width)
        self.height = float(height)
        self._bound = True
        self.reset()
        return self

    @property
    def start(self) -> np.ndarray:
        """Configured start position, defaulting to the world centre."""
        if self._start_cfg is None:
            return np.array([self.width / 2.0, self.height / 2.0])
        return np.array(self._start_cfg, dtype=np.float64)

    def reset(self) -> None:
        """Return to ``t = 0`` and replay the exact same path afterwards."""
        if not self._bound:
            raise RuntimeError("Trajectory.bind(width, height) must be called first")
        self.t = 0.0
        self.vel[:] = 0.0
        self._reset_state()

    # --------------------------------------------------------------- stepping
    def step(self, dt: float) -> np.ndarray:
        """Advance the trajectory by ``dt`` seconds and return the new position.

        The returned array is a copy ``[x, y]`` (float64, world pixels).
        """
        if dt < 0:
            raise ValueError("dt must be non-negative")
        if not self._bound:
            raise RuntimeError("Trajectory.bind(width, height) must be called first")
        prev = self.pos.copy()
        self.t += dt
        self._advance(dt)
        if dt > 0:
            self.vel = (self.pos - prev) / dt
        return self.pos.copy()

    @property
    def position(self) -> Vec2:
        """Current ``(x, y)`` position in world pixels."""
        return float(self.pos[0]), float(self.pos[1])

    @property
    def velocity(self) -> Vec2:
        """Velocity ``(vx, vy)`` in px/s over the last step (finite difference)."""
        return float(self.vel[0]), float(self.vel[1])

    # ------------------------------------------------------ subclass interface
    def _reset_state(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _advance(self, dt: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def _lo_hi(self) -> Tuple[np.ndarray, np.ndarray]:
        """Allowed position box ``(lo, hi)`` after applying the margin."""
        lo = self.margin.copy()
        hi = np.array([self.width - 1, self.height - 1]) - self.margin
        if (hi < lo).any():
            raise ValueError(f"margin {self.margin.tolist()} too large for world {self.width:.0f}x{self.height:.0f}")
        return lo, hi

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a JSON-friendly config dictionary."""
        m = self.margin.tolist()
        d: Dict[str, Any] = {"type": self.type_name, "margin": m[0] if m[0] == m[1] else m}
        if self._start_cfg is not None:
            d["start"] = list(self._start_cfg)
        return d

    def __repr__(self) -> str:
        return f"{type(self).__name__}(t={self.t:.3f}, pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}))"


class StaticTrajectory(Trajectory):
    """A beacon that does not move (useful for unit tests and calibration)."""

    type_name = "static"

    def _reset_state(self) -> None:
        self.pos = self.start.astype(np.float64)

    def _advance(self, dt: float) -> None:
        pass


class LinearTrajectory(Trajectory):
    """Constant-velocity straight-line motion.

    When the beacon reaches the border (minus ``margin``) it either

    * ``"bounce"`` -- reflects specularly (velocity component is negated), or
    * ``"wrap"``   -- reappears on the opposite side (toroidal world).

    Parameters
    ----------
    velocity:
        ``(vx, vy)`` in pixels per second.
    mode:
        ``"bounce"`` (default) or ``"wrap"``.
    """

    type_name = "linear"

    def __init__(
        self,
        start: Optional[Vec2] = None,
        velocity: Vec2 = (150.0, 60.0),
        mode: str = "bounce",
        margin: Union[float, Sequence[float]] = 20.0,
    ) -> None:
        super().__init__(start, margin)
        if mode not in ("bounce", "wrap"):
            raise ValueError(f"LinearTrajectory mode must be 'bounce' or 'wrap', got {mode!r}")
        self.mode = mode
        self._v0 = np.array(velocity, dtype=np.float64)
        self._v = self._v0.copy()

    def _reset_state(self) -> None:
        lo, hi = self._lo_hi()
        self.pos = np.clip(self.start, lo, hi).astype(np.float64)
        self._v = self._v0.copy()

    def _advance(self, dt: float) -> None:
        lo, hi = self._lo_hi()
        p = self.pos + self._v * dt
        if self.mode == "wrap":
            span = hi - lo
            p = lo + np.mod(p - lo, span)
        else:
            # Reflect as many times as needed (handles very large dt robustly).
            for k in range(2):
                span = hi[k] - lo[k]
                if span <= 0:
                    p[k] = lo[k]
                    continue
                u = (p[k] - lo[k]) % (2 * span)          # position on an "unfolded" line
                n_reflections = math.floor((p[k] - lo[k]) / span)
                p[k] = lo[k] + (u if u <= span else 2 * span - u)
                if n_reflections % 2 != 0:
                    self._v[k] = -self._v[k]
        self.pos = p

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(velocity=self._v0.tolist(), mode=self.mode)
        return d


class SinusoidalTrajectory(Trajectory):
    """Drift along a direction while oscillating perpendicular to it.

    ``p(t) = start + v * t + A * sin(2*pi*f*t + phase) * n``

    where ``n`` is the unit vector perpendicular to ``v`` (or the ``y`` axis
    when ``v`` is zero).  This mimics, e.g., a UAV that weaves while flying
    across the field of regard.  The drift part wraps around the world so the
    motion continues indefinitely.

    Parameters
    ----------
    velocity:
        Drift velocity ``(vx, vy)`` in px/s.
    amplitude:
        Peak perpendicular deviation in pixels.
    frequency:
        Oscillation frequency in Hz.
    phase:
        Initial phase in radians.
    wrap:
        Legacy switch: ``True`` -> ``edge="wrap"``, ``False`` -> ``edge="clip"``.
    edge:
        Border behaviour, overrides ``wrap``: ``"wrap"`` (toroidal),
        ``"bounce"`` (mirror-fold, the beacon sweeps back and forth -- best for
        long tracking demos) or ``"clip"`` (stick to the border).
    """

    type_name = "sinusoidal"

    def __init__(
        self,
        start: Optional[Vec2] = None,
        velocity: Vec2 = (250.0, 0.0),
        amplitude: float = 200.0,
        frequency: float = 0.4,
        phase: float = 0.0,
        wrap: bool = True,
        edge: Optional[str] = None,
        margin: Union[float, Sequence[float]] = 20.0,
    ) -> None:
        super().__init__(start, margin)
        if edge is None:
            edge = "wrap" if wrap else "clip"
        if edge not in ("wrap", "bounce", "clip"):
            raise ValueError(f"SinusoidalTrajectory edge must be 'wrap', 'bounce' or 'clip', got {edge!r}")
        self.edge = edge
        self.v = np.array(velocity, dtype=np.float64)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase = float(phase)
        speed = float(np.hypot(*self.v))
        # Unit normal to the drift direction (perpendicular, rotated +90 deg).
        self._n = np.array([-self.v[1] / speed, self.v[0] / speed]) if speed > 1e-9 else np.array([0.0, 1.0])

    def _position_at(self, t: float) -> np.ndarray:
        base = self.start + self.v * t
        offset = self.amplitude * math.sin(2 * math.pi * self.frequency * t + self.phase) * self._n
        p = base + offset
        lo, hi = self._lo_hi()
        span = hi - lo
        if self.edge == "wrap":
            p = lo + np.mod(p - lo, span)
        elif self.edge == "bounce":
            u = np.mod(p - lo, 2 * span)  # triangle-wave fold
            p = lo + np.where(u <= span, u, 2 * span - u)
        else:
            p = np.clip(p, lo, hi)
        return p

    def _reset_state(self) -> None:
        self.pos = self._position_at(0.0)

    def _advance(self, dt: float) -> None:
        self.pos = self._position_at(self.t)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            velocity=self.v.tolist(),
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=self.phase,
            edge=self.edge,
        )
        return d


class RandomWaypointTrajectory(Trajectory):
    """Classic random-waypoint mobility model.

    The beacon flies at ``speed`` px/s towards a uniformly random waypoint
    inside the world (respecting ``margin``), optionally pauses for ``pause``
    seconds, then picks the next waypoint.  Waypoints come from a private RNG
    seeded with ``seed`` so the path is reproducible.

    Parameters
    ----------
    speed:
        Cruise speed in px/s.  If ``speed_jitter`` > 0 each leg uses a speed
        drawn uniformly from ``speed * [1 - jitter, 1 + jitter]``.
    pause:
        Dwell time in seconds at each waypoint.
    seed:
        Integer seed for the waypoint generator (default 0, so a standalone
        trajectory is reproducible).  Inside a :class:`~sim.scene.Scene` a
        per-beacon seed is derived from the scene's master seed whenever the
        JSON config does not set ``seed`` explicitly.  ``None`` = OS entropy
        (non-deterministic; avoid for experiments).
    """

    type_name = "random_waypoint"

    def __init__(
        self,
        start: Optional[Vec2] = None,
        speed: float = 200.0,
        pause: float = 0.0,
        speed_jitter: float = 0.0,
        seed: Optional[int] = 0,
        margin: Union[float, Sequence[float]] = 40.0,
    ) -> None:
        super().__init__(start, margin)
        self.speed = float(speed)
        self.pause = float(pause)
        self.speed_jitter = float(speed_jitter)
        self.seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self.target = np.zeros(2)
        self._leg_speed = self.speed
        self._pause_left = 0.0

    def _new_target(self) -> None:
        lo, hi = self._lo_hi()
        self.target = self._rng.uniform(lo, hi)
        j = self.speed_jitter
        self._leg_speed = self.speed * (self._rng.uniform(1 - j, 1 + j) if j > 0 else 1.0)

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        lo, hi = self._lo_hi()
        self.pos = np.clip(self.start, lo, hi).astype(np.float64)
        self._pause_left = 0.0
        self._new_target()

    def _advance(self, dt: float) -> None:
        remaining = dt
        # Loop so that a single large dt can traverse several legs exactly.
        while remaining > 1e-12:
            if self._pause_left > 0:
                used = min(self._pause_left, remaining)
                self._pause_left -= used
                remaining -= used
                continue
            delta = self.target - self.pos
            dist = float(np.hypot(*delta))
            if self._leg_speed <= 0:
                return
            t_arrive = dist / self._leg_speed
            if t_arrive <= remaining:
                self.pos = self.target.copy()
                remaining -= t_arrive
                self._pause_left = self.pause
                self._new_target()
            else:
                self.pos = self.pos + delta / dist * self._leg_speed * remaining
                remaining = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(speed=self.speed, pause=self.pause, speed_jitter=self.speed_jitter, seed=self.seed)
        return d


#: Registry mapping JSON ``"type"`` strings to classes.  Add new models here.
TRAJECTORY_TYPES: Dict[str, Type[Trajectory]] = {
    cls.type_name: cls
    for cls in (StaticTrajectory, LinearTrajectory, SinusoidalTrajectory, RandomWaypointTrajectory)
}


def trajectory_from_dict(cfg: Dict[str, Any]) -> Trajectory:
    """Build a trajectory from a config dict such as ``{"type": "linear", ...}``.

    Unknown keys raise ``TypeError`` (fail loudly on typos in JSON configs).
    """
    cfg = dict(cfg)
    kind = cfg.pop("type", None)
    if kind not in TRAJECTORY_TYPES:
        raise ValueError(f"Unknown trajectory type {kind!r}; expected one of {sorted(TRAJECTORY_TYPES)}")
    if "start" in cfg and cfg["start"] is not None:
        cfg["start"] = tuple(cfg["start"])
    if "velocity" in cfg:
        cfg["velocity"] = tuple(cfg["velocity"])
    return TRAJECTORY_TYPES[kind](**cfg)
