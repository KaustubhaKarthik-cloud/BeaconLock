"""
tracking.tracker -- Kalman filter + acquisition/lock state machine.

State machine (coarse PAT logic)::

      start()
    IDLE ──────► SEARCHING ──detection──► ACQUIRING ──N_acq hits in M_acq──► LOCKED
                    ▲  (spiral scan)          │                                │
                    │                         │ too many misses                │ N_lose misses
                    │◄────────────────────────┘                                │ in M_lose frames
                    │                                                          ▼
                    └──────────── lost_timeout expired ◄──────────────────── LOST
                                                           detection near    │
                                              ACQUIRING ◄──── prediction ────┘  (re-acquire)

* **SEARCHING** -- the aim point follows an Archimedean spiral centred on the
  last known beacon position (or the configured search centre / field-of-regard
  centre).  Ring spacing is a fraction of the FOV so consecutive rings overlap.
* **ACQUIRING** -- the Kalman filter is initialised on the first detection; the
  camera slews to the estimate.  Lock is declared after ``acquire_n`` hits in
  the last ``acquire_m`` frames (N-of-M confirmation) **and** once the
  estimate is within ``lock_radius_px`` of the boresight (beacon centred).
* **LOCKED** -- closed-loop tracking.  ``lose_n`` misses within the last
  ``lose_m`` frames -> LOST.
* **LOST** -- coast on the Kalman prediction (velocity decays); a detection
  near the prediction re-enters ACQUIRING; after ``lost_timeout_s`` fall back
  to SEARCHING around the last estimate.

All positions are in **world pixels** (the camera's boresight is known from
the gimbal "encoders"), which makes the filter independent of camera motion.
Pure NumPy; no web-framework imports.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["TrackState", "KalmanFilter2D", "SpiralScan", "TrackerConfig", "TrackerOutput", "BeaconTracker"]


class TrackState(str, Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    LOST = "LOST"


# --------------------------------------------------------------------------- #
# Kalman filter                                                                #
# --------------------------------------------------------------------------- #
class KalmanFilter2D:
    """Constant-velocity Kalman filter, state ``[x, y, vx, vy]``.

    Parameters
    ----------
    process_accel:
        Std-dev of the (white) target acceleration, px/s^2 -- how manoeuvrable
        the beacon is assumed to be.
    meas_sigma:
        Std-dev of a centroid measurement, px (turbulence + noise).
    """

    def __init__(self, process_accel: float = 400.0, meas_sigma: float = 3.0) -> None:
        self.q = float(process_accel)
        self.r = float(meas_sigma)
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * self.r ** 2
        self.initialized = False

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, pos_var: float = 25.0,
             vel_var: float = 400.0 ** 2) -> None:
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var])
        self.initialized = True

    def predict(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # Discrete white-noise acceleration model.
        g = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = g @ g.T * self.q ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def update(self, zx: float, zy: float) -> np.ndarray:
        z = np.array([zx, zy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    def innovation_distance(self, zx: float, zy: float) -> float:
        """Euclidean distance between a measurement and the predicted position."""
        return float(math.hypot(zx - self.x[0], zy - self.x[1]))

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def pos_std(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1])))


# --------------------------------------------------------------------------- #
# Spiral search                                                                #
# --------------------------------------------------------------------------- #
class SpiralScan:
    """Archimedean spiral ``r = b * theta`` traversed at constant linear speed.

    Parameters
    ----------
    center:
        Spiral centre (world px).
    spacing:
        Distance between successive rings, px (``b = spacing / 2 pi``).
    speed:
        Linear scan speed along the spiral, px/s.
    max_radius:
        When exceeded, the spiral restarts from the centre.
    """

    def __init__(self, center: Tuple[float, float], spacing: float, speed: float, max_radius: float) -> None:
        self.center = np.array(center, dtype=np.float64)
        self.b = spacing / (2 * math.pi)
        self.speed = float(speed)
        self.max_radius = float(max_radius)
        self.theta = 0.0
        self.restarts = 0

    def point(self, theta: Optional[float] = None) -> np.ndarray:
        th = self.theta if theta is None else theta
        r = self.b * th
        return self.center + r * np.array([math.cos(th), math.sin(th)])

    def advance(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Advance by ``dt`` seconds; returns ``(point, velocity)``."""
        prev = self.point()
        r = self.b * self.theta
        # ds = sqrt(r^2 + b^2) dtheta  ->  dtheta = v dt / sqrt(r^2 + b^2)
        self.theta += self.speed * dt / math.sqrt(r * r + self.b * self.b)
        if self.b * self.theta > self.max_radius:
            self.theta = 0.0
            self.restarts += 1
        p = self.point()
        vel = (p - prev) / dt if dt > 0 else np.zeros(2)
        if np.hypot(*vel) > 3 * self.speed:  # restart jump: don't feed-forward it
            vel = np.zeros(2)
        return p, vel


# --------------------------------------------------------------------------- #
# Tracker                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class TrackerConfig:
    acquire_n: int = 4          # hits needed ...
    acquire_m: int = 6          # ... within this many frames to declare LOCKED
    lose_n: int = 6             # misses ...
    lose_m: int = 10            # ... within this many frames to declare LOST
    lost_timeout_s: float = 1.5  # coast time in LOST before falling back to SEARCHING
    gate_px: float = 90.0       # max innovation for a detection to be associated
    lock_radius_px: float = 40.0  # LOCKED also requires |estimate - boresight| below this
    lost_gate_growth_px_s: float = 200.0  # gate grows while coasting
    search_speed: float = 1100.0  # spiral scan speed, px/s
    search_spacing_frac: float = 0.7  # ring spacing as a fraction of min(FOV w, h)
    search_center: Optional[Tuple[float, float]] = None  # None -> point given to start() / world centre
    lead_s: float = 0.0          # aim this far ahead of the estimate (latency comp.)
    process_accel: float = 2000.0  # px/s^2; tuned so the velocity estimate follows manoeuvres
    meas_sigma: float = 3.0
    coast_velocity_decay: float = 0.5  # per second, while LOST

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TrackerConfig":
        d = dict(d or {})
        if d.get("search_center") is not None:
            d["search_center"] = tuple(d["search_center"])
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if out["search_center"] is not None:
            out["search_center"] = list(out["search_center"])
        return out


@dataclass
class TrackerOutput:
    """What the tracker tells the controller (and the UI) for one frame."""

    state: TrackState
    aim: Optional[Tuple[float, float]]           # world px the boresight should point at
    aim_velocity: Tuple[float, float]            # px/s feed-forward
    estimate: Optional[Tuple[float, float]]      # Kalman position (world px), if any
    estimate_velocity: Optional[Tuple[float, float]]
    detection_used: bool
    state_changed: bool
    events: List[str] = field(default_factory=list)


class BeaconTracker:
    """Kalman tracker with the IDLE/SEARCHING/ACQUIRING/LOCKED/LOST machine.

    Parameters
    ----------
    world_size:
        ``(width, height)`` of the world, px.
    fov_size:
        ``(width, height)`` of the camera, px (sets spiral ring spacing).
    config:
        :class:`TrackerConfig`.
    """

    def __init__(self, world_size: Tuple[int, int], fov_size: Tuple[int, int],
                 config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self.world_size = world_size
        self.fov_size = fov_size
        self.reset()

    # ------------------------------------------------------------- control
    def reset(self) -> None:
        self.state = TrackState.IDLE
        c = self.config
        self.kf = KalmanFilter2D(c.process_accel, c.meas_sigma)
        self.window: Deque[bool] = deque()
        self.spiral: Optional[SpiralScan] = None
        self.time_in_state = 0.0
        self.t = 0.0
        self.lock_count = 0          # number of times LOCKED was entered
        self.loss_count = 0          # number of times LOST was entered
        self.reacquire_count = 0     # LOST -> ACQUIRING transitions
        self.first_lock_time: Optional[float] = None
        self.history: List[Tuple[float, str]] = []

    def start(self, from_point: Optional[Tuple[float, float]] = None) -> None:
        """Begin autonomous acquisition (IDLE -> SEARCHING)."""
        if self.state == TrackState.IDLE:
            self._enter(TrackState.SEARCHING, search_center=from_point)

    def stop(self) -> None:
        """Go back to IDLE (camera holds position)."""
        self._enter(TrackState.IDLE)

    # ----------------------------------------------------------- internals
    def _enter(self, state: TrackState, search_center: Optional[Tuple[float, float]] = None) -> None:
        c = self.config
        self.state = state
        self.time_in_state = 0.0
        self.history.append((self.t, state.value))
        if state == TrackState.SEARCHING:
            center = c.search_center or search_center or (self.world_size[0] / 2, self.world_size[1] / 2)
            W, H = self.world_size
            max_r = math.hypot(W, H) * 0.6
            self.spiral = SpiralScan(center, c.search_spacing_frac * min(self.fov_size), c.search_speed, max_r)
            self.kf.initialized = False
        elif state == TrackState.ACQUIRING:
            self.window = deque(maxlen=c.acquire_m)
        elif state == TrackState.LOCKED:
            self.window = deque(maxlen=c.lose_m)
            self.lock_count += 1
            if self.first_lock_time is None:
                self.first_lock_time = self.t
        elif state == TrackState.LOST:
            self.loss_count += 1

    def _aim_from_kf(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x, y = self.kf.position
        vx, vy = self.kf.velocity
        L = self.config.lead_s
        return (x + vx * L, y + vy * L), (vx, vy)

    # --------------------------------------------------------------- update
    def update(self, measurement: Optional[Tuple[float, float]], dt: float,
               boresight: Optional[Tuple[float, float]] = None) -> TrackerOutput:
        """Advance one frame.

        Parameters
        ----------
        measurement:
            Detected beacon centroid in **world px** (camera origin + detection),
            or ``None`` when the detector found nothing.
        dt:
            Frame interval, s.
        boresight:
            Current camera pointing (world px).  If given, LOCKED additionally
            requires the estimate to be within ``lock_radius_px`` of it.
        """
        c = self.config
        events: List[str] = []
        used = False
        if self.state == TrackState.IDLE:
            return TrackerOutput(self.state, None, (0.0, 0.0), None, None, False, False, events)

        # Tracker time only runs while active, so first_lock_time == acquisition
        # time measured from start().
        self.t += dt
        self.time_in_state += dt
        prev_state = self.state

        if self.kf.initialized:
            self.kf.predict(dt)

        def associate() -> bool:
            if measurement is None or not self.kf.initialized:
                return False
            gate = c.gate_px
            if self.state == TrackState.LOST:
                gate += c.lost_gate_growth_px_s * self.time_in_state
            return self.kf.innovation_distance(*measurement) <= gate

        aim: Optional[Tuple[float, float]] = None
        aim_v: Tuple[float, float] = (0.0, 0.0)

        if self.state == TrackState.SEARCHING:
            if measurement is not None:
                self.kf.init(*measurement)
                used = True
                self._enter(TrackState.ACQUIRING)
                self.window.append(True)
                events.append("beacon detected")
            else:
                assert self.spiral is not None
                p, v = self.spiral.advance(dt)
                aim, aim_v = (float(p[0]), float(p[1])), (float(v[0]), float(v[1]))

        elif self.state == TrackState.ACQUIRING:
            hit = associate()
            if hit:
                self.kf.update(*measurement)  # type: ignore[misc]
                used = True
            self.window.append(hit)
            hits = sum(self.window)
            misses = len(self.window) - hits
            centred = boresight is None or math.hypot(self.kf.position[0] - boresight[0],
                                                       self.kf.position[1] - boresight[1]) <= c.lock_radius_px
            if hits >= c.acquire_n and centred:
                self._enter(TrackState.LOCKED)
                events.append("lock acquired")
            elif misses > c.acquire_m - c.acquire_n:
                center = self.kf.position if self.kf.initialized else None
                self._enter(TrackState.SEARCHING, search_center=center)
                events.append("acquisition failed")

        elif self.state == TrackState.LOCKED:
            hit = associate()
            if hit:
                self.kf.update(*measurement)  # type: ignore[misc]
                used = True
            self.window.append(hit)
            if len(self.window) - sum(self.window) >= c.lose_n:
                self._enter(TrackState.LOST)
                events.append("lock lost")

        elif self.state == TrackState.LOST:
            if associate():
                self.kf.update(*measurement)  # type: ignore[misc]
                used = True
                self.reacquire_count += 1
                self._enter(TrackState.ACQUIRING)
                self.window.append(True)
                events.append("re-acquiring")
            else:
                decay = math.exp(-c.coast_velocity_decay * dt)
                self.kf.x[2:] *= decay
                if self.time_in_state >= c.lost_timeout_s:
                    self._enter(TrackState.SEARCHING, search_center=self.kf.position)
                    events.append("search restarted")

        if aim is None and self.state != TrackState.SEARCHING and self.kf.initialized:
            aim, aim_v = self._aim_from_kf()
        elif aim is None and self.state == TrackState.SEARCHING and self.spiral is not None:
            p = self.spiral.point()
            aim = (float(p[0]), float(p[1]))

        est = self.kf.position if self.kf.initialized else None
        est_v = self.kf.velocity if self.kf.initialized else None
        return TrackerOutput(self.state, aim, aim_v, est, est_v, used, self.state != prev_state, events)
