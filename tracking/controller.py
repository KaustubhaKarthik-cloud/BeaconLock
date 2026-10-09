"""
tracking.controller -- PID pointing controller (pixel error -> pan/tilt).

The tracker supplies an aim point; the controller turns the **pixel error**
(aim point minus boresight, i.e. beacon offset from the frame centre) into a
slew-rate command for each gimbal axis:

    rate = Kp * e + Ki * integral(e) + Kd * de/dt + Kff * v_target

``v_target`` is the target velocity estimated by the Kalman filter (feed-
forward), which removes the steady-state lag a pure PID shows on a moving
beacon.  The commanded displacement for this step is ``rate * dt`` px, which is
then passed to :meth:`sim.camera.PanTiltCamera.move` (where the slew-rate
limit and mechanical clamps apply).

Includes integrator anti-windup (clamp + conditional integration: the
integrator only runs when the error is already small, so large acquisition
slews do not wind it up) and a first-order low-pass filter on
the derivative term.  Pure NumPy; no web-framework imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["PIDConfig", "PID", "PanTiltController"]


@dataclass
class PIDConfig:
    kp: float = 9.0           # 1/s : px/s of slew per px of error
    ki: float = 0.6           # 1/s^2 (small: feed-forward does the heavy lifting)
    kd: float = 0.15          # s
    kff: float = 1.0          # velocity feed-forward gain (0 disables)
    max_rate: float = 1500.0  # px/s output clamp per axis
    i_limit: float = 300.0    # anti-windup clamp on the integral term output, px/s
    i_zone_px: float = 25.0   # integrate only when |error| < this (conditional integration)
    d_filter_hz: float = 8.0  # derivative low-pass cut-off

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PIDConfig":
        return cls(**dict(d or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PID:
    """Vector PID (one independent loop per element of the error vector)."""

    def __init__(self, kp: float, ki: float, kd: float, out_limit: float = np.inf,
                 i_limit: float = np.inf, d_filter_hz: float = 0.0, dims: int = 2,
                 i_zone: float = np.inf) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_zone = i_zone
        self.out_limit, self.i_limit, self.d_filter_hz = out_limit, i_limit, d_filter_hz
        self.dims = dims
        self.reset()

    def reset(self) -> None:
        """Clear integrator and derivative memory (call on mode changes)."""
        self.integral = np.zeros(self.dims)
        self.prev_error: Optional[np.ndarray] = None
        self.d_state = np.zeros(self.dims)

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        e = np.asarray(error, dtype=np.float64)
        if dt <= 0:
            return np.clip(self.kp * e, -self.out_limit, self.out_limit)
        self.integral += np.where(np.abs(e) < self.i_zone, e, 0.0) * dt
        if self.ki > 0:  # anti-windup: bound the integral's contribution
            lim = self.i_limit / self.ki
            self.integral = np.clip(self.integral, -lim, lim)
        d_raw = np.zeros(self.dims) if self.prev_error is None else (e - self.prev_error) / dt
        if self.d_filter_hz > 0:
            alpha = dt / (dt + 1.0 / (2 * np.pi * self.d_filter_hz))
            self.d_state += alpha * (d_raw - self.d_state)
        else:
            self.d_state = d_raw
        self.prev_error = e
        out = self.kp * e + self.ki * self.integral + self.kd * self.d_state
        return np.clip(out, -self.out_limit, self.out_limit)


class PanTiltController:
    """Two-axis pointing controller: pixel error -> ``(pan_dx, tilt_dy)`` px."""

    def __init__(self, config: Optional[PIDConfig] = None) -> None:
        self.config = config or PIDConfig()
        c = self.config
        self.pid = PID(c.kp, c.ki, c.kd, out_limit=np.inf, i_limit=c.i_limit, d_filter_hz=c.d_filter_hz,
                       i_zone=c.i_zone_px)
        self.last_rate = np.zeros(2)

    def reset(self) -> None:
        self.pid.reset()
        self.last_rate = np.zeros(2)

    def update(
        self,
        error_px: Tuple[float, float],
        dt: float,
        target_velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[float, float]:
        """Compute this step's pan/tilt displacement in px.

        Parameters
        ----------
        error_px:
            ``(ex, ey)`` = aim point minus boresight, px (+x -> pan right).
        dt:
            Step length in seconds.
        target_velocity:
            Estimated target velocity in px/s (feed-forward).
        """
        c = self.config
        rate = self.pid.update(np.asarray(error_px, float), dt) + c.kff * np.asarray(target_velocity, float)
        rate = np.clip(rate, -c.max_rate, c.max_rate)
        self.last_rate = rate
        step = rate * dt
        # Never command more than the full correction (error + feed-forward motion)
        # in a single step: prevents overshoot with high gains / low frame rates.
        full = np.asarray(error_px, float) + c.kff * np.asarray(target_velocity, float) * dt
        for k in range(2):
            if step[k] * full[k] > 0 and abs(step[k]) > abs(full[k]):
                step[k] = full[k]
        return float(step[0]), float(step[1])
