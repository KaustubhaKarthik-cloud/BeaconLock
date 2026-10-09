"""
sim.disturbances -- image-plane disturbances for the virtual camera.

Four independent effects, each with an ``intensity`` in ``[0, 100]``
(0 = off, 100 = severe).  They are applied to the camera frame in physical
order: optics/atmosphere first, then platform motion, then the sensor.

1. **Atmospheric turbulence** (Perlin-noise warp).
   Smooth, spatially-correlated pixel displacement field (local "seeing"
   wobble) + a global low-frequency image shift (beam wander / angle-of-arrival
   fluctuation) + mild scintillation (intensity flicker) and blur.  The noise
   field is advected across the image over time (Taylor frozen-flow
   hypothesis) so the distortion evolves smoothly frame to frame.
   At 100: ~6 px local warp, ~14 px beam wander, +/-35 % scintillation.
2. **Platform vibration** (sinusoidal jitter): sum of two sinusoids per axis
   (structural modes ~9 Hz and ~23 Hz by default); up to ~10 px at 100.
3. **Motion blur**: linear blur kernel up to ~18 px long, oriented along the
   current camera slew direction (or the vibration direction when static).
4. **Gaussian sensor noise**: additive read noise, sigma up to 20 DN at 100.

These are *synthetic, qualitative* models for exercising the PAT loop; they
are not calibrated against any real link or ISRO data.

No web-framework imports (UI-independent core).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

__all__ = ["Perlin2D", "DisturbanceConfig", "Disturbances"]


# --------------------------------------------------------------------------- #
# Perlin noise                                                                 #
# --------------------------------------------------------------------------- #
class Perlin2D:
    """Vectorised classic (improved) 2-D Perlin gradient noise.

    ``noise(x, y)`` accepts NumPy arrays of float coordinates (lattice units)
    and returns values in roughly ``[-1, 1]``.  Deterministic for a seed.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(256)
        self.p = np.concatenate([perm, perm]).astype(np.int64)
        ang = rng.uniform(0, 2 * np.pi, 256)
        self.g = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xf0, yf0 = np.floor(x), np.floor(y)
        xi = xf0.astype(np.int64) & 255
        yi = yf0.astype(np.int64) & 255
        xf, yf = x - xf0, y - yf0
        u, v = self._fade(xf), self._fade(yf)
        p, g = self.p, self.g

        def grad(h: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
            gg = g[h & 255]
            return gg[..., 0] * dx + gg[..., 1] * dy

        n00 = grad(p[p[xi] + yi], xf, yf)
        n10 = grad(p[p[xi + 1] + yi], xf - 1, yf)
        n01 = grad(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = grad(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        return (nx0 + v * (nx1 - nx0)) * math.sqrt(2.0)

    def fbm(self, x: np.ndarray, y: np.ndarray, octaves: int = 2, gain: float = 0.5) -> np.ndarray:
        """Fractal Brownian motion: sum of ``octaves`` noise layers."""
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total = total + amp * self.noise(x * freq, y * freq)
            norm += amp
            amp *= gain
            freq *= 2.0
        return total / norm


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class DisturbanceConfig:
    """Intensities (0-100) and tuning knobs for :class:`Disturbances`."""

    turbulence: float = 0.0
    vibration: float = 0.0
    noise: float = 0.0
    blur: float = 0.0
    seed: int = 7
    # Scale factors at intensity 100 (documented physical-ish meaning):
    max_warp_px: float = 6.0          # local turbulence displacement amplitude
    max_wander_px: float = 14.0       # global beam-wander amplitude
    max_scintillation: float = 0.35   # +/- relative intensity flicker
    max_vibration_px: float = 10.0    # platform jitter amplitude
    vibration_hz: Tuple[float, float] = (9.0, 23.0)
    max_blur_px: float = 18.0         # motion-blur kernel length
    max_noise_sigma: float = 20.0     # additive Gaussian read noise (DN)
    turbulence_cell_px: float = 90.0  # spatial correlation length of the warp
    wind_px_s: Tuple[float, float] = (60.0, 25.0)  # frozen-flow advection speed

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DisturbanceConfig":
        d = dict(d or {})
        for k in ("vibration_hz", "wind_px_s"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["vibration_hz"] = list(self.vibration_hz)
        out["wind_px_s"] = list(self.wind_px_s)
        return out


def _level(v: float) -> float:
    """Clamp an intensity to [0, 100] and map to [0, 1]."""
    return min(max(float(v), 0.0), 100.0) / 100.0


class Disturbances:
    """Apply turbulence, vibration, motion blur and sensor noise to frames.

    Usage::

        dist = Disturbances(DisturbanceConfig(turbulence=50, vibration=30))
        out = dist.apply(frame, t=sim_time, camera_velocity=(vx, vy))

    Intensities can be changed live with :meth:`set` (e.g. from UI sliders).
    Output is deterministic for a given seed and call sequence.
    """

    def __init__(self, config: Optional[DisturbanceConfig] = None) -> None:
        self.config = config or DisturbanceConfig()
        self._grid_shape: Optional[Tuple[int, int]] = None
        self.reset()
        #: Total global image shift (dx, dy) applied on the last call, px.
        self.last_shift = (0.0, 0.0)

    def reset(self) -> None:
        """Re-seed all random streams (replays identical disturbances)."""
        c = self.config
        ss = np.random.SeedSequence(c.seed)
        s_warp, s_noise, s_vib = ss.spawn(3)
        self._perlin_x = Perlin2D(int(s_warp.generate_state(1)[0]))
        self._perlin_y = Perlin2D(int(s_warp.generate_state(2)[1]))
        self._noise_rng = np.random.default_rng(s_noise)
        self._vib_phase = np.random.default_rng(s_vib).uniform(0, 2 * np.pi, 4)

    def set(self, **intensities: float) -> None:
        """Update intensities live, e.g. ``set(turbulence=80, blur=0)``."""
        for k, v in intensities.items():
            if k not in ("turbulence", "vibration", "noise", "blur"):
                raise KeyError(f"unknown disturbance {k!r}")
            setattr(self.config, k, min(max(float(v), 0.0), 100.0))

    @property
    def intensities(self) -> Dict[str, float]:
        c = self.config
        return {"turbulence": c.turbulence, "vibration": c.vibration, "noise": c.noise, "blur": c.blur}

    # ----------------------------------------------------------- components
    def vibration_offset(self, t: float) -> np.ndarray:
        """Platform jitter displacement ``(dx, dy)`` in px at time ``t``."""
        a = _level(self.config.vibration) * self.config.max_vibration_px
        if a == 0:
            return np.zeros(2)
        f1, f2 = self.config.vibration_hz
        ph = self._vib_phase
        dx = 0.7 * math.sin(2 * math.pi * f1 * t + ph[0]) + 0.3 * math.sin(2 * math.pi * f2 * t + ph[1])
        dy = 0.7 * math.sin(2 * math.pi * f1 * 1.13 * t + ph[2]) + 0.3 * math.sin(2 * math.pi * f2 * 0.91 * t + ph[3])
        return a * np.array([dx, dy])

    def turbulence_wander(self, t: float) -> np.ndarray:
        """Global beam-wander shift ``(dx, dy)`` in px (slow, ~0.3-1 Hz)."""
        a = _level(self.config.turbulence) * self.config.max_wander_px
        if a == 0:
            return np.zeros(2)
        return a * np.array([
            float(self._perlin_x.fbm(np.array(t * 0.7), np.array(3.3))),
            float(self._perlin_y.fbm(np.array(t * 0.7), np.array(7.7))),
        ])

    def _ensure_grid(self, h: int, w: int) -> None:
        if self._grid_shape == (h, w):
            return
        self._grid_shape = (h, w)
        self._gx, self._gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        # Low-resolution lattice on which the Perlin field is evaluated (cheap),
        # then bilinearly up-sampled to full resolution.
        self._lw, self._lh = max(4, w // 16), max(4, h // 16)
        lx, ly = np.meshgrid(np.linspace(0, w, self._lw), np.linspace(0, h, self._lh))
        self._lx, self._ly = lx, ly

    # ---------------------------------------------------------------- apply
    def apply(
        self,
        frame: np.ndarray,
        t: float,
        camera_velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Return a disturbed copy of ``frame`` (uint8 grayscale ``(H, W)``).

        Parameters
        ----------
        frame:
            Clean camera frame from :meth:`sim.camera.PanTiltCamera.get_frame`.
        t:
            Simulation time in seconds (drives the time-varying effects).
        camera_velocity:
            Boresight slew velocity in px/s; orients the motion-blur kernel.
        """
        c = self.config
        lt, lv, lb, ln = _level(c.turbulence), _level(c.vibration), _level(c.blur), _level(c.noise)
        h, w = frame.shape[:2]
        out = frame.astype(np.float32)

        # ---- 1+2. geometric warp: turbulence (local + wander) + vibration ----
        shift = self.vibration_offset(t) + self.turbulence_wander(t)
        self.last_shift = (float(shift[0]), float(shift[1]))
        if lt > 0 or lv > 0:
            self._ensure_grid(h, w)
            map_x = self._gx - np.float32(shift[0])
            map_y = self._gy - np.float32(shift[1])
            if lt > 0:
                cell = c.turbulence_cell_px
                wx, wy = c.wind_px_s
                X = (self._lx + wx * t) / cell
                Y = (self._ly + wy * t) / cell
                amp = lt * c.max_warp_px
                dx = (amp * self._perlin_x.fbm(X, Y)).astype(np.float32)
                dy = (amp * self._perlin_y.fbm(X + 17.1, Y - 5.3)).astype(np.float32)
                map_x += cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
                map_y += cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if lt > 0:
            # Scintillation: smooth multiplicative flicker of signal above the background.
            X = (self._lx - 40.0 * t) / (c.turbulence_cell_px * 1.5)
            Y = (self._ly + 30.0 * t) / (c.turbulence_cell_px * 1.5)
            gain = 1.0 + lt * c.max_scintillation * self._perlin_x.fbm(X + 50.0, Y + 50.0)
            gain = cv2.resize(gain.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            bg = np.float32(np.median(out[::8, ::8]))
            out = bg + (out - bg) * gain
            # Seeing blur (slight PSF broadening).
            s = 1.2 * lt
            if s > 0.3:
                out = cv2.GaussianBlur(out, (0, 0), s)

        # ---- 3. motion blur along slew (or vibration) direction ----
        if lb > 0:
            length = lb * c.max_blur_px
            vx, vy = camera_velocity
            if math.hypot(vx, vy) < 1.0:  # static camera: blur along vibration velocity
                d = self.vibration_offset(t + 1e-3) - self.vibration_offset(t)
                vx, vy = (d[0], d[1]) if np.hypot(*d) > 1e-9 else (1.0, 0.0)
            out = self._motion_blur(out, length, math.atan2(vy, vx))

        # ---- 4. additive Gaussian sensor noise ----
        if ln > 0:
            out = out + self._noise_rng.standard_normal(out.shape, dtype=np.float32) * np.float32(ln * c.max_noise_sigma)

        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8)

    @staticmethod
    def _motion_blur(img: np.ndarray, length: float, angle: float) -> np.ndarray:
        """Convolve with a normalised line kernel of ``length`` px at ``angle`` rad."""
        if length < 1.5:
            return img
        k = int(math.ceil(length)) | 1
        kernel = np.zeros((k, k), np.float32)
        c = k // 2
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
        kernel /= max(kernel.sum(), 1e-6)
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
