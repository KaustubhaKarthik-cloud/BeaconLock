"""
sim.scene -- the synthetic night-sky "world" in which BeaconLock beacons move.

The world is a large grayscale image (default 1920 x 1080) that stands in for
the field of regard of a mobile FSOC terminal.  It contains

1. a dark, slightly non-zero **background** (sky glow / sensor dark level),
2. a **static star field** (seeded, varied brightness and size; a power law
   gives many faint stars and a few bright ones -- realistic distractors for the
   beacon detector built in a later step),
3. one or more **beacons** -- bright 2-D Gaussian spots moving on configurable
   :mod:`sim.trajectories`,
4. per-frame **Gaussian sensor noise**, and
5. an optional dim ``SYNTHETIC SCENE`` watermark so that every rendered frame
   is clearly labelled as simulated data.

The virtual pan-tilt camera (``sim/camera.py``, step 2) will crop a viewport out
of :meth:`Scene.render`; camera-side disturbances (turbulence, vibration, blur)
live in ``sim/disturbances.py`` and are *not* applied here.

Typical use::

    from sim import Scene
    scene = Scene.from_json("configs/default_scene.json")
    for _ in range(150):           # 5 s at 30 FPS
        scene.step(1 / 30)
        frame = scene.render()     # uint8, shape (H, W)
        truth = scene.ground_truth()

Only NumPy and OpenCV are used -- no web framework imports -- so the module can
be frozen into a standalone executable with PyInstaller.

IMPORTANT: everything produced here is SYNTHETIC.  No real ISRO / mission data
is used, reproduced or imitated.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .trajectories import RandomWaypointTrajectory, Trajectory, trajectory_from_dict

__all__ = ["Beacon", "BeaconState", "Scene", "DEFAULT_CONFIG", "stamp_gaussian"]


# --------------------------------------------------------------------------- #
# Default configuration (mirrors configs/default_scene.json)                  #
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "default",
    "description": "SYNTHETIC scene: one beacon weaving across a night sky.",
    "synthetic": True,
    "seed": 42,
    "world": {"width": 1920, "height": 1080, "background_level": 6.0},
    "stars": {
        "count": 700,
        "min_brightness": 25.0,
        "max_brightness": 170.0,
        "min_sigma": 0.5,
        "max_sigma": 1.4,
        "power": 3.0,
    },
    "noise": {"sigma": 3.0},
    "watermark": {"enabled": True, "text": "SYNTHETIC SCENE", "intensity": 60.0},
    "beacons": [
        {
            "id": "beacon-1",
            "sigma": 4.0,
            "intensity": 255.0,
            "trajectory": {
                "type": "sinusoidal",
                "start": [200, 540],
                "velocity": [300, -40],
                "amplitude": 220,
                "frequency": 0.35,
                "phase": 0.0,
                "wrap": True,
            },
        }
    ],
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (lists replaced)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------- #
# Rendering helper                                                            #
# --------------------------------------------------------------------------- #
def stamp_gaussian(img: np.ndarray, x: float, y: float, sigma: float, peak: float) -> None:
    """Add a sub-pixel-accurate 2-D Gaussian spot to ``img`` **in place**.

    ``img`` must be a float32 2-D array.  The spot has value ``peak`` at its
    centre ``(x, y)`` and standard deviation ``sigma`` pixels.  Only a window of
    radius ``ceil(4 * sigma)`` is touched, so this is cheap even on large
    worlds.  Spots partially or fully outside the image are clipped safely.
    """
    if sigma <= 0 or peak == 0:
        return
    h, w = img.shape[:2]
    r = int(math.ceil(4.0 * sigma))
    x0, x1 = max(int(math.floor(x)) - r, 0), min(int(math.floor(x)) + r + 2, w)
    y0, y1 = max(int(math.floor(y)) - r, 0), min(int(math.floor(y)) + r + 2, h)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float32) - np.float32(x)
    ys = np.arange(y0, y1, dtype=np.float32) - np.float32(y)
    inv = np.float32(-0.5 / (sigma * sigma))
    # Separable Gaussian: outer product of 1-D profiles.
    patch = np.outer(np.exp(ys * ys * inv), np.exp(xs * xs * inv)) * np.float32(peak)
    img[y0:y1, x0:x1] += patch


# --------------------------------------------------------------------------- #
# Beacons                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class BeaconState:
    """Ground-truth snapshot of one beacon (world pixel coordinates)."""

    id: str
    x: float
    y: float
    vx: float
    vy: float
    sigma: float
    intensity: float

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Beacon:
    """An optical beacon: a bright Gaussian spot that follows a trajectory.

    Attributes
    ----------
    id:
        Human-readable identifier (e.g. ``"beacon-1"``).
    trajectory:
        Motion model; see :mod:`sim.trajectories`.
    sigma:
        Gaussian spot standard deviation in pixels (spot size).
    intensity:
        Peak added brightness (0-255 scale; values > 255 saturate the sensor).
    """

    id: str
    trajectory: Trajectory
    sigma: float = 4.0
    intensity: float = 255.0
    config: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def position(self) -> Tuple[float, float]:
        return self.trajectory.position

    def state(self) -> BeaconState:
        x, y = self.trajectory.position
        vx, vy = self.trajectory.velocity
        return BeaconState(self.id, x, y, vx, vy, self.sigma, self.intensity)


# --------------------------------------------------------------------------- #
# Scene                                                                       #
# --------------------------------------------------------------------------- #
class Scene:
    """Synthetic night-sky world containing moving optical beacons.

    Build it with :meth:`from_json` / :meth:`from_dict` (preferred) or directly
    from a config dict.  The scene is fully deterministic for a given ``seed``:
    star field, sensor noise sequence and random-waypoint paths all derive
    from it, and :meth:`reset` replays the run bit-for-bit.

    Public API
    ----------
    ``step(dt)``          advance simulated time by ``dt`` seconds
    ``render()``          full-world uint8 grayscale frame, shape ``(H, W)``
    ``render_bgr()``      same frame as 3-channel BGR (for overlays / video)
    ``ground_truth()``    list of :class:`BeaconState` (true beacon positions)
    ``beacon_positions()``  ``(N, 2)`` float array of ``[x, y]``
    ``reset()``           back to ``t = 0`` with identical randomness
    ``width``, ``height``, ``t``, ``frame_index``, ``beacons``, ``config``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = _deep_merge(DEFAULT_CONFIG, config or {})
        if not cfg.get("beacons"):
            raise ValueError("Scene config must define at least one beacon")
        self.config: Dict[str, Any] = cfg
        self.name: str = str(cfg.get("name", "unnamed"))
        self.seed: int = int(cfg.get("seed", 0))

        world = cfg["world"]
        self.width: int = int(world["width"])
        self.height: int = int(world["height"])
        if self.width < 16 or self.height < 16:
            raise ValueError("World must be at least 16x16 pixels")
        self.background_level: float = float(world.get("background_level", 6.0))
        self.noise_sigma: float = float(cfg["noise"].get("sigma", 0.0))

        # Independent, reproducible random streams derived from the master seed.
        ss = np.random.SeedSequence(self.seed)
        self._star_seed, self._noise_seed, self._traj_seed = ss.spawn(3)

        self.beacons: List[Beacon] = [self._make_beacon(i, b) for i, b in enumerate(cfg["beacons"])]
        self._static_layer = self._build_static_layer()
        self.reset()

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Scene":
        """Create a scene from a (possibly partial) config dict.

        Missing keys fall back to :data:`DEFAULT_CONFIG`.
        """
        return cls(config)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Scene":
        """Create a scene from a JSON file (see ``configs/*.json``)."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def to_dict(self) -> Dict[str, Any]:
        """Return the fully-resolved config (defaults merged in)."""
        return copy.deepcopy(self.config)

    # ------------------------------------------------------------ build steps
    def _make_beacon(self, index: int, bcfg: Dict[str, Any]) -> Beacon:
        tcfg = dict(bcfg.get("trajectory", {"type": "static"}))
        # Give random-waypoint trajectories a deterministic per-beacon seed
        # derived from the scene seed unless one is given explicitly.
        if tcfg.get("type") == RandomWaypointTrajectory.type_name and tcfg.get("seed") is None:
            child = np.random.SeedSequence(self._traj_seed.entropy, spawn_key=(index,))
            tcfg["seed"] = int(child.generate_state(1)[0])
        traj = trajectory_from_dict(tcfg).bind(self.width, self.height)
        return Beacon(
            id=str(bcfg.get("id", f"beacon-{index + 1}")),
            trajectory=traj,
            sigma=float(bcfg.get("sigma", 4.0)),
            intensity=float(bcfg.get("intensity", 255.0)),
            config=dict(bcfg),
        )

    def _build_static_layer(self) -> np.ndarray:
        """Pre-render background + stars + watermark once (float32, H x W).

        Stars never move in the world frame, so they are baked into this layer;
        each call to :meth:`render` only adds beacons and fresh noise on top.
        """
        layer = np.full((self.height, self.width), self.background_level, dtype=np.float32)

        s = self.config["stars"]
        rng = np.random.default_rng(self._star_seed)
        n = int(s.get("count", 0))
        if n > 0:
            xs = rng.uniform(0, self.width, n)
            ys = rng.uniform(0, self.height, n)
            # Power-law brightness: u**power concentrates most stars near the
            # faint end, leaving only a handful of bright distractors.
            u = rng.uniform(0, 1, n) ** float(s.get("power", 3.0))
            bright = s["min_brightness"] + (s["max_brightness"] - s["min_brightness"]) * u
            # Brighter stars look slightly larger (PSF wings).
            sig = s["min_sigma"] + (s["max_sigma"] - s["min_sigma"]) * np.sqrt(u) * rng.uniform(0.7, 1.0, n)
            for x, y, b, sg in zip(xs, ys, bright, sig):
                stamp_gaussian(layer, x, y, float(sg), float(b))

        wm = self.config.get("watermark", {})
        if wm.get("enabled", True):
            text = str(wm.get("text", "SYNTHETIC SCENE"))
            scale = max(self.width / 1920.0, 0.35)
            thick = max(1, int(round(2 * scale)))
            mask = np.zeros_like(layer, dtype=np.uint8)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            org = (self.width - tw - int(16 * scale), self.height - int(16 * scale))
            cv2.putText(mask, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
            # Kept dim on purpose so later beacon detectors do not latch onto it.
            layer += mask.astype(np.float32) * (float(wm.get("intensity", 60.0)) / 255.0)
        return layer

    # ---------------------------------------------------------------- runtime
    def reset(self) -> None:
        """Reset time, trajectories and the noise RNG to their initial state."""
        self.t: float = 0.0
        self.frame_index: int = 0
        self._noise_rng = np.random.default_rng(self._noise_seed)
        for b in self.beacons:
            b.trajectory.reset()

    def step(self, dt: float) -> List[BeaconState]:
        """Advance the world by ``dt`` seconds; returns the new ground truth."""
        for b in self.beacons:
            b.trajectory.step(dt)
        self.t += dt
        self.frame_index += 1
        return self.ground_truth()

    def render(self, noise: bool = True) -> np.ndarray:
        """Render the full world at the current time.

        Parameters
        ----------
        noise:
            Add per-frame Gaussian sensor noise (``noise.sigma`` in the config).
            Each call draws a new noise realisation from the seeded stream.

        Returns
        -------
        numpy.ndarray
            ``uint8`` array of shape ``(height, width)`` -- a single-channel
            (monochrome) sensor image, like a typical beacon camera.
        """
        img = self._static_layer.copy()
        for b in self.beacons:
            x, y = b.trajectory.position
            stamp_gaussian(img, x, y, b.sigma, b.intensity)
        if noise and self.noise_sigma > 0:
            img += self._noise_rng.standard_normal(img.shape, dtype=np.float32) * np.float32(self.noise_sigma)
        np.clip(img, 0, 255, out=img)
        return img.astype(np.uint8)

    def render_region(self, x0: int, y0: int, w: int, h: int, noise: bool = True) -> np.ndarray:
        """Render only the world window ``[y0:y0+h, x0:x0+w]`` (uint8, ``(h, w)``).

        Pixel-identical (apart from the noise realisation) to cropping
        :meth:`render`, but ~10x cheaper because only the window is copied,
        stamped and noised.  Used by :class:`sim.camera.PanTiltCamera`.
        The window must lie inside the world.
        """
        x0, y0, w, h = int(x0), int(y0), int(w), int(h)
        if x0 < 0 or y0 < 0 or x0 + w > self.width or y0 + h > self.height or w <= 0 or h <= 0:
            raise ValueError(f"region ({x0},{y0},{w},{h}) outside world {self.width}x{self.height}")
        img = self._static_layer[y0:y0 + h, x0:x0 + w].copy()
        for b in self.beacons:
            x, y = b.trajectory.position
            stamp_gaussian(img, x - x0, y - y0, b.sigma, b.intensity)
        if noise and self.noise_sigma > 0:
            img += self._noise_rng.standard_normal(img.shape, dtype=np.float32) * np.float32(self.noise_sigma)
        np.clip(img, 0, 255, out=img)
        return img.astype(np.uint8)

    @property
    def static_layer(self) -> np.ndarray:
        """Read-only view of the pre-rendered background+stars layer (float32)."""
        v = self._static_layer.view()
        v.flags.writeable = False
        return v

    def render_bgr(self, noise: bool = True) -> np.ndarray:
        """:meth:`render` converted to 3-channel BGR (``(H, W, 3)`` uint8)."""
        return cv2.cvtColor(self.render(noise=noise), cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------ ground truth
    def ground_truth(self) -> List[BeaconState]:
        """True state of every beacon (position/velocity in world pixels)."""
        return [b.state() for b in self.beacons]

    def beacon_positions(self) -> np.ndarray:
        """``(N, 2)`` float64 array of true ``[x, y]`` beacon positions."""
        return np.array([b.position for b in self.beacons], dtype=np.float64).reshape(-1, 2)

    @property
    def size(self) -> Tuple[int, int]:
        """World size as ``(width, height)``."""
        return self.width, self.height

    def __repr__(self) -> str:
        return (
            f"Scene(name={self.name!r}, size={self.width}x{self.height}, beacons={len(self.beacons)}, "
            f"t={self.t:.3f}s, frame={self.frame_index}, SYNTHETIC)"
        )
