"""
web.runner -- drives a SimulationEngine in real time on a background thread.

Framework-free on purpose (threading + OpenCV only): the FastAPI app in
``web/app.py`` -- or a future desktop GUI -- just calls these thread-safe
methods.  The loop steps the engine at the configured FPS (simulated time ==
wall time), renders the camera overlay and minimap to JPEG, and keeps them
ready for streaming.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, Optional

import cv2

from sim.engine import TRAJECTORY_PRESETS, SimulationEngine, build_config
from sim.overlay import Minimap, draw_camera_overlay

#: Knobs exposed in the dashboard's config panel (and their defaults).
DEFAULT_SETTINGS: Dict[str, Any] = {
    "trajectory": "sinusoidal",
    "speed": 220.0,
    "turbulence": 40.0,
    "vibration": 30.0,
    "noise": 20.0,
    "blur": 15.0,
    "acquire_n": 4,
    "acquire_m": 6,
    "lose_n": 6,
    "lose_m": 10,
    "seed": 42,
    "show_truth": False,
}

LIMITS = {
    "speed": (20.0, 800.0),
    "turbulence": (0.0, 100.0), "vibration": (0.0, 100.0), "noise": (0.0, 100.0), "blur": (0.0, 100.0),
    "acquire_n": (1, 30), "acquire_m": (1, 30), "lose_n": (1, 60), "lose_m": (1, 60),
    "seed": (0, 2**31 - 1),
}
DIST_KEYS = ("turbulence", "vibration", "noise", "blur")


def validate_settings(new: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge ``new`` into ``base`` (default settings), coercing and range-checking."""
    s = copy.deepcopy(base or DEFAULT_SETTINGS)
    for k, v in (new or {}).items():
        if k not in DEFAULT_SETTINGS:
            raise ValueError(f"unknown setting {k!r}")
        if k == "trajectory":
            if v not in TRAJECTORY_PRESETS:
                raise ValueError(f"trajectory must be one of {TRAJECTORY_PRESETS}")
            s[k] = v
        elif k == "show_truth":
            s[k] = bool(v)
        else:
            lo, hi = LIMITS[k]
            num = type(DEFAULT_SETTINGS[k])(v)
            if not lo <= num <= hi:
                raise ValueError(f"{k} must be in [{lo}, {hi}]")
            s[k] = num
    if s["acquire_n"] > s["acquire_m"] or s["lose_n"] > s["lose_m"]:
        raise ValueError("N must not exceed M for the acquire/lose thresholds")
    return s


def settings_to_config(s: Dict[str, Any]) -> Dict[str, Any]:
    return build_config(
        trajectory=s["trajectory"], speed=s["speed"],
        disturbances={k: s[k] for k in DIST_KEYS},
        acquire_n=s["acquire_n"], acquire_m=s["acquire_m"], lose_n=s["lose_n"], lose_m=s["lose_m"],
        seed=s["seed"],
    )


class SimRunner:
    """Thread-safe real-time wrapper around :class:`SimulationEngine`."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None, jpeg_quality: int = 80,
                 minimap_width: int = 480) -> None:
        self._lock = threading.RLock()
        self.settings = validate_settings(settings or {})
        self.jpeg_quality = jpeg_quality
        self.minimap_width = minimap_width
        self.engine = SimulationEngine(settings_to_config(self.settings))
        self.minimap = Minimap(self.engine, minimap_width)
        self.running = False
        self.frame_seq = 0
        self.fps = 0.0
        self._last_tick: Optional[float] = None
        self._camera_jpeg = b""
        self._minimap_jpeg = b""
        self._thread: Optional[threading.Thread] = None
        self._alive = False
        self._render()

    # ---------------------------------------------------------- thread loop
    def start_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._alive = True
        self._thread = threading.Thread(target=self._loop, name="beaconlock-sim", daemon=True)
        self._thread.start()

    def stop_thread(self, timeout: float = 2.0) -> None:
        self._alive = False
        if self._thread:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while self._alive:
            t0 = time.perf_counter()
            if self.running:
                self.step_once()
                period = self.engine.dt
            else:
                period = 0.05
            sleep = period - (time.perf_counter() - t0)
            time.sleep(max(sleep, 0.001))

    def step_once(self) -> None:
        """Advance one frame and refresh the JPEGs (also used by tests)."""
        with self._lock:
            self.engine.step()
            now = time.perf_counter()
            if self._last_tick is not None:
                inst = 1.0 / max(now - self._last_tick, 1e-6)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            self._last_tick = now
            self._render(minimap=self.engine.frame_index % 3 == 0)

    def _render(self, minimap: bool = True) -> None:
        q = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        cam = draw_camera_overlay(self.engine, show_truth=self.settings["show_truth"])
        self._camera_jpeg = cv2.imencode(".jpg", cam, q)[1].tobytes()
        if minimap or not self._minimap_jpeg:
            self._minimap_jpeg = cv2.imencode(".jpg", self.minimap.render(), q)[1].tobytes()
        self.frame_seq += 1

    # -------------------------------------------------------------- control
    def start(self) -> None:
        with self._lock:
            self.engine.start()
            self.running = True
            self._last_tick = None

    def stop(self) -> None:
        with self._lock:
            self.running = False
            self.fps = 0.0

    def reset(self, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if settings is not None:
                self.settings = validate_settings(settings, self.settings)
            self.running = False
            self.fps = 0.0
            self.engine.reset(settings_to_config(self.settings))
            self.minimap = Minimap(self.engine, self.minimap_width)
            self._render()
            return copy.deepcopy(self.settings)

    def set_live(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Apply disturbance intensities / show_truth without resetting."""
        with self._lock:
            unknown = set(values) - set(DIST_KEYS) - {"show_truth"}
            if unknown:
                raise ValueError(f"not live-adjustable: {sorted(unknown)}")
            self.settings = validate_settings(values, self.settings)
            self.engine.set_disturbances(**{k: self.settings[k] for k in DIST_KEYS})
            if not self.running:
                self._render()
            return copy.deepcopy(self.settings)

    # ---------------------------------------------------------------- views
    @property
    def camera_jpeg(self) -> bytes:
        return self._camera_jpeg

    @property
    def minimap_jpeg(self) -> bytes:
        return self._minimap_jpeg

    def state(self, history: int = 300) -> Dict[str, Any]:
        with self._lock:
            e = self.engine
            s = e.metrics.summary()
            r = e.last
            recs = e.metrics.recent(history)
            return {
                "synthetic": True,
                "running": self.running,
                "state": e.state.value,
                "frame": e.frame_index,
                "t": round(e.tracker.t, 3),
                "fps": round(self.fps, 1),
                "proc_ms": round(r.proc_ms, 2) if r else None,
                "error_px": round(r.error_px, 2) if (r and e.state.value != "IDLE") else None,
                "detected": bool(r and r.detection is not None),
                "boresight": [round(v, 1) for v in e.camera.center],
                "acquisition_time_s": s.get("acquisition_time_s"),
                "lock_pct": s.get("lock_pct"),
                "lock_retention_pct": s.get("lock_retention_pct"),
                "err_locked_avg_px": s.get("err_locked_avg_px"),
                "err_locked_max_px": s.get("err_locked_max_px"),
                "lock_losses": s.get("lock_losses", 0),
                "reacquisitions": s.get("reacquisitions", 0),
                "fps_processing_capacity": s.get("fps_processing_capacity"),
                "lock_radius_px": e.tracker.config.lock_radius_px,
                "disturbances": e.disturbances.intensities,
                "history": {
                    "t": [round(x.t, 3) for x in recs],
                    "err": [round(x.err_px, 2) for x in recs],
                    "state": [x.state for x in recs],
                },
            }

    def report_zip(self) -> bytes:
        with self._lock:
            z = self.engine.report_zip_bytes()
        return z
