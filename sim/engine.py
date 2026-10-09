"""
sim.engine -- headless closed-loop simulation runner.

:class:`SimulationEngine` wires every core component into the coarse-alignment
loop executed once per frame::

    Scene.step(dt)                         beacon moves (ground truth)
      -> PanTiltCamera.get_frame()         crop the field of view
      -> Disturbances.apply()              turbulence / vibration / blur / noise
      -> BeaconDetector.detect()           centroid in camera px (or None)
      -> BeaconTracker.update()            Kalman + IDLE/SEARCHING/ACQUIRING/LOCKED/LOST
      -> PanTiltController.update()        pixel error -> pan/tilt step
      -> PanTiltCamera.move()              slew-limited gimbal motion
      -> MetricsRecorder.log()             error vs ground truth, timing

It has **no UI dependencies**: the web dashboard (``web/``), the headless CLI
(``scripts/run_headless.py``), the pytest suite and a future PyInstaller
desktop build all drive this same class.

Quick start::

    from sim.engine import SimulationEngine
    eng = SimulationEngine.from_json("configs/demo.json")
    eng.start()
    summary = eng.run(seconds=20)
    print(eng.metrics.summary_text())
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple, Union

import numpy as np

from metrics import FrameRecord, MetricsRecorder
from tracking.controller import PanTiltController, PIDConfig
from tracking.detector import BeaconDetector, Detection, DetectorConfig
from tracking.tracker import BeaconTracker, TrackerConfig, TrackerOutput, TrackState

from .camera import PanTiltCamera
from .disturbances import DisturbanceConfig, Disturbances
from .scene import DEFAULT_CONFIG as DEFAULT_SCENE_CONFIG
from .scene import Scene, _deep_merge

__all__ = ["DEFAULT_ENGINE_CONFIG", "FrameResult", "SimulationEngine", "build_config", "TRAJECTORY_PRESETS"]

# Margin that keeps the beacon centrable by a 640x480 camera clamped to the world.
_CENTRABLE_MARGIN = [330, 250]

DEFAULT_ENGINE_CONFIG: Dict[str, Any] = {
    "name": "demo",
    "description": "SYNTHETIC demo: camera starts off-target, spiral-searches, locks and tracks a weaving beacon under turbulence.",
    "synthetic": True,
    "sim": {"fps": 30.0, "target_beacon": 0},
    "scene": _deep_merge(DEFAULT_SCENE_CONFIG, {
        "name": "demo",
        "beacons": [{
            "id": "beacon-1", "sigma": 4.0, "intensity": 255.0,
            "trajectory": {"type": "sinusoidal", "start": [520, 420], "velocity": [220, 0], "amplitude": 160,
                           "frequency": 0.18, "phase": 0.0, "edge": "bounce", "margin": _CENTRABLE_MARGIN},
        }],
    }),
    "camera": {"width": 640, "height": 480, "center": [1550, 830], "max_slew": 1500.0},
    "disturbances": {"turbulence": 40, "vibration": 30, "noise": 20, "blur": 15, "seed": 7},
    "detector": {},
    "tracker": {},
    "controller": {},
}

#: Trajectory presets used by :func:`build_config` (and the UI drop-down).
TRAJECTORY_PRESETS = ("sinusoidal", "linear", "random_waypoint")


def build_config(
    trajectory: str = "sinusoidal",
    speed: float = 220.0,
    disturbances: Optional[Dict[str, float]] = None,
    acquire_n: Optional[int] = None,
    acquire_m: Optional[int] = None,
    lose_n: Optional[int] = None,
    lose_m: Optional[int] = None,
    seed: Optional[int] = None,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Make an engine config from the handful of knobs exposed in the UI.

    ``speed`` is the beacon speed in px/s (drift speed for sinusoidal, cruise
    speed for random waypoint, |v| for linear).
    """
    cfg = copy.deepcopy(base or DEFAULT_ENGINE_CONFIG)
    speed = float(speed)
    if trajectory == "sinusoidal":
        traj = {"type": "sinusoidal", "start": [520, 420], "velocity": [speed, 0], "amplitude": 160,
                "frequency": 0.18, "edge": "bounce"}
    elif trajectory == "linear":
        traj = {"type": "linear", "start": [520, 420], "velocity": [speed * 0.8, speed * 0.6], "mode": "bounce"}
    elif trajectory == "random_waypoint":
        traj = {"type": "random_waypoint", "start": [520, 420], "speed": speed, "pause": 0.5, "speed_jitter": 0.2}
    else:
        raise ValueError(f"unknown trajectory preset {trajectory!r}; choose from {TRAJECTORY_PRESETS}")
    traj["margin"] = list(_CENTRABLE_MARGIN)
    cfg["scene"]["beacons"] = [{"id": "beacon-1", "sigma": 4.0, "intensity": 255.0, "trajectory": traj}]
    if disturbances:
        cfg["disturbances"].update({k: float(v) for k, v in disturbances.items()})
    tr = cfg.setdefault("tracker", {})
    for k, v in (("acquire_n", acquire_n), ("acquire_m", acquire_m), ("lose_n", lose_n), ("lose_m", lose_m)):
        if v is not None:
            tr[k] = int(v)
    if tr.get("acquire_n", 1) > tr.get("acquire_m", 10**9) or tr.get("lose_n", 1) > tr.get("lose_m", 10**9):
        raise ValueError("N must not exceed M for the N/M thresholds")
    if seed is not None:
        cfg["scene"]["seed"] = int(seed)
        cfg["disturbances"]["seed"] = int(seed) + 1
    return cfg


@dataclass
class FrameResult:
    """Everything produced by one :meth:`SimulationEngine.step`."""

    index: int
    t: float                          # simulated seconds since start()
    frame: np.ndarray                 # disturbed camera image (uint8 gray)
    detection: Optional[Detection]
    track: TrackerOutput
    viewport: Tuple[int, int, int, int]
    boresight: Tuple[float, float]    # world px at image time
    truth: Tuple[float, float]        # ground-truth target beacon, world px
    error_px: float
    command: Tuple[float, float]      # pan/tilt applied this step, px
    proc_ms: float

    @property
    def state(self) -> TrackState:
        return self.track.state


class SimulationEngine:
    """Headless BeaconLock simulation (scene + camera + disturbances + PAT + metrics)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = _deep_merge(DEFAULT_ENGINE_CONFIG, config or {})
        if config and "scene" in config and "beacons" in config["scene"]:
            self.config["scene"]["beacons"] = copy.deepcopy(config["scene"]["beacons"])
        self._build()

    # ------------------------------------------------------------- creation
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SimulationEngine":
        return cls(config)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SimulationEngine":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def _build(self) -> None:
        c = self.config
        self.fps = float(c["sim"].get("fps", 30.0))
        self.dt = 1.0 / self.fps
        self.target_index = int(c["sim"].get("target_beacon", 0))
        self.scene = Scene.from_dict(c["scene"])
        cam = c["camera"]
        self.camera = PanTiltCamera(self.scene, int(cam.get("width", 640)), int(cam.get("height", 480)),
                                    center=tuple(cam["center"]) if cam.get("center") else None,
                                    max_slew=cam.get("max_slew"))
        self.disturbances = Disturbances(DisturbanceConfig.from_dict(c["disturbances"]))
        self.detector = BeaconDetector(DetectorConfig.from_dict(c["detector"]))
        self.tracker = BeaconTracker(self.scene.size, (self.camera.width, self.camera.height),
                                     TrackerConfig.from_dict(c["tracker"]))
        self.controller = PanTiltController(PIDConfig.from_dict(c["controller"]))
        self.metrics = MetricsRecorder()
        self.frame_index = 0
        self.started = False
        self.last: Optional[FrameResult] = None
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=240)

    def reset(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Rebuild everything (optionally with a new config); back to IDLE."""
        if config is not None:
            self.config = _deep_merge(DEFAULT_ENGINE_CONFIG, config)
            if "scene" in config and "beacons" in config["scene"]:
                self.config["scene"]["beacons"] = copy.deepcopy(config["scene"]["beacons"])
        self._build()

    # -------------------------------------------------------------- control
    def start(self) -> None:
        """Begin autonomous acquisition (tracker IDLE -> SEARCHING)."""
        if not self.started:
            d = self.disturbances.intensities
            tc = self.tracker.config
            self.metrics.begin({
                "scenario": self.config.get("name", "unnamed"),
                "trajectory": self.config["scene"]["beacons"][self.target_index]["trajectory"].get("type"),
                "disturbances": ", ".join(f"{k} {v:.0f}" for k, v in d.items()),
                "thresholds": f"acquire {tc.acquire_n}/{tc.acquire_m}, lose {tc.lose_n}/{tc.lose_m}",
                "seed": self.config["scene"].get("seed"),
            })
            self.started = True
        # Spiral search starts around the camera's current pointing (the a-priori
        # direction) unless the tracker config fixes a search centre.
        self.tracker.start(from_point=self.camera.center)

    def set_disturbances(self, **intensities: float) -> None:
        """Change disturbance intensities live (0-100), e.g. from UI sliders."""
        self.disturbances.set(**intensities)
        self.config["disturbances"].update(self.disturbances.intensities)

    @property
    def state(self) -> TrackState:
        return self.tracker.state

    @property
    def truth(self) -> Tuple[float, float]:
        return self.scene.beacons[self.target_index].position

    def preview(self) -> np.ndarray:
        """Render the current (disturbed) view without advancing time."""
        raw = self.camera.get_frame()
        return self.disturbances.apply(raw, self.scene.t, tuple(self.camera.velocity))

    # ----------------------------------------------------------------- step
    def step(self) -> FrameResult:
        """Run one closed-loop frame and log it to :attr:`metrics`."""
        t0 = time.perf_counter()
        dt = self.dt
        self.scene.step(dt)
        self.camera.tick(dt)

        raw = self.camera.get_frame()
        frame = self.disturbances.apply(raw, self.scene.t, tuple(self.camera.velocity))

        hint = None
        kf = self.tracker.kf
        if kf.initialized and self.tracker.state != TrackState.SEARCHING:
            px, py = kf.position
            vx, vy = kf.velocity
            hint = self.camera.world_to_cam(px + vx * dt, py + vy * dt)
        det = self.detector.detect(frame, hint=hint)
        meas = self.camera.cam_to_world(det.x, det.y) if det is not None else None

        out = self.tracker.update(meas, dt, boresight=self.camera.center)
        if out.state_changed:
            self.controller.reset()

        bx, by = self.camera.center
        viewport = self.camera.viewport
        gx, gy = self.truth
        err = math.hypot(gx - bx, gy - by)

        cmd = (0.0, 0.0)
        if out.aim is not None:
            ex, ey = out.aim[0] - bx, out.aim[1] - by
            dx, dy = self.controller.update((ex, ey), dt, out.aim_velocity)
            cmd = self.camera.move(dx, dy)
        proc_ms = (time.perf_counter() - t0) * 1000.0

        est = out.estimate
        if self.tracker.state != TrackState.IDLE:
            self.metrics.log(FrameRecord(
                frame=self.frame_index, t=self.tracker.t, state=out.state.value, detected=det is not None,
                det_u=det.x if det else None, det_v=det.y if det else None,
                est_x=est[0] if est else None, est_y=est[1] if est else None,
                cam_x=bx, cam_y=by, gt_x=gx, gt_y=gy, err_px=err,
                est_err_px=math.hypot(gx - est[0], gy - est[1]) if est else None,
                gt_in_fov=bool(viewport[0] <= gx < viewport[0] + viewport[2]
                               and viewport[1] <= gy < viewport[1] + viewport[3]),
                proc_ms=proc_ms,
            ))
        self.trail.append((bx, by))
        res = FrameResult(self.frame_index, self.tracker.t, frame, det, out, viewport, (bx, by), (gx, gy), err,
                          cmd, proc_ms)
        self.frame_index += 1
        self.last = res
        return res

    def run(self, seconds: float, start: bool = True, callback=None) -> Dict[str, Any]:
        """Run ``seconds`` of simulated time as fast as possible; returns the summary.

        ``callback(result)`` is invoked after every frame if given (e.g. to
        write video).
        """
        if start:
            self.start()
        for _ in range(int(round(seconds * self.fps))):
            r = self.step()
            if callback is not None:
                callback(r)
        return self.metrics.summary()

    # --------------------------------------------------------------- report
    def summary(self) -> Dict[str, Any]:
        return self.metrics.summary()

    def report_zip_bytes(self) -> bytes:
        return self.metrics.report_zip_bytes(self.config)

    def write_report(self, directory: Union[str, Path], prefix: str = "beaconlock") -> Dict[str, Path]:
        return self.metrics.write_report(directory, self.config, prefix)
