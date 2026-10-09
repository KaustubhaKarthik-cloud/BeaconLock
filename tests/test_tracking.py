"""Tests for tracking.detector, tracking.tracker and tracking.controller."""

import numpy as np
import pytest

from sim import Scene
from sim.camera import PanTiltCamera
from sim.disturbances import DisturbanceConfig, Disturbances
from tracking import (BeaconDetector, BeaconTracker, KalmanFilter2D, PanTiltController, PIDConfig,
                      SpiralScan, TrackerConfig, TrackState)


def cam_on(start, **dist):
    s = Scene.from_dict({"seed": 5, "beacons": [{"id": "b", "trajectory": {"type": "static", "start": start}}]})
    cam = PanTiltCamera(s, center=(960, 540))
    cam.tick(1 / 30)
    f = cam.get_frame()
    if dist:
        f = Disturbances(DisturbanceConfig(**dist)).apply(f, 0.4)
    return s, cam, f


# ---------------------------------------------------------------- detector
def test_detector_finds_beacon_subpixel():
    s, cam, f = cam_on([1000.3, 500.7])
    det = BeaconDetector().detect(f)
    assert det is not None
    u, v = cam.world_to_cam(1000.3, 500.7)
    assert abs(det.x - u) < 0.5 and abs(det.y - v) < 0.5
    x, y, w, h = det.bbox
    assert x <= det.x <= x + w and y <= det.y <= y + h


def test_detector_none_when_only_stars():
    s, cam, f = cam_on([100, 100])          # beacon outside the FOV, stars + noise only
    assert BeaconDetector().detect(f) is None
    assert BeaconDetector().detect(np.zeros((480, 640), np.uint8)) is None


@pytest.mark.parametrize("lvl", [30, 60, 100])
def test_detector_under_disturbances(lvl):
    s, cam, f = cam_on([1000, 500], turbulence=lvl, vibration=lvl, noise=lvl, blur=lvl)
    det = BeaconDetector().detect(f)
    assert det is not None
    u, v = cam.world_to_cam(1000, 500)
    assert np.hypot(det.x - u, det.y - v) < 30


def test_detector_hint_prefers_nearby_blob():
    f = np.full((200, 300), 5, np.uint8)
    yy, xx = np.mgrid[:200, :300]
    for cx, cy, a in ((50, 50, 250), (240, 150, 230)):
        f = np.maximum(f, (a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 32)).astype(np.uint8))
    d = BeaconDetector()
    assert abs(d.detect(f).x - 50) < 1
    assert abs(d.detect(f, hint=(235, 148)).x - 240) < 1
    assert len(d.candidates) == 2


# ------------------------------------------------------------------ kalman
def test_kalman_converges_to_constant_velocity():
    kf = KalmanFilter2D(process_accel=50, meas_sigma=2)
    rng = np.random.default_rng(0)
    kf.init(0, 0)
    for k in range(1, 91):
        kf.predict(1 / 30)
        t = k / 30
        kf.update(200 * t + rng.normal(0, 2), -80 * t + rng.normal(0, 2))
    vx, vy = kf.velocity
    assert vx == pytest.approx(200, abs=15) and vy == pytest.approx(-80, abs=15)
    assert kf.position[0] == pytest.approx(600, abs=4)


def test_spiral_expands_at_constant_speed():
    sp = SpiralScan((500, 500), spacing=300, speed=1000, max_radius=5000)
    pts = [sp.advance(1 / 30)[0] for _ in range(300)]
    r = [np.hypot(*(p - 500)) for p in pts]
    assert r[-1] > r[30] > 0
    steps = np.hypot(*np.diff(np.array(pts), axis=0).T)
    assert np.allclose(steps[10:], 1000 / 30, rtol=0.05)


# ---------------------------------------------------------- state machine
def make_tracker(**kw):
    return BeaconTracker((1920, 1080), (640, 480), TrackerConfig(**kw))


def test_state_machine_full_cycle():
    tr = make_tracker(acquire_n=3, acquire_m=5, lose_n=3, lose_m=5, lost_timeout_s=0.5)
    dt = 1 / 30
    assert tr.update((100, 100), dt).state == TrackState.IDLE
    tr.start()
    out = tr.update(None, dt)
    assert out.state == TrackState.SEARCHING and out.aim is not None
    assert tr.update((500, 500), dt).state == TrackState.ACQUIRING
    tr.update((502, 500), dt)
    out = tr.update((504, 500), dt)
    assert out.state == TrackState.LOCKED and out.state_changed
    assert tr.first_lock_time == pytest.approx(4 * dt)
    for k in range(10):
        assert tr.update((506 + 2 * k, 500), dt).state == TrackState.LOCKED
    tr.update(None, dt)
    tr.update(None, dt)
    assert tr.update(None, dt).state == TrackState.LOST
    out = tr.update((530, 500), dt)           # near prediction -> re-acquire
    assert out.state == TrackState.ACQUIRING and tr.reacquire_count == 1
    tr.update((532, 500), dt)
    assert tr.update((534, 500), dt).state == TrackState.LOCKED
    for _ in range(3):
        tr.update(None, dt)
    assert tr.state == TrackState.LOST
    for _ in range(16):                       # 0.5 s coast timeout
        tr.update(None, dt)
    assert tr.state == TrackState.SEARCHING
    assert tr.lock_count == 2 and tr.loss_count == 2


def test_n_of_m_tolerates_sparse_misses_but_fails_acquisition_on_many():
    tr = make_tracker(acquire_n=4, acquire_m=6)
    tr.start()
    tr.update((500, 500), 1 / 30)             # -> ACQUIRING (1 hit)
    for z in [(501, 500), None, (503, 500)]:
        tr.update(z, 1 / 30)
    assert tr.update((505, 500), 1 / 30).state == TrackState.LOCKED  # 4 of 5

    tr2 = make_tracker(acquire_n=4, acquire_m=6)
    tr2.start()
    tr2.update((500, 500), 1 / 30)
    for _ in range(3):
        tr2.update(None, 1 / 30)
    assert tr2.state == TrackState.SEARCHING   # 3 misses > 6 - 4


def test_gate_rejects_far_detection_while_locked():
    tr = make_tracker(acquire_n=2, acquire_m=3, lose_n=2, lose_m=4, gate_px=50)
    tr.start()
    tr.update((500, 500), 1 / 30)
    tr.update((500, 500), 1 / 30)
    assert tr.state == TrackState.LOCKED
    out = tr.update((900, 900), 1 / 30)       # a distractor far away
    assert not out.detection_used
    tr.update((900, 900), 1 / 30)
    assert tr.state == TrackState.LOST


# -------------------------------------------------------------- controller
def test_pid_drives_error_to_zero():
    ctl = PanTiltController(PIDConfig(kp=8, ki=0.5, kd=0.05, kff=0))
    pos = np.array([0.0, 0.0])
    target = np.array([120.0, -60.0])
    for _ in range(90):
        dx, dy = ctl.update(target - pos, 1 / 30)
        pos += (dx, dy)
    assert np.hypot(*(target - pos)) < 1.0


def test_feedforward_tracks_moving_target_with_small_lag():
    ctl = PanTiltController(PIDConfig(kp=8, ki=1, kd=0.0, kff=1.0))
    pos, tgt, v = np.zeros(2), np.zeros(2), np.array([300.0, 100.0])
    errs = []
    for _ in range(120):
        tgt = tgt + v / 30
        pos += ctl.update(tgt - pos, 1 / 30, target_velocity=v)
        errs.append(np.hypot(*(tgt - pos)))
    assert max(errs[30:]) < 15


def test_controller_rate_clamp_and_no_overshoot():
    ctl = PanTiltController(PIDConfig(kp=1000, ki=0, kd=0, kff=0, max_rate=600))
    dx, dy = ctl.update((5000, 2), 1 / 30)
    assert dx == pytest.approx(20) and dy == pytest.approx(2)   # 600 px/s * dt ; no overshoot of 2 px
