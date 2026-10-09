"""End-to-end tests of the headless closed loop (sim.engine.SimulationEngine)."""

import io
import zipfile

import numpy as np
import pytest

from sim.engine import DEFAULT_ENGINE_CONFIG, SimulationEngine, build_config
from sim.overlay import Minimap, compose_dashboard_frame, draw_camera_overlay
from tracking import TrackState

DEMO_DIST = dict(turbulence=40, vibration=30, noise=20, blur=15)


def test_e2e_demo_reaches_locked_with_disturbances_on():
    """The 'wow' test: start -> spiral sweep -> LOCKED -> holds lock under turbulence."""
    eng = SimulationEngine()
    assert all(eng.disturbances.intensities[k] > 0 for k in DEMO_DIST)  # disturbances really on
    assert not eng.camera.in_view(*eng.truth)                           # starts off-target
    assert eng.state == TrackState.IDLE
    s = eng.run(20)
    states = [r.state for r in eng.metrics.records]
    assert states[0] == "SEARCHING" and "ACQUIRING" in states
    assert s["acquired"] and s["acquisition_time_s"] < 8.0
    assert s["lock_retention_pct"] > 95.0
    assert s["err_locked_avg_px"] < 10.0
    assert eng.state == TrackState.LOCKED


def test_e2e_heavy_turbulence_still_locks():
    eng = SimulationEngine(build_config("sinusoidal", 220, disturbances=dict(turbulence=90, vibration=60, noise=50, blur=40)))
    s = eng.run(20)
    assert s["acquired"] and s["lock_retention_pct"] > 85.0


@pytest.mark.parametrize("traj", ["linear", "random_waypoint"])
def test_e2e_other_trajectories(traj):
    s = SimulationEngine(build_config(traj, 300)).run(15)
    assert s["acquired"] and s["lock_retention_pct"] > 90.0 and s["err_locked_avg_px"] < 15


def test_lost_and_reacquire_when_beacon_hidden():
    eng = SimulationEngine(build_config("sinusoidal", 150, disturbances=dict(turbulence=0, vibration=0, noise=0, blur=0)))
    eng.run(6)
    assert eng.state == TrackState.LOCKED
    b = eng.scene.beacons[0]
    b.intensity = 0.0                       # occlusion: beacon switched off
    for _ in range(15):
        eng.step()
    assert eng.state in (TrackState.LOST, TrackState.SEARCHING)
    b.intensity = 255.0
    for _ in range(150):
        eng.step()
    assert eng.state == TrackState.LOCKED
    s = eng.summary()
    assert s["lock_losses"] >= 1


def test_deterministic_runs():
    a, b = SimulationEngine(), SimulationEngine()
    a.run(3)
    b.run(3)
    ea = [r.err_px for r in a.metrics.records]
    eb = [r.err_px for r in b.metrics.records]
    assert np.allclose(ea, eb)


def test_idle_until_start_and_reset():
    eng = SimulationEngine()
    c0 = eng.camera.center
    for _ in range(5):
        eng.step()
    assert eng.state == TrackState.IDLE and eng.camera.center == c0 and len(eng.metrics) == 0
    eng.start()
    eng.step()
    assert eng.state == TrackState.SEARCHING and len(eng.metrics) == 1
    eng.reset()
    assert eng.state == TrackState.IDLE and len(eng.metrics) == 0 and eng.camera.center == c0


def test_live_disturbance_change_and_report():
    eng = SimulationEngine()
    eng.set_disturbances(turbulence=75)
    assert eng.config["disturbances"]["turbulence"] == 75
    eng.run(2)
    z = zipfile.ZipFile(io.BytesIO(eng.report_zip_bytes()))
    assert {"frames.csv", "summary.txt", "summary.json", "config.json"} <= set(z.namelist())


def test_build_config_validation():
    cfg = build_config("linear", 300, acquire_n=3, acquire_m=5, lose_n=4, lose_m=8, seed=9)
    assert cfg["tracker"]["acquire_n"] == 3 and cfg["scene"]["seed"] == 9
    with pytest.raises(ValueError):
        build_config("teleport")
    with pytest.raises(ValueError):
        build_config(acquire_n=9, acquire_m=3)
    assert DEFAULT_ENGINE_CONFIG["synthetic"] is True


def test_overlay_and_minimap_render():
    eng = SimulationEngine()
    eng.run(4)
    img = draw_camera_overlay(eng, show_truth=True)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    mm = Minimap(eng, width=400)
    assert mm.render().shape == (225, 400, 3)
    comp = compose_dashboard_frame(eng, mm)
    assert comp.shape[0] == 480
