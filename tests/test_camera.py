"""Tests for sim.camera.PanTiltCamera."""

import numpy as np
import pytest

from sim import Scene
from sim.camera import PanTiltCamera


def scene():
    return Scene.from_dict({"seed": 3, "world": {"width": 1200, "height": 800},
                            "beacons": [{"id": "b", "trajectory": {"type": "static", "start": [700, 300]}}]})


def test_defaults_and_frame_shape():
    s = scene()
    cam = PanTiltCamera(s)
    assert cam.center == (600, 400)
    f = cam.get_frame()
    assert f.shape == (480, 640) and f.dtype == np.uint8


def test_frame_matches_world_crop_without_noise():
    s = scene()
    cam = PanTiltCamera(s, center=(700, 300))
    x0, y0, w, h = cam.viewport
    assert np.array_equal(cam.get_frame(noise=False), s.render(noise=False)[y0:y0 + h, x0:x0 + w])


def test_pan_tilt_and_coordinate_transforms():
    s = scene()
    cam = PanTiltCamera(s)
    cam.pan(50)
    cam.tilt(-30)
    assert cam.center == pytest.approx((650, 370))
    u, v = cam.world_to_cam(700, 300)
    assert cam.cam_to_world(u, v) == pytest.approx((700, 300))
    assert cam.in_view(700, 300) and not cam.in_view(5, 5)


def test_beacon_at_crosshair_when_pointed():
    s = scene()
    cam = PanTiltCamera(s, center=(700, 300))
    f = cam.get_frame(noise=False)
    u, v = cam.world_to_cam(700, 300)
    assert f[int(v), int(u)] >= 230
    assert cam.frame_center == pytest.approx((320, 240))


def test_clamped_to_world():
    s = scene()
    cam = PanTiltCamera(s)
    cam.pan(-10_000)
    cam.tilt(10_000)
    x0, y0, w, h = cam.viewport
    assert x0 == 0 and y0 + h == 800
    cam.get_frame()  # must not raise


def test_slew_rate_limit():
    s = scene()
    cam = PanTiltCamera(s, max_slew=300)
    cam.tick(0.1)                       # budget 30 px per axis
    assert cam.pan(100) == pytest.approx(30)
    assert cam.pan(10) == pytest.approx(0)   # budget exhausted this step
    cam.tick(0.1)
    assert cam.velocity[0] == pytest.approx(300)
    assert cam.move(-5, 50) == pytest.approx((-5, 30))


def test_unlimited_slew_and_reset():
    s = scene()
    cam = PanTiltCamera(s)
    cam.tick(1 / 30)
    cam.point_at(1000, 600)
    assert cam.center == (880, 560)     # clamped to (1200 - 320, 800 - 240)
    cam.reset()
    assert cam.center == (600, 400)


def test_viewport_too_large():
    with pytest.raises(ValueError):
        PanTiltCamera(scene(), width=2000, height=480)
