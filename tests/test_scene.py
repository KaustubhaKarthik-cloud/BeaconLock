"""Unit tests for sim.scene (synthetic world rendering + ground truth)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sim import Scene, stamp_gaussian

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("*.json"))


def small_cfg(**over):
    cfg = {
        "seed": 1,
        "world": {"width": 320, "height": 240, "background_level": 6},
        "stars": {"count": 40},
        "noise": {"sigma": 3.0},
        "beacons": [{"id": "b", "sigma": 3.0, "intensity": 255,
                     "trajectory": {"type": "linear", "start": [50, 60], "velocity": [120, 45]}}],
    }
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_load_and_render(path):
    cfg = json.loads(path.read_text())
    assert cfg.get("synthetic") is True, "every shipped config must be labelled synthetic"
    if "scene" in cfg:  # engine config (scene nested inside)
        from sim.engine import SimulationEngine
        cfg = SimulationEngine.from_json(path).config["scene"]
        assert cfg.get("synthetic") is True
        s = Scene.from_dict(cfg)
    else:
        s = Scene.from_json(path)
    img = s.render()
    assert img.shape == (cfg["world"]["height"], cfg["world"]["width"])
    assert img.dtype == np.uint8
    assert len(s.ground_truth()) == len(cfg["beacons"])


def test_default_world_size():
    s = Scene.from_dict({})
    assert s.size == (1920, 1080)
    assert s.render().shape == (1080, 1920)


def test_render_shapes_and_dtypes():
    s = Scene.from_dict(small_cfg())
    g, c = s.render(), s.render_bgr()
    assert g.shape == (240, 320) and g.dtype == np.uint8
    assert c.shape == (240, 320, 3) and c.dtype == np.uint8


def test_background_is_dark():
    s = Scene.from_dict(small_cfg())
    img = s.render()
    assert np.median(img) < 20


def test_beacon_bright_at_ground_truth_through_motion():
    s = Scene.from_dict(small_cfg())
    for _ in range(60):
        s.step(1 / 30)
        img = s.render()
        x, y = s.beacon_positions()[0]
        xi, yi = int(round(x)), int(round(y))
        assert img[yi, xi] >= 230                        # beacon core saturates
        win = img[yi - 2:yi + 3, xi - 2:xi + 3].astype(float)
        # brightest pixel in the 5x5 window is within 1 px of ground truth
        py, px = np.unravel_index(np.argmax(win), win.shape)
        assert abs(px - 2) <= 1 and abs(py - 2) <= 1


def test_beacon_is_brightest_blob_in_noise_free_frame():
    s = Scene.from_dict(small_cfg(noise={"sigma": 0}, watermark={"enabled": False}))
    s.step(0.5)
    img = s.render(noise=False).astype(float)
    x, y = s.beacon_positions()[0]
    # Intensity-weighted centroid of saturated-ish pixels ~= ground truth (sub-pixel)
    ys, xs = np.nonzero(img > 200)
    assert abs(xs.mean() - x) < 0.6 and abs(ys.mean() - y) < 0.6


def test_determinism_same_seed():
    a, b = Scene.from_dict(small_cfg()), Scene.from_dict(small_cfg())
    for _ in range(10):
        a.step(1 / 30)
        b.step(1 / 30)
        assert np.array_equal(a.render(), b.render())


def test_different_seed_differs():
    a = Scene.from_dict(small_cfg(seed=1)).render(noise=False)
    b = Scene.from_dict(small_cfg(seed=2)).render(noise=False)
    assert not np.array_equal(a, b)  # different star field


def test_reset_replays_bit_for_bit():
    s = Scene.from_json(REPO / "configs" / "multi_beacon.json")
    first = []
    for _ in range(5):
        s.step(1 / 30)
        first.append((s.render(), s.beacon_positions()))
    s.reset()
    assert s.t == 0 and s.frame_index == 0
    for img, pos in first:
        s.step(1 / 30)
        assert np.array_equal(s.render(), img)
        assert np.array_equal(s.beacon_positions(), pos)


def test_noise_changes_per_frame_but_not_stars():
    s = Scene.from_dict(small_cfg())
    a, b = s.render(), s.render()
    assert not np.array_equal(a, b)
    assert np.array_equal(s.render(noise=False), s.render(noise=False))


def test_ground_truth_fields_and_velocity():
    s = Scene.from_dict(small_cfg())
    gt = s.step(0.1)
    st = gt[0]
    assert st.id == "b"
    assert st.x == pytest.approx(62) and st.y == pytest.approx(64.5)
    assert st.vx == pytest.approx(120) and st.vy == pytest.approx(45)
    assert s.t == pytest.approx(0.1) and s.frame_index == 1


def test_requires_a_beacon():
    with pytest.raises(ValueError):
        Scene.from_dict({"beacons": []})


def test_stamp_gaussian_offscreen_is_safe():
    img = np.zeros((10, 10), np.float32)
    stamp_gaussian(img, -50, -50, 2.0, 255)
    stamp_gaussian(img, 9.5, 9.5, 2.0, 255)
    assert img[9, 9] > 0 and img[0, 0] == 0


def test_core_has_no_web_imports():
    """sim/, tracking/ and metrics.py must stay UI-independent (PyInstaller deliverable)."""
    banned = ("flask", "fastapi", "django", "starlette", "uvicorn", "aiohttp", "tornado", "streamlit")
    for f in list((REPO / "sim").rglob("*.py")) + list((REPO / "tracking").rglob("*.py")) + [REPO / "metrics.py"]:
        src = f.read_text().lower()
        for mod in banned:
            assert f"import {mod}" not in src and f"from {mod}" not in src, f"{f} imports {mod}"
