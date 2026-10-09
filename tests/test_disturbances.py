"""Tests for sim.disturbances."""

import numpy as np
import pytest

from sim.disturbances import DisturbanceConfig, Disturbances, Perlin2D


def frame():
    f = np.full((240, 320), 8, np.uint8)
    yy, xx = np.mgrid[:240, :320]
    f[:] = np.clip(8 + 247 * np.exp(-((xx - 160) ** 2 + (yy - 120) ** 2) / (2 * 4.0 ** 2)), 0, 255).astype(np.uint8)
    return f


def test_perlin_range_smooth_deterministic():
    p = Perlin2D(1)
    x, y = np.meshgrid(np.linspace(0, 8, 200), np.linspace(0, 8, 200))
    n = p.noise(x, y)
    assert n.min() >= -1.05 and n.max() <= 1.05 and n.std() > 0.1
    assert np.abs(np.diff(n, axis=1)).max() < 0.2          # spatially smooth
    assert np.array_equal(n, Perlin2D(1).noise(x, y))
    assert np.allclose(p.noise(np.arange(5.0), np.arange(5.0)), 0)  # zero at lattice points


def test_zero_intensity_is_identity():
    f = frame()
    assert np.array_equal(Disturbances().apply(f, 1.0), f)


@pytest.mark.parametrize("kind", ["turbulence", "vibration", "noise", "blur"])
def test_each_disturbance_changes_frame_keeps_shape(kind):
    f = frame()
    d = Disturbances(DisturbanceConfig(**{kind: 60}))
    out = d.apply(f, 0.37, camera_velocity=(200, 0))
    assert out.shape == f.shape and out.dtype == np.uint8
    assert not np.array_equal(out, f)


def test_beacon_survives_max_disturbance():
    f = frame()
    d = Disturbances(DisturbanceConfig(turbulence=100, vibration=100, noise=100, blur=100))
    out = d.apply(f, 0.5, (300, 0)).astype(float)
    y, x = np.unravel_index(np.argmax(out), out.shape)
    assert np.hypot(x - 160, y - 120) < 40      # displaced, but still the brightest thing around


def test_vibration_is_bounded_and_periodic_motion():
    d = Disturbances(DisturbanceConfig(vibration=100))
    offs = np.array([d.vibration_offset(t) for t in np.linspace(0, 2, 400)])
    assert np.abs(offs).max() <= d.config.max_vibration_px + 1e-9
    assert offs.std(axis=0).min() > 1.0


def test_deterministic_and_reset():
    f = frame()
    cfg = dict(turbulence=50, vibration=50, noise=50, blur=50, seed=11)
    a, b = Disturbances(DisturbanceConfig(**cfg)), Disturbances(DisturbanceConfig(**cfg))
    ra = [a.apply(f, i / 30) for i in range(5)]
    rb = [b.apply(f, i / 30) for i in range(5)]
    assert all(np.array_equal(x, y) for x, y in zip(ra, rb))
    a.reset()
    assert np.array_equal(a.apply(f, 0.0), ra[0])


def test_set_clamps_and_rejects_unknown():
    d = Disturbances()
    d.set(turbulence=150, noise=-5)
    assert d.intensities["turbulence"] == 100 and d.intensities["noise"] == 0
    with pytest.raises(KeyError):
        d.set(gravity=10)
