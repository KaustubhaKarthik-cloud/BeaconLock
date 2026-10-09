"""Unit tests for sim.trajectories (motion models)."""

import numpy as np
import pytest

from sim.trajectories import (
    LinearTrajectory,
    RandomWaypointTrajectory,
    SinusoidalTrajectory,
    StaticTrajectory,
    TRAJECTORY_TYPES,
    trajectory_from_dict,
)

W, H = 1920, 1080
DT = 1 / 30


def run(traj, n=300, dt=DT):
    traj.bind(W, H)
    return np.array([traj.step(dt) for _ in range(n)])


def in_bounds(path, margin):
    return (
        (path[:, 0] >= margin - 1e-6).all()
        and (path[:, 0] <= W - 1 - margin + 1e-6).all()
        and (path[:, 1] >= margin - 1e-6).all()
        and (path[:, 1] <= H - 1 - margin + 1e-6).all()
    )


def test_static_does_not_move():
    p = run(StaticTrajectory(start=(100, 200)), n=10)
    assert np.allclose(p, [100, 200])


def test_linear_constant_velocity_before_border():
    t = LinearTrajectory(start=(500, 500), velocity=(90, -30)).bind(W, H)
    t.step(1.0)
    assert np.allclose(t.position, (590, 470))
    assert np.allclose(t.velocity, (90, -30))


def test_linear_bounce_stays_inside_and_reflects():
    traj = LinearTrajectory(start=(1800, 100), velocity=(600, -400), mode="bounce", margin=20)
    path = run(traj, n=600)
    assert in_bounds(path, 20)
    # After hitting the right/top border the velocity sign must have flipped at some point.
    vx = np.diff(path[:, 0])
    assert (vx > 0).any() and (vx < 0).any()


def test_linear_bounce_large_dt_is_exact():
    """One big step must equal many small steps (reflection is analytic)."""
    a = LinearTrajectory(start=(300, 300), velocity=(700, 450), mode="bounce").bind(W, H)
    b = LinearTrajectory(start=(300, 300), velocity=(700, 450), mode="bounce").bind(W, H)
    a.step(7.3)
    for _ in range(73):
        b.step(0.1)
    assert np.allclose(a.position, b.position, atol=1e-6)


def test_linear_wrap_reappears_on_other_side():
    traj = LinearTrajectory(start=(1890, 540), velocity=(300, 0), mode="wrap", margin=20).bind(W, H)
    traj.step(0.5)  # +150 px -> wraps past right border
    x, y = traj.position
    assert 20 <= x < 400 and y == pytest.approx(540)


def test_linear_rejects_bad_mode():
    with pytest.raises(ValueError):
        LinearTrajectory(mode="teleport")


def test_sinusoidal_matches_formula():
    traj = SinusoidalTrajectory(start=(200, 540), velocity=(100, 0), amplitude=50, frequency=0.5, wrap=False)
    traj.bind(W, H)
    traj.step(0.5)
    # normal to +x drift is +y; sin(2*pi*0.5*0.5) = 1
    assert np.allclose(traj.position, (250, 590))


def test_sinusoidal_wrap_in_bounds():
    path = run(SinusoidalTrajectory(start=(100, 540), velocity=(800, 0), amplitude=300, frequency=1.0), n=600)
    assert in_bounds(path, 20)


def test_random_waypoint_in_bounds_and_moving():
    traj = RandomWaypointTrajectory(start=(960, 540), speed=300, seed=1, margin=40)
    path = run(traj, n=900)
    assert in_bounds(path, 40)
    steps = np.hypot(*np.diff(path, axis=0).T)
    assert steps.max() <= 300 * DT + 1e-6       # never faster than cruise speed
    assert np.ptp(path[:, 0]) > 200               # actually explores the world


def test_random_waypoint_pause():
    traj = RandomWaypointTrajectory(start=(960, 540), speed=500, pause=1.0, seed=3)
    path = run(traj, n=600)
    steps = np.hypot(*np.diff(path, axis=0).T)
    assert (steps < 1e-9).sum() >= 25            # >= ~1 s of dwell observed


def test_random_waypoint_deterministic_and_reset():
    a = run(RandomWaypointTrajectory(seed=7), n=200)
    b = run(RandomWaypointTrajectory(seed=7), n=200)
    c = run(RandomWaypointTrajectory(seed=8), n=200)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    t = RandomWaypointTrajectory(seed=7).bind(W, H)
    first = [t.step(DT).copy() for _ in range(50)]
    t.reset()
    again = [t.step(DT).copy() for _ in range(50)]
    assert np.array_equal(first, again)


def test_from_dict_roundtrip_all_types():
    for name, cls in TRAJECTORY_TYPES.items():
        t = trajectory_from_dict({"type": name, "start": [300, 300]})
        assert isinstance(t, cls)
        t2 = trajectory_from_dict(t.to_dict())
        assert type(t2) is cls
        a, b = t.bind(W, H), t2.bind(W, H)
        for _ in range(30):
            assert np.allclose(a.step(DT), b.step(DT))


def test_from_dict_errors():
    with pytest.raises(ValueError):
        trajectory_from_dict({"type": "hyperspace"})
    with pytest.raises(TypeError):
        trajectory_from_dict({"type": "linear", "velocty": [1, 2]})  # typo must fail loudly


def test_step_requires_bind_and_nonnegative_dt():
    t = LinearTrajectory()
    with pytest.raises(RuntimeError):
        t.step(0.1)
    t.bind(W, H)
    with pytest.raises(ValueError):
        t.step(-0.1)


def test_per_axis_margin_and_sinusoidal_bounce():
    traj = SinusoidalTrajectory(start=(400, 540), velocity=(900, 0), amplitude=150, frequency=0.5,
                                edge="bounce", margin=(330, 250))
    path = run(traj, n=600)
    assert path[:, 0].min() >= 330 - 1e-6 and path[:, 0].max() <= W - 1 - 330 + 1e-6
    assert path[:, 1].min() >= 250 - 1e-6 and path[:, 1].max() <= H - 1 - 250 + 1e-6
    # bounce => no wrap-around jumps
    assert np.abs(np.diff(path[:, 0])).max() < 900 / 30 + 1


def test_margin_too_large_raises():
    with pytest.raises(ValueError):
        LinearTrajectory(margin=(1000, 10)).bind(W, H)
