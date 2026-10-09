"""
BeaconLock simulation core (``sim``).

UI-independent, pure Python + NumPy + OpenCV.  Must never import a web
framework: the same core is wrapped by the web dashboard and, later, by a
desktop GUI frozen with PyInstaller.

Modules
-------
trajectories   beacon motion models (static, linear, sinusoidal, random waypoint)  [step 1]
scene          synthetic night-sky world with moving beacons                       [step 1]
camera         virtual pan-tilt camera viewport (clamped, slew-rate limited)      [step 2]
disturbances   Perlin turbulence, vibration jitter, motion blur, sensor noise      [step 2]
engine         headless closed-loop runner (scene+camera+PAT+metrics)              [step 5]
overlay        UI-agnostic OpenCV drawing: camera overlay, minimap, composite      [step 5]

All scenes are SYNTHETIC; no real ISRO / mission data is used.
"""

from .scene import DEFAULT_CONFIG, Beacon, BeaconState, Scene, stamp_gaussian
from .trajectories import (
    TRAJECTORY_TYPES,
    LinearTrajectory,
    RandomWaypointTrajectory,
    SinusoidalTrajectory,
    StaticTrajectory,
    Trajectory,
    trajectory_from_dict,
)

__all__ = [
    "Scene", "Beacon", "BeaconState", "DEFAULT_CONFIG", "stamp_gaussian",
    "Trajectory", "StaticTrajectory", "LinearTrajectory", "SinusoidalTrajectory",
    "RandomWaypointTrajectory", "TRAJECTORY_TYPES", "trajectory_from_dict",
]

__version__ = "1.0.0"
