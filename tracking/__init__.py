"""
BeaconLock vision + control (``tracking``).

Pure Python + OpenCV + NumPy, no web-framework imports.

detector.py    threshold + contour/blob detection -> beacon centroid (or None)
tracker.py     constant-velocity Kalman filter + lock state machine
               IDLE -> SEARCHING (spiral scan) -> ACQUIRING -> LOCKED -> LOST -> re-acquire
controller.py  PID (+ velocity feed-forward): pixel error -> pan/tilt commands
"""

from .controller import PID, PanTiltController, PIDConfig
from .detector import BeaconDetector, Detection, DetectorConfig
from .tracker import BeaconTracker, KalmanFilter2D, SpiralScan, TrackerConfig, TrackerOutput, TrackState

__all__ = [
    "BeaconDetector", "Detection", "DetectorConfig",
    "BeaconTracker", "KalmanFilter2D", "SpiralScan", "TrackerConfig", "TrackerOutput", "TrackState",
    "PID", "PIDConfig", "PanTiltController",
]
