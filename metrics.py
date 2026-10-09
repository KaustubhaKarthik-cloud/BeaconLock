"""
metrics -- per-run performance logging for BeaconLock.

Every simulation frame is logged as a :class:`FrameRecord`; at any time the
:class:`MetricsRecorder` can produce

* a **CSV** with one row per frame (``to_csv`` / ``write_report``),
* a **summary** dict + human-readable text report (``summary`` /
  ``summary_text``), and
* a ready-to-download **ZIP** bundle (CSV + summary.txt + summary.json +
  config.json) via ``report_zip_bytes`` -- used by the web dashboard's
  "Download performance report" button and by the desktop build.

Key metrics
-----------
duration            simulated seconds and wall-clock seconds of the run
FPS                 achieved loop rate (frames / wall time) and processing
                    capacity (1000 / mean per-frame processing ms)
acquisition time    simulated seconds from start() to the first LOCKED frame
tracking error      boresight-to-beacon distance in px measured against the
                    simulator's **ground truth** (avg / RMS / p95 / max), over
                    LOCKED frames
lock retention %    LOCKED frames / frames since first lock
lock %              LOCKED frames / all frames since start
per-frame time      processing ms per frame (mean / p95 / max)

Standard library + NumPy only; no web-framework imports.  All numbers come
from SYNTHETIC simulations.
"""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

__all__ = ["FrameRecord", "MetricsRecorder"]


@dataclass
class FrameRecord:
    """One row of the per-frame log (world px unless noted)."""

    frame: int
    t: float                  # simulated seconds since start()
    state: str                # tracker state name
    detected: bool            # detector returned a blob this frame
    det_u: Optional[float]    # detection in camera px
    det_v: Optional[float]
    est_x: Optional[float]    # Kalman estimate
    est_y: Optional[float]
    cam_x: float              # boresight
    cam_y: float
    gt_x: float               # ground-truth beacon position
    gt_y: float
    err_px: float             # |ground truth - boresight|
    est_err_px: Optional[float]  # |ground truth - Kalman estimate|
    gt_in_fov: bool
    proc_ms: float            # processing time of this frame (render..command)


def _r(x: Optional[float], nd: int = 3) -> Optional[float]:
    return None if x is None else round(float(x), nd)


class MetricsRecorder:
    """Collects :class:`FrameRecord` rows and derives the run summary."""

    COLUMNS = [f.name for f in fields(FrameRecord)]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.records: List[FrameRecord] = []
        self.wall_start: Optional[float] = None
        self.wall_end: Optional[float] = None
        self.meta: Dict[str, Any] = {}

    # ---------------------------------------------------------------- logging
    def begin(self, meta: Optional[Dict[str, Any]] = None) -> None:
        """Mark the wall-clock start of a run (``meta`` goes into the report)."""
        self.wall_start = time.perf_counter()
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        if meta:
            self.meta.update(meta)

    def log(self, rec: FrameRecord) -> None:
        if self.wall_start is None:
            self.begin()
        self.records.append(rec)
        self.wall_end = time.perf_counter()

    def __len__(self) -> int:
        return len(self.records)

    def recent(self, n: int = 300) -> List[FrameRecord]:
        return self.records[-n:]

    # ---------------------------------------------------------------- summary
    def summary(self) -> Dict[str, Any]:
        """Compute all run metrics (safe to call while running)."""
        recs = self.records
        n = len(recs)
        out: Dict[str, Any] = {"synthetic": True, "frames": n}
        if n == 0:
            return out
        states = np.array([r.state for r in recs])
        err = np.array([r.err_px for r in recs], dtype=float)
        proc = np.array([r.proc_ms for r in recs], dtype=float)
        t = np.array([r.t for r in recs], dtype=float)
        dt = float(np.median(np.diff(t))) if n > 1 else 0.0
        locked = states == "LOCKED"
        wall = (self.wall_end or 0) - (self.wall_start or 0)

        out["sim_duration_s"] = round(float(t[-1]) + dt, 3) if n else 0.0
        out["wall_duration_s"] = round(wall, 3)
        out["fps_loop"] = round(n / wall, 2) if wall > 0 else None
        out["fps_processing_capacity"] = round(1000.0 / proc.mean(), 1) if proc.mean() > 0 else None
        out["proc_ms_mean"] = round(float(proc.mean()), 3)
        out["proc_ms_p95"] = round(float(np.percentile(proc, 95)), 3)
        out["proc_ms_max"] = round(float(proc.max()), 3)

        first_lock = int(np.argmax(locked)) if locked.any() else None
        out["acquired"] = first_lock is not None
        out["acquisition_time_s"] = round(float(t[first_lock]), 3) if first_lock is not None else None
        out["lock_pct"] = round(100.0 * locked.mean(), 2)
        if first_lock is not None:
            after = locked[first_lock:]
            out["lock_retention_pct"] = round(100.0 * after.mean(), 2)
            e = err[locked]
            out["err_locked_avg_px"] = round(float(e.mean()), 3)
            out["err_locked_rms_px"] = round(float(np.sqrt((e ** 2).mean())), 3)
            out["err_locked_p95_px"] = round(float(np.percentile(e, 95)), 3)
            out["err_locked_max_px"] = round(float(e.max()), 3)
            est = np.array([r.est_err_px for r in recs if r.state == "LOCKED" and r.est_err_px is not None])
            out["est_err_locked_avg_px"] = round(float(est.mean()), 3) if est.size else None
            out["err_after_acq_avg_px"] = round(float(err[first_lock:].mean()), 3)
        else:
            for k in ("lock_retention_pct", "err_locked_avg_px", "err_locked_rms_px", "err_locked_p95_px",
                      "err_locked_max_px", "est_err_locked_avg_px", "err_after_acq_avg_px"):
                out[k] = None
        # Transitions
        trans = list(zip(states[:-1], states[1:]))
        out["lock_losses"] = sum(1 for a, b in trans if a == "LOCKED" and b != "LOCKED")
        out["reacquisitions"] = sum(1 for a, b in trans if a == "LOST" and b == "ACQUIRING")
        out["detection_rate_pct"] = round(100.0 * float(np.mean([r.detected for r in recs])), 2)
        out["time_in_state_s"] = {s: round(float((states == s).sum() * dt), 3)
                                  for s in ("SEARCHING", "ACQUIRING", "LOCKED", "LOST")}
        out["final_state"] = str(states[-1])
        return out

    def summary_text(self, title: str = "BeaconLock performance report") -> str:
        """Human-readable report (also written as ``summary.txt``)."""
        s = self.summary()

        def f(v: Any, unit: str = "", nd: int = 2) -> str:
            if v is None:
                return "n/a"
            return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{v}{unit}"

        lines = [
            title,
            "=" * len(title),
            "SYNTHETIC SIMULATION -- all scenes, beacons and disturbances are procedurally generated;",
            "no real ISRO / mission data is used. Errors are in camera pixels vs simulator ground truth.",
            "",
            f"Generated:            {datetime.now().astimezone().isoformat(timespec='seconds')}",
        ]
        for k in ("scenario", "trajectory", "disturbances", "thresholds", "seed"):
            if k in self.meta:
                lines.append(f"{k.capitalize() + ':':<22}{self.meta[k]}")
        if s.get("frames", 0) == 0:
            return "\n".join(lines + ["", "No frames recorded."])
        lines += [
            "",
            f"Frames:               {s['frames']}",
            f"Duration:             {f(s['sim_duration_s'], ' s')} simulated, {f(s['wall_duration_s'], ' s')} wall-clock",
            f"FPS:                  {f(s['fps_loop'], '', 1)} loop, {f(s['fps_processing_capacity'], '', 0)} processing capacity",
            f"Per-frame time:       mean {f(s['proc_ms_mean'], ' ms')}, p95 {f(s['proc_ms_p95'], ' ms')}, max {f(s['proc_ms_max'], ' ms')}",
            "",
            f"Acquired:             {'YES' if s['acquired'] else 'NO'}",
            f"Acquisition time:     {f(s['acquisition_time_s'], ' s')}  (start -> first LOCKED)",
            f"Lock retention:       {f(s['lock_retention_pct'], ' %')}  (LOCKED frames since first lock)",
            f"Lock (overall):       {f(s['lock_pct'], ' %')}",
            f"Lock losses:          {s['lock_losses']}   re-acquisitions: {s['reacquisitions']}",
            f"Tracking error (LOCKED, boresight vs truth):",
            f"    avg {f(s['err_locked_avg_px'], ' px')}   rms {f(s['err_locked_rms_px'], ' px')}   "
            f"p95 {f(s['err_locked_p95_px'], ' px')}   max {f(s['err_locked_max_px'], ' px')}",
            f"Kalman estimate error (LOCKED): avg {f(s['est_err_locked_avg_px'], ' px')}",
            f"Detection rate:       {f(s['detection_rate_pct'], ' %')}",
            f"Time in state:        " + ", ".join(f"{k} {v:.2f}s" for k, v in s["time_in_state_s"].items()),
            f"Final state:          {s['final_state']}",
            "",
            "Verdict: " + self._verdict(s),
        ]
        return "\n".join(lines)

    @staticmethod
    def _verdict(s: Dict[str, Any]) -> str:
        if not s.get("acquired"):
            return "beacon NOT acquired in this run."
        return (f"beacon acquired after {s['acquisition_time_s']:.2f} s and held LOCKED for "
                f"{s['lock_retention_pct']:.1f} % of the remaining run with a mean pointing error of "
                f"{s['err_locked_avg_px']:.2f} px (max {s['err_locked_max_px']:.2f} px).")

    # ----------------------------------------------------------------- export
    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        """Return the per-frame log as CSV text (and write it if ``path``)."""
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        for r in self.records:
            row = []
            for k, v in asdict(r).items():
                if isinstance(v, float):
                    v = _r(v)
                elif isinstance(v, bool):
                    v = int(v)
                row.append("" if v is None else v)
            w.writerow(row)
        text = buf.getvalue()
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8", newline="")
        return text

    def write_report(self, directory: Union[str, Path], config: Optional[Dict[str, Any]] = None,
                     prefix: str = "beaconlock") -> Dict[str, Path]:
        """Write ``<prefix>_frames.csv``, ``_summary.txt``, ``_summary.json`` (+ config)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "csv": d / f"{prefix}_frames.csv",
            "summary_txt": d / f"{prefix}_summary.txt",
            "summary_json": d / f"{prefix}_summary.json",
        }
        self.to_csv(paths["csv"])
        paths["summary_txt"].write_text(self.summary_text(), encoding="utf-8")
        paths["summary_json"].write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        if config is not None:
            paths["config"] = d / f"{prefix}_config.json"
            paths["config"].write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        return paths

    def report_zip_bytes(self, config: Optional[Dict[str, Any]] = None) -> bytes:
        """In-memory ZIP: frames.csv, summary.txt, summary.json (+ config.json)."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("frames.csv", self.to_csv())
            z.writestr("summary.txt", self.summary_text())
            z.writestr("summary.json", json.dumps(self.summary(), indent=2))
            if config is not None:
                z.writestr("config.json", json.dumps(config, indent=2, default=str))
        return buf.getvalue()
