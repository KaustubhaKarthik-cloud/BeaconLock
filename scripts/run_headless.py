#!/usr/bin/env python3
"""
scripts/run_headless.py -- end-to-end BeaconLock run without any UI.

Runs the full closed loop (scene -> camera -> disturbances -> detector ->
tracker -> PID -> gimbal) for N simulated seconds, prints the performance
summary, writes the report (CSV + summary.txt/json + config) and, with
``--video``, an H.264 MP4 + small GIF of the dashboard view.

Exit status is 0 only if the beacon was acquired (useful for CI).

Examples (from the repo root, any OS)::

    python scripts/run_headless.py                                  # demo config, 20 s
    python scripts/run_headless.py --seconds 30 --turbulence 80 --video
    python scripts/run_headless.py --trajectory random_waypoint --speed 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.engine import SimulationEngine, build_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Headless BeaconLock closed-loop run.")
    ap.add_argument("--config", help="engine JSON config (default: built-in demo)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--trajectory", choices=["sinusoidal", "linear", "random_waypoint"])
    ap.add_argument("--speed", type=float, default=220.0, help="beacon speed px/s (with --trajectory)")
    for k in ("turbulence", "vibration", "noise", "blur"):
        ap.add_argument(f"--{k}", type=float, help=f"{k} intensity 0-100")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default=str(REPO / "output" / "headless"))
    ap.add_argument("--video", action="store_true", help="also write MP4 + GIF of the dashboard view")
    args = ap.parse_args(argv)

    if args.config:
        import json
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        cfg = None
    if args.trajectory or args.seed is not None:
        cfg = build_config(args.trajectory or "sinusoidal", args.speed, seed=args.seed, base=cfg)
    eng = SimulationEngine(cfg)
    dist = {k: getattr(args, k) for k in ("turbulence", "vibration", "noise", "blur") if getattr(args, k) is not None}
    if dist:
        eng.set_disturbances(**dist)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    callback = None
    sink = None
    if args.video:
        from scripts.test_scene import VideoSink, make_gif  # reuse the ffmpeg helpers
        from sim.overlay import Minimap, compose_dashboard_frame

        mm = Minimap(eng, width=480)
        probe = compose_dashboard_frame(eng, mm)
        sink = VideoSink(out / "headless_run.mp4", probe.shape[1], probe.shape[0], eng.fps)

        def callback(_res):
            sink.write(compose_dashboard_frame(eng, mm))

    print(f"Running {args.seconds:.1f} s of SYNTHETIC simulation at {eng.fps:.0f} FPS "
          f"(disturbances: {eng.disturbances.intensities}) ...")
    eng.run(args.seconds, callback=callback)
    paths = eng.write_report(out, prefix="headless")
    if sink is not None:
        sink.close()
        if sink.ffmpeg:
            make_gif(sink.ffmpeg, out / "headless_run.mp4", out / "headless_run.gif", 480, 8)
    print()
    print(eng.metrics.summary_text())
    print("\nReport files:")
    for p in list(paths.values()) + ([out / "headless_run.mp4", out / "headless_run.gif"] if args.video else []):
        if Path(p).exists():
            print(f"  {Path(p).resolve()}")
    return 0 if eng.summary().get("acquired") else 1


if __name__ == "__main__":
    sys.exit(main())
