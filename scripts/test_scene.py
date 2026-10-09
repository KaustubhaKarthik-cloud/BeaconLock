#!/usr/bin/env python3
"""
scripts/test_scene.py -- render N seconds of the BeaconLock synthetic world.

Step-1 deliverable: proves the simulation core works *before* the camera is
built.  It steps a :class:`sim.Scene`, renders every frame, optionally draws a
ground-truth overlay (trail + ring + label per beacon, HUD with time) and writes

* ``scene_<name>.mp4``        -- H.264 / yuv420p / faststart (plays in browsers)
* ``scene_<name>.gif``        -- downscaled preview (default 640 px wide, 10 fps)
* ``scene_<name>_kf{0..3}.png`` -- 4 full-resolution keyframes (t = 0, 1/3, 2/3, end)

Video encoding uses the ``ffmpeg`` binary (system PATH, else the one bundled
with the ``imageio-ffmpeg`` package).  If neither exists it falls back to
OpenCV's ``mp4v`` writer (not browser-playable) and skips the GIF.

The overlay is drawn on a *copy* of the frame; the simulation output itself
only contains the synthetic sky, the beacons and the "SYNTHETIC SCENE" mark.

Usage (from the repo root)::

    python scripts/test_scene.py                         # default config, 5 s @ 30 FPS
    python scripts/test_scene.py --config configs/multi_beacon.json
    python scripts/test_scene.py --seconds 10 --no-trail --out output/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # allow running without installing the package

from sim import Scene  # noqa: E402

# Distinct BGR colours for up to 6 beacons' overlays.
PALETTE = [(0, 200, 255), (80, 255, 80), (255, 140, 60), (255, 80, 255), (60, 220, 220), (180, 180, 255)]


def find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg executable (system first, then imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoSink:
    """Write BGR frames to an H.264 MP4 through an ffmpeg pipe (cv2 fallback)."""

    def __init__(self, path: Path, w: int, h: int, fps: float, crf: int = 23) -> None:
        self.path = path
        self.ffmpeg = find_ffmpeg()
        self.proc: Optional[subprocess.Popen] = None
        self.writer: Optional[cv2.VideoWriter] = None
        if self.ffmpeg:
            cmd = [
                self.ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("WARNING: ffmpeg not found; using OpenCV mp4v (may not play in browsers)")
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc:
            assert self.proc.stdin is not None
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        else:
            assert self.writer is not None
            self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.proc:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed while encoding the MP4")
        elif self.writer:
            self.writer.release()


def make_gif(ffmpeg: str, mp4: Path, gif: Path, width: int, fps: int) -> None:
    """Convert the MP4 to a compact palette GIF (two-pass palettegen/paletteuse)."""
    vf = (
        f"fps={fps},scale={width}:-2:flags=area,split[a][b];"
        f"[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)], check=True)


def draw_overlay(
    frame_bgr: np.ndarray,
    scene: Scene,
    trails: Dict[str, Deque[Tuple[int, int]]],
    show_trail: bool,
    fps: float,
) -> np.ndarray:
    """Draw ground-truth trail, ring and label for each beacon plus a HUD."""
    out = frame_bgr.copy()
    scale = max(scene.width / 1920.0, 0.4)
    for i, st in enumerate(scene.ground_truth()):
        col = PALETTE[i % len(PALETTE)]
        p = (int(round(st.x)), int(round(st.y)))
        if show_trail:
            pts = trails[st.id]
            for a, b in zip(pts, list(pts)[1:]):
                # Skip segments that jump across the world (wrap-around).
                if abs(a[0] - b[0]) < scene.width / 2 and abs(a[1] - b[1]) < scene.height / 2:
                    cv2.line(out, a, b, col, max(1, int(2 * scale)), cv2.LINE_AA)
        r = int(max(14, 4 * st.sigma) * scale)
        cv2.circle(out, p, r, col, max(1, int(2 * scale)), cv2.LINE_AA)
        label = f"{st.id} ({st.x:.0f},{st.y:.0f})"
        fs, th = 0.6 * scale, max(1, int(1.5 * scale))
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        lx = p[0] + r + 4 if p[0] + r + 4 + tw < scene.width else p[0] - r - 4 - tw  # keep label on-screen
        ly = max(p[1] - r, int(20 * scale))
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    hud = [
        f"BeaconLock | step 1: synthetic world | scene '{scene.name}'",
        f"t = {scene.t:5.2f} s   frame {scene.frame_index:4d}   {fps:.0f} FPS   "
        f"world {scene.width}x{scene.height}   beacons {len(scene.beacons)}",
        "Overlay = ground truth (not part of sensor image)",
    ]
    for k, line in enumerate(hud):
        cv2.putText(out, line, (int(16 * scale), int((34 + 30 * k) * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * scale, (200, 200, 200), max(1, int(1.5 * scale)), cv2.LINE_AA)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render the BeaconLock synthetic scene to MP4/GIF/PNG.")
    ap.add_argument("--config", default=str(REPO / "configs" / "default_scene.json"))
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=str(REPO / "output"))
    ap.add_argument("--no-trail", action="store_true", help="disable ground-truth trail overlay")
    ap.add_argument("--no-overlay", action="store_true", help="write raw sensor frames only")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--trail-len", type=int, default=90, help="trail length in frames")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = Scene.from_json(args.config)
    tag = scene.name
    n_frames = int(round(args.seconds * args.fps))
    dt = 1.0 / args.fps
    keyframes = {0, n_frames // 3, (2 * n_frames) // 3, n_frames - 1}

    mp4 = out_dir / f"scene_{tag}.mp4"
    gif = out_dir / f"scene_{tag}.gif"
    sink = VideoSink(mp4, scene.width, scene.height, args.fps)
    trails: Dict[str, Deque[Tuple[int, int]]] = {b.id: deque(maxlen=args.trail_len) for b in scene.beacons}
    kf_paths: List[Path] = []

    print(scene)
    t0 = time.perf_counter()
    render_ms: List[float] = []
    for i in range(n_frames):
        if i > 0:
            scene.step(dt)  # frame 0 shows t = 0
        r0 = time.perf_counter()
        frame = scene.render_bgr()
        render_ms.append((time.perf_counter() - r0) * 1e3)
        for st in scene.ground_truth():
            trails[st.id].append((int(round(st.x)), int(round(st.y))))
        if not args.no_overlay:
            frame = draw_overlay(frame, scene, trails, not args.no_trail, args.fps)
        sink.write(frame)
        if i in keyframes:
            p = out_dir / f"scene_{tag}_kf{len(kf_paths)}_t{scene.t:.2f}s.png"
            cv2.imwrite(str(p), frame)
            kf_paths.append(p)
    sink.close()
    elapsed = time.perf_counter() - t0

    if sink.ffmpeg:
        make_gif(sink.ffmpeg, mp4, gif, args.gif_width, args.gif_fps)
    else:
        gif = None  # type: ignore[assignment]

    print(f"Rendered {n_frames} frames ({args.seconds:.1f} s sim @ {args.fps:.0f} FPS) in {elapsed:.2f} s wall; "
          f"scene.render() mean {np.mean(render_ms):.1f} ms/frame")
    print("Final ground truth:")
    for st in scene.ground_truth():
        print(f"  {st.id}: x={st.x:.1f} y={st.y:.1f} vx={st.vx:.1f} vy={st.vy:.1f} px/s")
    print("Outputs:")
    for p in [mp4, gif, *kf_paths]:
        if p is not None and Path(p).exists():
            print(f"  {p}  ({Path(p).stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
