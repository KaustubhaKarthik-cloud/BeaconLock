#!/usr/bin/env python3
"""
run.py -- start the BeaconLock web dashboard.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 8080 --open

Works on Windows, macOS and Linux; fully offline (no CDNs, no external APIs).
All scenes are SYNTHETIC.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BeaconLock dashboard server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    args = ap.parse_args(argv)

    import uvicorn

    from web.app import create_app

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print(" BeaconLock - FSOC coarse-alignment virtual testbed (SYNTHETIC)")
    print(f" Dashboard:  {url}")
    print(" Press Ctrl+C to quit.")
    print("=" * 60, flush=True)
    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning",
                timeout_graceful_shutdown=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
