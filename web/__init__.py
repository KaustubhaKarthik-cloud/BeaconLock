"""
BeaconLock web dashboard (thin UI layer).

This is the ONLY package allowed to import a web framework (FastAPI).  It
wraps :class:`sim.engine.SimulationEngine` in a background thread
(:mod:`web.runner`) and serves a single-page dashboard whose assets are all
local (no CDNs), so it works fully offline.

Entry point: ``python run.py`` -> http://127.0.0.1:8000
"""
