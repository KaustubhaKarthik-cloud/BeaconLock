"""Tests for the web layer (FastAPI app + SimRunner)."""

import io
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from web.app import create_app  # noqa: E402
from web.runner import SimRunner, validate_settings  # noqa: E402


@pytest.fixture()
def client_runner():
    runner = SimRunner()
    app = create_app(runner, start_thread=False)  # step manually: deterministic & fast
    with TestClient(app) as c:
        yield c, runner


def test_index_and_static_are_local(client_runner):
    c, _ = client_runner
    html = c.get("/").text
    assert "BeaconLock" in html and "SYNTHETIC" in html
    assert "http://" not in html and "https://" not in html      # no CDNs
    for asset in ("/static/app.js", "/static/style.css"):
        r = c.get(asset)
        assert r.status_code == 200 and "https://" not in r.text


def test_start_step_state_and_report(client_runner):
    c, runner = client_runner
    assert c.get("/api/state").json()["state"] == "IDLE"
    assert c.post("/api/start").json()["running"] is True
    for _ in range(150):
        runner.step_once()
    s = c.get("/api/state").json()
    assert s["state"] == "LOCKED" and s["acquisition_time_s"] is not None
    assert len(s["history"]["err"]) == 150 and s["synthetic"] is True
    r = c.get("/api/report")
    assert r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert {"frames.csv", "summary.txt", "summary.json"} <= set(z.namelist())
    assert "Acquisition time" in z.read("summary.txt").decode()


def test_images(client_runner):
    c, _ = client_runner
    for p in ("/api/frame.jpg", "/api/minimap.jpg"):
        r = c.get(p)
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_config_apply_validation_and_live_disturbances(client_runner):
    c, runner = client_runner
    r = c.post("/api/config", json={"trajectory": "linear", "speed": 300, "acquire_n": 3, "acquire_m": 5})
    assert r.status_code == 200 and r.json()["settings"]["trajectory"] == "linear"
    assert runner.engine.tracker.config.acquire_n == 3
    assert c.post("/api/config", json={"acquire_n": 9, "acquire_m": 2}).status_code == 400
    assert c.post("/api/config", json={"trajectory": "warp"}).status_code == 400
    r = c.post("/api/disturbances", json={"turbulence": 90})
    assert r.status_code == 200 and runner.engine.disturbances.intensities["turbulence"] == 90
    assert c.post("/api/disturbances", json={"speed": 5}).status_code == 400
    c.post("/api/stop")
    assert c.get("/api/state").json()["running"] is False
    c.post("/api/reset")
    assert c.get("/api/state").json()["state"] == "IDLE"


def test_validate_settings_defaults():
    s = validate_settings({})
    assert s["trajectory"] == "sinusoidal" and s["turbulence"] > 0
    with pytest.raises(ValueError):
        validate_settings({"bogus": 1})
