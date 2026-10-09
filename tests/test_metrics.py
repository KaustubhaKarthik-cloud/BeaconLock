"""Tests for metrics.MetricsRecorder."""

import csv
import io
import json
import zipfile

import pytest

from metrics import FrameRecord, MetricsRecorder


def rec(i, state, err, proc=2.0, det=True):
    return FrameRecord(i, i / 10, state, det, 1.0, 2.0, 10.0, 10.0, 0.0, 0.0, 3.0, 4.0, err, 0.5,
                       True, proc)


def build():
    m = MetricsRecorder()
    m.begin({"scenario": "unit"})
    seq = ["SEARCHING"] * 5 + ["ACQUIRING"] * 3 + ["LOCKED"] * 10 + ["LOST"] * 2 + ["ACQUIRING"] + ["LOCKED"] * 4
    errs = [500] * 8 + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [50, 60, 40] + [2, 2, 2, 2]
    for i, (s, e) in enumerate(zip(seq, errs)):
        m.log(rec(i, s, e, proc=float(i % 4 + 1), det=s != "SEARCHING"))
    return m


def test_summary_values():
    s = build().summary()
    assert s["frames"] == 25
    assert s["acquisition_time_s"] == pytest.approx(0.8)
    assert s["lock_pct"] == pytest.approx(100 * 14 / 25)
    assert s["lock_retention_pct"] == pytest.approx(100 * 14 / 17, abs=0.01)
    assert s["err_locked_max_px"] == 10
    assert s["err_locked_avg_px"] == pytest.approx((55 + 8) / 14, abs=1e-3)
    assert s["lock_losses"] == 1 and s["reacquisitions"] == 1
    assert s["proc_ms_max"] == 4 and s["fps_processing_capacity"] == pytest.approx(1000 / 2.44, abs=0.1)
    assert s["synthetic"] is True and s["final_state"] == "LOCKED"


def test_never_locked():
    m = MetricsRecorder()
    for i in range(5):
        m.log(rec(i, "SEARCHING", 400, det=False))
    s = m.summary()
    assert s["acquired"] is False and s["acquisition_time_s"] is None and s["lock_pct"] == 0
    assert "NOT acquired" in m.summary_text()


def test_empty():
    m = MetricsRecorder()
    assert m.summary()["frames"] == 0
    assert "No frames" in m.summary_text()


def test_csv_and_report_files(tmp_path):
    m = build()
    rows = list(csv.DictReader(io.StringIO(m.to_csv())))
    assert len(rows) == 25 and rows[0]["state"] == "SEARCHING" and rows[0]["detected"] == "0"
    paths = m.write_report(tmp_path, config={"a": 1})
    assert all(p.exists() for p in paths.values())
    assert "SYNTHETIC" in paths["summary_txt"].read_text()
    assert json.loads(paths["summary_json"].read_text())["frames"] == 25


def test_zip_bundle():
    z = zipfile.ZipFile(io.BytesIO(build().report_zip_bytes({"seed": 1})))
    assert set(z.namelist()) == {"frames.csv", "summary.txt", "summary.json", "config.json"}
    assert "Acquisition time" in z.read("summary.txt").decode()
