import json
import subprocess
import time
from pathlib import Path

import av
from fastapi.testclient import TestClient

from handscribe_xr.contracts import ROOT, Report
from handscribe_xr.pipeline import Options, analyze
from handscribe_xr.studio import create_app

VIDEO = ROOT / "web" / "assets" / "kitchen-demo.mp4"


def test_real_clip_strict_and_preview(tmp_path: Path):
    strict = analyze(VIDEO, "kitchen-demo", Options(weights=tmp_path))
    assert strict.report.status == "ineligible_short"
    assert strict.report.highlights == ()
    assert strict.report.budget_ms == 3499
    options = Options(allow_short=True, weights=tmp_path)
    first = analyze(VIDEO, "kitchen-demo", options)
    second = analyze(VIDEO, "kitchen-demo", options)
    assert first.report == second.report
    assert first.report.status == "short_clip_preview"
    assert first.report.highlight_duration_ms == first.report.budget_ms
    assert first.report.actions == ()
    assert first.motion == second.motion


def test_eligible_video_respects_minimum_and_budget(tmp_path: Path):
    footage = tmp_path / "longer.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-stream_loop", "2", "-i", str(VIDEO),
        "-map", "0:v:0", "-c", "copy", str(footage),
    ], check=True, timeout=10)
    report = analyze(footage, footage.stem, Options(weights=tmp_path)).report
    assert report.status == "complete"
    assert report.highlights
    assert all(clip.length >= 4000 for clip in report.highlights)
    assert report.highlight_duration_ms <= report.budget_ms
    assert Report.model_validate_json(report.model_dump_json()) == report


def test_local_upload_export_and_removal(tmp_path: Path):
    options = Options(allow_short=True, weights=tmp_path)
    with TestClient(create_app(options)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["mode"] == "preview"
        response = client.post(
            "/api/jobs?name=kitchen.mp4", content=VIDEO.read_bytes(), headers={"Content-Type": "video/mp4"},
        )
        assert response.status_code == 202
        path = f"/api/jobs/{response.json()['id']}"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = client.get(path).json()
            if job["state"] not in {"uploading", "running"}:
                break
            time.sleep(0.15)
        assert job["state"] == "complete", job
        report = client.get(f"/{job['report']}").json()
        assert report["video_id"] == "kitchen"
        assert report["highlight_duration_ms"] <= report["budget_ms"]
        assert client.get(f"/{job['source']}", headers={"Range": "bytes=0-63"}).status_code == 206
        export = client.get(f"/{job['highlight']}")
        assert export.status_code == 200
        cut = tmp_path / "cut.mp4"
        cut.write_bytes(export.content)
        with av.open(str(cut)) as container:
            stream = container.streams.video[0]
            assert 0 < stream.duration * stream.time_base * 1000 <= report["budget_ms"]

        assert client.delete(path).status_code == 204
        assert client.get(path).status_code == 404


def test_invalid_upload_is_bounded(tmp_path: Path):
    options = Options(weights=tmp_path)
    with TestClient(create_app(options, max_upload=16)) as client:
        headers = {"Content-Type": "video/mp4"}
        assert client.post("/api/jobs?name=a.mp4", content=b"1" * 17, headers=headers).status_code == 413
        assert client.post("/api/jobs?name=a.png", content=b"1", headers=headers).status_code == 415
        response = client.post("/api/jobs?name=a.mp4", content=b"a", headers={**headers, "Origin": "https://invalid.test"})
        assert response.status_code == 403

        response = client.post("/api/jobs?name=a.mp4", content=b"broken", headers=headers)
        assert response.status_code == 202
        path = f"/api/jobs/{response.json()['id']}"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get(path).json()
            if job["state"] != "running":
                break
            time.sleep(0.05)
        assert job["state"] == "failed"
        assert "MP4" in job["error"]


def test_bundled_results_come_from_source_timestamps():
    data = json.loads((VIDEO.parent / "kitchen-demo.json").read_text())
    trace = json.loads((VIDEO.parent / "kitchen-trace.json").read_text())
    Report.model_validate(data)
    assert data["schema_version"] == "handscribe.xr/v1"
    assert data["status"] == "short_clip_preview"
    assert 0 < data["highlight_duration_ms"] <= data["duration_ms"] * 0.15
    assert len(trace["motion"]) == data["diagnostics"]["keyframes"]
    assert any(len(hand["points"]) == 21 for frame in trace["hands"] for hand in frame["hands"])
