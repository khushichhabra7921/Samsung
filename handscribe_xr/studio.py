"""Same-origin local upload and analysis with bounded, ephemeral job storage."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4

from egovision.ingest import VideoError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

from .contracts import ROOT, write_json
from .pipeline import Options, analyze
from .recognition import ModelSetupError
from .render import RenderError, render

log = logging.getLogger(__name__)
MAX_UPLOAD = 512 * 1024 * 1024
MAX_RESULTS = 4
RETENTION_S = 3600
PRIVATE = {"Cache-Control": "private, no-store"}


@dataclass
class Job:
    id: str
    name: str
    directory: Path
    state: str = "uploading"
    stage: str = "Receiving video"
    fraction: float = 0
    error: str | None = None
    export_error: str | None = None
    finished: float = 0

    def snapshot(self) -> dict:
        current = {
            "id": self.id, "name": self.name, "state": self.state,
            "stage": self.stage, "fraction": self.fraction, "error": self.error,
        }
        if self.state == "complete":
            root = f"api/jobs/{self.id}"
            current.update(
                source=f"{root}/source.mp4", report=f"{root}/report.json", trace=f"{root}/trace.json",
                highlight=f"{root}/highlights.mp4" if (self.directory / "highlights.mp4").is_file() else None,
                export_error=self.export_error,
            )
        return current


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.url.netloc:
        raise HTTPException(403, "Open the Studio on the same local address as its API.")


def create_app(options: Options | None = None, *, max_upload: int = MAX_UPLOAD) -> FastAPI:
    options = options if options is not None else Options(allow_short=True)
    jobs: dict[str, Job] = {}
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with TemporaryDirectory(prefix="handscribe-xr-") as directory, ThreadPoolExecutor(max_workers=1) as executor:
            app.state.root = Path(directory)
            app.state.executor = executor
            yield
        jobs.clear()

    app = FastAPI(title="HandScribe XR", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self'; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers.update(PRIVATE)
        return response

    def remove(job: Job) -> None:
        jobs.pop(job.id, None)
        shutil.rmtree(job.directory, ignore_errors=True)

    def expire(maximum: int = MAX_RESULTS) -> None:
        now = time.monotonic()
        done = sorted((job for job in jobs.values() if job.finished), key=lambda job: job.finished)
        for index, job in enumerate(done):
            if now - job.finished > RETENTION_S or len(done) - index > maximum:
                remove(job)

    def process(job: Job) -> None:
        def progress(stage: str, fraction: float) -> None:
            with lock:
                job.stage = stage
                job.fraction = fraction

        state, error = "failed", None
        try:
            source = job.directory / "source.mp4"
            analysis = analyze(source, Path(job.name).stem, options, progress)
            write_json(job.directory / "report.json", analysis.report.model_dump(mode="json"))
            write_json(job.directory / "trace.json", {"motion": analysis.motion, "hands": analysis.hands}, indent=None)
            if analysis.report.highlights:
                progress("Rendering highlight MP4", 0.97)
                try:
                    render(source, job.directory / "highlights.mp4", analysis.report)
                except RenderError as exc:
                    job.export_error = str(exc)
            state = "complete"
        except (VideoError, ModelSetupError) as exc:
            error = str(exc)
        except Exception:
            log.exception("Analysis failed for local job %s", job.id)
            error = "Analysis failed. See the local server log for details."
        finally:
            with lock:
                job.state = state
                job.error = error
                job.stage = "Ready" if state == "complete" else "Analysis failed"
                job.fraction = 1 if state == "complete" else 0
                job.finished = time.monotonic()
                if state == "failed":
                    (job.directory / "source.mp4").unlink(missing_ok=True)

    @app.get("/api/health")
    def health():
        return {
            "status": "ok", "mode": options.mode, "max_upload_bytes": max_upload,
            "hand_model": (options.weights / "hand_landmarker.task").is_file(),
        }

    @app.post("/api/jobs", status_code=202)
    async def upload(request: Request, name: str):
        _same_origin(request)
        filename = Path(name).name
        if len(filename) > 180 or any(ord(char) < 32 for char in filename) or Path(filename).suffix.lower() != ".mp4":
            raise HTTPException(415, "Choose an MP4 video.")
        if request.headers.get("content-type", "").partition(";")[0] != "video/mp4":
            raise HTTPException(415, "Send the MP4 as a video/mp4 request body.")
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise HTTPException(400, "Invalid upload length.") from exc
        if declared > max_upload:
            raise HTTPException(413, "The MP4 exceeds the upload size limit.")

        with lock:
            expire(MAX_RESULTS - 1)
            if any(job.state in ("uploading", "running") for job in jobs.values()):
                raise HTTPException(429, "The local engine is busy. Try again when its current analysis completes.")
            id_ = uuid4().hex
            directory = app.state.root / id_
            directory.mkdir(mode=0o700)
            job = Job(id_, filename, directory)
            jobs[id_] = job

        try:
            size = 0
            async with asyncio.timeout(180):
                with (directory / "source.mp4").open("wb") as stream:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > max_upload:
                            raise HTTPException(413, "The MP4 exceeds the upload size limit.")
                        stream.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded MP4 is empty.")
        except BaseException as exc:
            with lock:
                remove(job)
            if isinstance(exc, (ClientDisconnect, TimeoutError)):
                raise HTTPException(408, "The video upload was interrupted.") from exc
            raise

        with lock:
            job.state = "running"
            job.stage = "Preparing local analysis"
            app.state.executor.submit(process, job)
            return job.snapshot()

    def find(id_: str) -> Job:
        expire()
        job = jobs.get(id_)
        if job is None:
            raise HTTPException(404, "This local analysis has expired or was removed.")
        return job

    @app.get("/api/jobs/{id_}")
    def status(id_: str):
        with lock:
            return find(id_).snapshot()

    @app.get("/api/jobs/{id_}/{file}")
    def artifact(id_: str, file: str):
        files = {
            "report.json": "application/json", "trace.json": "application/json",
            "highlights.mp4": "video/mp4", "source.mp4": "video/mp4",
        }
        with lock:
            job = find(id_)
            path = job.directory / file
            if job.state != "complete" or file not in files or not path.is_file():
                raise HTTPException(404, "The requested analysis file is unavailable.")
        return FileResponse(path, media_type=files[file])

    @app.delete("/api/jobs/{id_}", status_code=204)
    def delete(id_: str, request: Request):
        _same_origin(request)
        with lock:
            job = find(id_)
            if not job.finished:
                raise HTTPException(409, "Wait for the current analysis to finish before removing it.")
            remove(job)

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="site")
    return app
