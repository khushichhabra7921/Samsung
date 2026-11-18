"""Render original source intervals as a silent, browser-compatible MP4."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import av

from .contracts import Report

FPS = 30


class RenderError(RuntimeError):
    """The selected intervals could not be rendered safely."""


def _run(arguments: list[str]) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *arguments],
            check=True, capture_output=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderError("Could not render the highlights. Check FFmpeg and the source MP4.") from exc


def render(source: Path, destination: Path, report: Report) -> None:
    if not report.highlights:
        raise RenderError("There are no selected intervals to render.")
    source = source.resolve()
    if source == destination.resolve():
        raise RenderError("The highlight destination must differ from the input video.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".highlight-") as directory:
        staging = Path(directory)
        segments = []
        for index, clip in enumerate(report.highlights):
            frames = clip.length * FPS // 1000
            if frames == 0:
                continue
            part = staging / f"part-{index}.mp4"
            _run([
                "-ss", f"{clip.start_ms / 1000:.3f}", "-protocol_whitelist", "file,pipe",
                "-threads", "2", "-i", str(source), "-map", "0:v:0", "-an", "-sn",
                "-t", f"{clip.length / 1000:.3f}",
                "-vf", "scale=w='min(1280,iw)':h=-2,setsar=1,fps=30,format=yuv420p",
                "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-threads", "2",
                "-bf", "0", "-video_track_timescale", "30000", "-map_metadata", "-1", str(part),
            ])
            with av.open(str(part)) as container:
                if not container.streams.video:
                    continue
                stream = container.streams.video[0]
                if stream.duration and stream.time_base:
                    segments.append(f"file '{part.name}'\nduration {float(stream.duration * stream.time_base):.9f}\n")
        if not segments:
            raise RenderError("The intervals are shorter than one export frame.")
        playlist = staging / "segments.txt"
        playlist.write_text("".join(segments), encoding="utf-8")
        output = staging / "result.mp4"
        _run([
            "-f", "concat", "-safe", "1", "-i", str(playlist), "-c", "copy",
            "-movflags", "+faststart", "-video_track_timescale", "30000", str(output),
        ])
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            if not stream.duration or stream.duration * stream.time_base * 1000 > report.budget_ms:
                raise RenderError("The encoded highlight exceeded the duration budget.")
        output.replace(destination)
