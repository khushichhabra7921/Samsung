"""One-pass, timestamp-aware video analysis with reusable BIOS hand tracking."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
from egovision.config import HandsConfig, QCConfig
from egovision.hands import make_backend
from egovision.hands.postprocess import bridge_dropouts
from egovision.hands.tracking import smooth_tracks, track_hands
from egovision.ingest import Frame, VideoError
from egovision.qc import frame_qc
from egovision.schema import FrameHands, VideoMeta

from .contracts import MIN_HIGHLIGHT_MS, ROOT, Diagnostics, Mode, Report, Status, budget_ms
from .recognition import ModelSetupError, Predictions, SemanticModels, Vocabulary, load_models
from .selection import Sample, Timeline, select

SAMPLE_EVERY_MS = 330
MAX_DURATION_MS = 35 * 60 * 1000
MAX_PIXELS = 4096 * 2304
MAX_FRAMES = 252_000
Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class Options:
    mode: Mode = "preview"
    weights: Path = ROOT / "weights"
    vocabulary: Path | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    allow_short: bool = False

    def models(self) -> SemanticModels | None:
        if self.mode == "preview":
            return None
        if self.vocabulary is None:
            raise ModelSetupError("Semantic mode requires a Track 04 label vocabulary via --vocabulary.")
        for asset in ("hand_landmarker.task", "owlv2/config.json", "siglip/config.json"):
            if not (self.weights / asset).is_file():
                raise ModelSetupError(f"Semantic mode requires {asset} under {self.weights}.")
        return load_models(self.weights.resolve(), Vocabulary.load(self.vocabulary), self.device)


DEFAULT_OPTIONS = Options()


@dataclass(slots=True)
class Keyframe:
    index: int
    sample: Sample
    qc: tuple[float, float, float]
    prediction: Predictions | None = None


@dataclass(slots=True)
class Analysis:
    report: Report
    motion: list[list[int | float]]
    hands: list[dict]
    video_categories: tuple[tuple[str, float], ...] = ()


def _display_frame(frame: av.VideoFrame, rotation: int) -> np.ndarray:
    scale = min(1.0, 960 / max(frame.width, frame.height))
    resized = frame.reformat(
        width=max(2, round(frame.width * scale)),
        height=max(2, round(frame.height * scale)),
        format="bgr24",
    ).to_ndarray()
    rotations = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    return cv2.rotate(resized, rotations[rotation]) if rotation in rotations else resized


def _preview_embedding(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (32, 18), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    spatial = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten() / 255
    histogram = np.sqrt(histogram / max(float(histogram.sum()), 1))
    spatial -= float(np.mean(spatial))
    spatial /= max(float(np.linalg.norm(spatial)), 1)
    descriptor = np.concatenate((histogram, spatial))
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-6)


def _scan(
    path: Path, options: Options, semantic: SemanticModels | None, progress: Progress,
) -> tuple[VideoMeta, list[Keyframe], list[FrameHands], HandsConfig, int]:
    if not path.is_file():
        raise VideoError("The input video could not be found.")
    qc = QCConfig()
    hands_config = HandsConfig(model_path=options.weights / "hand_landmarker.task")
    frames: list[FrameHands] = []
    keys: list[Keyframe] = []
    batch: list[np.ndarray] = []
    prev_gray = None
    motion = 0.0
    motion_count = 0
    next_key = 0
    next_progress = 0
    processed = 0
    last_pts = -1.0
    last_frame_ms = 0.0
    meta: VideoMeta | None = None

    try:
        with ExitStack() as resources:
            container = resources.enter_context(av.open(str(path), options={"protocol_whitelist": "file,pipe"}))
            if not container.streams.video or "mp4" not in container.format.name.split(","):
                raise VideoError("Choose a decodable MP4 with a video track.")
            stream = container.streams.video[0]
            if stream.codec_context.width * stream.codec_context.height > MAX_PIXELS:
                raise VideoError("The input exceeds the supported 4K frame size.")
            stream.thread_type = "AUTO"
            stream.thread_count = 2
            fps = float(stream.average_rate or stream.base_rate or 30)
            fps = fps if 1 <= fps <= 240 else 30
            origin = (container.start_time or 0) / av.time_base
            declared_ms = (container.duration or 0) / av.time_base * 1000
            if declared_ms > MAX_DURATION_MS + 1000:
                raise VideoError("The maximum video duration is 35 minutes.")

            hand_backend = None
            rotation = 0
            scale = 1.0

            def flush() -> None:
                if semantic is not None and batch:
                    results = semantic.predict(batch, scale)
                    for key, result in zip(keys[-len(batch):], results, strict=True):
                        key.prediction = result
                        key.sample.embedding = result.embedding
                    batch.clear()

            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_FRAMES:
                    raise VideoError("The input exceeds the decoded frame limit.")
                if frame.width * frame.height > MAX_PIXELS:
                    raise VideoError("The input exceeds the supported 4K frame size.")
                current = frame.time - origin if frame.time is not None else max(0, last_pts) + 1 / fps
                if not math.isfinite(current) or current < 0 or current <= last_pts:
                    continue
                last_pts = current
                frame_ms = int(current * 1000 + 1e-6)
                last_frame_ms = current * 1000 + (float(frame.duration * frame.time_base) * 1000
                                                   if frame.duration and frame.time_base else 1000 / fps)
                if last_frame_ms > MAX_DURATION_MS + 1:
                    raise VideoError("The maximum video duration is 35 minutes.")
                if index % 2:
                    continue

                if meta is None:
                    rotation = frame.rotation % 360
                    width, height = (frame.height, frame.width) if rotation in (90, 270) else (frame.width, frame.height)
                    meta = VideoMeta(path=str(path), name=path.stem, width=width, height=height, fps=fps,
                                     n_frames=0, duration_s=0, codec=stream.codec_context.name)
                    if hands_config.model_path.is_file():
                        try:
                            hand_backend = resources.enter_context(make_backend(hands_config, meta))
                        except (ImportError, OSError, RuntimeError, ValueError) as exc:
                            raise ModelSetupError("Could not initialize the local MediaPipe hand model.") from exc

                image = _display_frame(frame, rotation)
                scale = image.shape[1] / meta.width
                processed += 1
                if hand_backend is not None:
                    hands = hand_backend.process(Frame(index, current, image, scale))
                    frames.append(FrameHands(frame=index, t=current, hands=hands))

                gray = cv2.cvtColor(cv2.resize(image, (80, 45), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    motion += float(cv2.absdiff(gray, prev_gray).mean()) / 255
                    motion_count += 1
                prev_gray = gray
                if frame_ms >= next_key:
                    sample = Sample(frame_ms, _preview_embedding(image), motion / max(1, motion_count))
                    keys.append(Keyframe(index, sample, frame_qc(image, qc.size)))
                    if semantic is not None:
                        batch.append(image)
                        if len(batch) >= semantic.objects.batch_size:
                            flush()
                    next_key = (frame_ms // SAMPLE_EVERY_MS + 1) * SAMPLE_EVERY_MS
                    motion = 0.0
                    motion_count = 0
                if frame_ms >= next_progress:
                    progress("Analyzing footage", min(0.83, 0.05 + 0.78 * frame_ms / max(declared_ms, 1)))
                    next_progress = frame_ms + 1000

            flush()
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoError("The MP4 could not be fully decoded.") from exc

    if meta is None or not keys or last_frame_ms < 1:
        raise VideoError("The MP4 did not contain any readable video frames.")
    if declared_ms - last_frame_ms > max(1000, declared_ms * 0.02):
        raise VideoError("Video decoding ended early. The MP4 may be truncated.")
    duration = int(last_frame_ms + 1e-6)
    return meta.model_copy(update={"n_frames": index + 1, "duration_s": duration / 1000}), keys, frames, hands_config, processed


def _prepare_samples(
    meta: VideoMeta, keys: list[Keyframe], frames: list[FrameHands], config: HandsConfig,
    semantic: SemanticModels | None, processed: int,
) -> tuple[Diagnostics, list[dict]]:
    qc = QCConfig()
    sharpness = max(1, float(np.median([key.qc[0] for key in keys])))
    motion_scale = max(0.015, float(np.percentile([key.sample.motion for key in keys], 90)))
    blurry = covered = visible = 0
    filled = {}
    if frames:
        tracked = track_hands(frames, meta.fps, meta.width, meta.height, config)
        smoothed = smooth_tracks(tracked, config)
        filled = {frame.frame: frame.hands for frame in bridge_dropouts(smoothed, config.max_interp_gap)}
    previous: dict[int, tuple[int, float, float]] = {}
    landmarks = []
    for key in keys:
        sample = key.sample
        sharp, light, variation = key.qc
        bad_blur = sharp < qc.blur_rel_threshold * sharpness
        bad_cover = light < qc.covered_max_brightness or variation < qc.covered_max_std
        blurry += bad_blur
        covered += bad_cover
        sample.quality = 0.0 if bad_cover else 0.45 if bad_blur else 1.0
        sample.motion = min(1.0, sample.motion / motion_scale) * sample.quality
        hands = filled.get(key.index, [])
        detected = [hand for hand in hands if not hand.interp]
        visible += bool(detected)
        speeds = []
        view = []
        for hand in hands:
            if hand.id is None:
                continue
            x, y = hand.kp2d[0]
            prior = previous.get(hand.id)
            if prior is not None and 0 < sample.time_ms - prior[0] <= 750:
                displacement = math.hypot((x - prior[1]) / meta.width, (y - prior[2]) / meta.height)
                speeds.append(min(1.0, 4500 * displacement / (sample.time_ms - prior[0])))
            previous[hand.id] = (sample.time_ms, x, y)
            view.append({
                "id": hand.id,
                "side": hand.side,
                "points": [[round(px / meta.width, 4), round(py / meta.height, 4)] for px, py in hand.kp2d],
            })
        if frames:
            landmarks.append({"time_ms": sample.time_ms, "hands": view})
        sample.hand_motion = max(speeds) if speeds else None
        if semantic is not None and key.prediction is not None and not bad_cover:
            semantic.label(sample, key.prediction, detected)

    return Diagnostics(
        sampled_frames=processed,
        keyframes=len(keys),
        blur_pct=round(100 * blurry / len(keys), 1),
        covered_pct=round(100 * covered / len(keys), 1),
        hand_visibility_pct=round(100 * visible / len(keys), 1) if frames else None,
    ), landmarks


def analyze(
    path: Path, video_id: str, options: Options = DEFAULT_OPTIONS, progress: Progress | None = None,
) -> Analysis:
    report_progress = progress if progress is not None else lambda stage, fraction: None
    report_progress("Preparing local analysis", 0.03)
    semantic = options.models()
    meta, keys, frames, config, processed = _scan(path, options, semantic, report_progress)
    report_progress("Scoring evidence", 0.86)
    diagnostics, hands = _prepare_samples(meta, keys, frames, config, semantic, processed)
    duration = round(meta.duration_s * 1000)
    budget = budget_ms(duration)
    timeline = Timeline([key.sample for key in keys], duration)
    actions = timeline.actions()
    status: Status = "complete"
    minimum = MIN_HIGHLIGHT_MS
    if budget < minimum:
        status = "short_clip_preview" if options.allow_short and budget > 0 else "ineligible_short"
        minimum = budget if status == "short_clip_preview" else MIN_HIGHLIGHT_MS
    highlights = ()
    if budget >= minimum:
        report_progress("Selecting highlights", 0.92)
        highlights = select(timeline.candidates(actions, budget, minimum), budget, minimum)
    report = Report(
        video_id=video_id, duration_ms=duration, duration_sec=duration / 1000, budget_ms=budget,
        highlight_duration_ms=sum(clip.length for clip in highlights), mode=options.mode,
        device=semantic.device if semantic else "cpu", status=status, highlights=highlights,
        actions=actions, video_label=timeline.video_label(), diagnostics=diagnostics,
    )
    category_vectors = [
        key.prediction.category_scores
        for key in keys
        if key.prediction is not None and key.prediction.category_scores is not None
    ]
    video_categories: tuple[tuple[str, float], ...] = ()
    if semantic is not None and semantic.vocab.labels and category_vectors:
        mean_scores = np.mean(np.stack(category_vectors), axis=0)
        order = np.argsort(-mean_scores)
        video_categories = tuple(
            (semantic.vocab.labels[int(index)], float(mean_scores[int(index)]))
            for index in order
        )
    stride = max(1, math.ceil(len(keys) / 150))
    motion = [[key.sample.time_ms, round(key.sample.motion, 3)] for key in keys[::stride]]
    return Analysis(report=report, motion=motion, hands=hands, video_categories=video_categories)
