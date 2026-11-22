"""Track 04 single-video submission entry point.

This adapter deliberately keeps the existing handscribe.xr/v1 development report
unchanged and converts it to the organiser's six-field output contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

from egovision.ingest import VideoError

from .contracts import write_json
from .pipeline import Options, analyze
from .recognition import ModelSetupError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "code" else PROJECT_ROOT
MANIFEST = PROJECT_ROOT / "manifest.csv"
CLASSES = PROJECT_ROOT / "supported_classes.json"
MIN_HIGHLIGHT_SEC = 4.0
BUDGET_FRACTION = 0.15


def load_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Track 04 manifest is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"video_id", "duration_sec", "dataset"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifest.csv must contain video_id,duration_sec,dataset.")
    result = {}
    for row in rows:
        video_id = row["video_id"].strip()
        if not video_id or video_id in result:
            raise ValueError(f"Manifest contains an empty or duplicate video_id: {video_id!r}")
        result[video_id] = row
    return result


def _seconds(milliseconds: int) -> float:
    return round(milliseconds / 1000.0, 2)


def build_submission_payload(
    analysis,
    *,
    team_id: str,
    video_id: str,
    dataset: str,
    manifest_duration: float,
    supported_classes: dict,
) -> dict:
    report = analysis.report
    duration = float(report.duration_sec)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Inference returned an invalid video duration.")
    if abs(duration - manifest_duration) > 0.5:
        raise ValueError(
            f"Decoded duration {duration:.3f}s differs from manifest "
            f"duration {manifest_duration:.3f}s by more than 0.5s."
        )

    labels = set(supported_classes.get("labels", []))
    verbs = set(supported_classes.get("verbs", []))
    nouns = set(supported_classes.get("nouns", []))
    if not labels or not verbs or not nouns:
        raise ValueError("supported_classes.json is missing official labels, verbs or nouns.")

    if dataset == "epic_kitchens_100":
        video_label = {"name": "cooking", "score": 1.0}
    elif dataset == "fpvsum":
        fpv_labels = labels - {"cooking"}
        ranked = [
            (name, float(score))
            for name, score in getattr(analysis, "video_categories", ())
            if name in fpv_labels and math.isfinite(float(score))
        ]
        if not ranked:
            raise ValueError(
                "No FPVSum category scores were produced. Run semantic mode with the "
                "official supported_classes.json and local OWLv2/SigLIP weights."
            )
        name, score = ranked[0]
        video_label = {"name": name, "score": max(0.0, min(1.0, score))}
    else:
        raise ValueError(f"Unknown manifest dataset: {dataset!r}")

    actions = []
    for action in report.actions:
        verb = action.label.verb if action.label.verb in verbs else "other"
        noun = action.label.noun if action.label.noun in nouns else "other"
        start, end = _seconds(action.start_ms), _seconds(action.end_ms)
        if not (0 <= start < end <= duration + 0.5):
            continue
        score = float(action.confidence)
        if not math.isfinite(score):
            continue
        actions.append({
            "segment": [start, end],
            "verb": verb,
            "noun": noun,
            "score": max(0.0, min(1.0, score)),
        })
    actions.sort(key=lambda item: (item["segment"][0], item["segment"][1]))
    if len(actions) > 2000:
        actions = actions[:2000]

    highlights = []
    for clip in report.highlights:
        start, end = _seconds(clip.start_ms), _seconds(clip.end_ms)
        if 0 <= start < end <= duration + 0.5:
            highlights.append({"segment": [start, end]})
    highlights.sort(key=lambda item: (item["segment"][0], item["segment"][1]))

    if not highlights:
        raise ValueError("Inference produced no highlight segments.")
    if not any(end - start >= MIN_HIGHLIGHT_SEC
               for item in highlights for start, end in [item["segment"]]):
        raise ValueError("No highlight is at least 4.0 seconds; output cannot satisfy the student-kit rules.")
    previous_end = -1.0
    total = 0.0
    for item in highlights:
        start, end = item["segment"]
        if start < previous_end:
            raise ValueError("Highlights overlap or are not strictly chronological.")
        previous_end = end
        total += end - start
    if total > duration * BUDGET_FRACTION + 1e-9:
        raise ValueError("Rounded highlight segments exceed the 15% duration budget.")

    return {
        "team_id": team_id,
        "video_id": video_id,
        "duration_sec": round(duration, 2),
        "label": video_label,
        "actions": actions,
        "highlights": highlights,
    }


def run(video_path: Path, output_path: Path) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    video_id = video_path.stem
    manifest = load_manifest()
    if video_id not in manifest:
        raise ValueError(
            f"{video_id!r} is not in manifest.csv. Name the input file with its exact manifest video_id."
        )
    if output_path.stem != video_id:
        raise ValueError(f"Output filename must be {video_id}.json to match the manifest.")

    team_id = os.environ.get("HANDSCRIBE_TEAM_ID", "").strip()
    if not team_id:
        raise ValueError(
            "Set HANDSCRIBE_TEAM_ID to your organiser-assigned team ID before running inference."
        )
    if not CLASSES.is_file():
        raise FileNotFoundError(f"Official class file is missing: {CLASSES}")
    supported_classes = json.loads(CLASSES.read_text(encoding="utf-8"))
    manifest_duration = float(manifest[video_id]["duration_sec"])

    # The student kit requires >=4.0s highlight(s) and <=15% total duration.
    # These constraints cannot both be met when duration < 26.666... seconds.
    if manifest_duration * BUDGET_FRACTION < MIN_HIGHLIGHT_SEC:
        raise ValueError(
            f"{video_id} is {manifest_duration:.2f}s long: its 15% budget is under 4.0s, "
            "so the student-kit minimum-highlight and budget rules are mutually incompatible. "
            "Ask the organisers how these manifest entries should be handled."
        )

    options = Options(
        mode="semantic",
        weights=PACKAGE_ROOT / "weights",
        vocabulary=CLASSES,
        device=os.environ.get("HANDSCRIBE_DEVICE", "auto"),
        allow_short=False,
    )
    result = analyze(video_path, video_id, options)
    payload = build_submission_payload(
        result,
        team_id=team_id,
        video_id=video_id,
        dataset=manifest[video_id]["dataset"],
        manifest_duration=manifest_duration,
        supported_classes=supported_classes,
    )
    write_json(output_path, payload)
    print(
        f"Wrote {output_path}: {len(payload['actions'])} actions, "
        f"{len(payload['highlights'])} highlights, label={payload['label']['name']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one official-format Track 04 JSON from one manifest video."
    )
    parser.add_argument("video", type=Path, help="Input MP4; filename stem must equal manifest video_id")
    parser.add_argument("output", type=Path, help="Output JSON; filename must be <video_id>.json")
    args = parser.parse_args()
    try:
        run(args.video, args.output)
        return 0
    except (OSError, ValueError, VideoError, ModelSetupError) as exc:
        print(f"Track 04 submission error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
