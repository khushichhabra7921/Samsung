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
MANIFEST = PACKAGE_ROOT / "manifest.csv"
CLASSES = PACKAGE_ROOT / "supported_classes.json"
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


def _span(start_cs: int, end_cs: int) -> float:
    """Segment length exactly as a validator computes it from the emitted floats."""
    return end_cs / 100 - start_cs / 100


def finalize_highlights(clips, duration_cs: int) -> list[list[float]]:
    """Convert selected clips to 2-decimal segments that pass the student-kit rules.

    Rounding to centiseconds can turn a 4000 ms clip into 3.9999999999999996 s, which a
    validator treats as under the 4.0 s minimum, and can push the total a hair past the
    15% budget. Touching clips are merged so "strictly non-overlapping" holds under any
    reading. Videos shorter than 26.67 s cannot satisfy both the >=4.0 s and <=15% rules;
    the 4.0 s minimum is the hard (file-invalidating) one, while exceeding the budget is
    a warning with a soft scoring penalty, so those videos get one 4.0 s highlight
    centred on the best-scoring clip.
    """
    minimum = round(MIN_HIGHLIGHT_SEC * 100)
    budget_cs = duration_cs * 15 // 100
    segments = sorted(
        [(int(clip.start_ms) + 5) // 10, min((int(clip.end_ms) + 5) // 10, duration_cs),
         float(getattr(clip, "score", 0.0))]
        for clip in clips
    )
    segments = [segment for segment in segments if 0 <= segment[0] < segment[1]]
    if budget_cs < minimum:
        if duration_cs <= minimum:
            raise ValueError("The video is too short for a 4.0 s highlight.")
        best = max(segments, key=lambda segment: segment[2], default=None)
        centre = (best[0] + best[1]) // 2 if best else duration_cs // 2
        start = max(0, min(centre - minimum // 2, duration_cs - minimum))
        segments = [[start, start + minimum, 0.0]]

    merged: list[list] = []
    for segment in segments:
        if merged and segment[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], segment[1])
            merged[-1][2] = max(merged[-1][2], segment[2])
        else:
            merged.append(segment)

    for index, segment in enumerate(merged):
        if segment[1] - segment[0] >= minimum and _span(segment[0], segment[1]) < MIN_HIGHLIGHT_SEC:
            following = merged[index + 1][0] if index + 1 < len(merged) else duration_cs + 1
            preceding = merged[index - 1][1] if index else -1
            if segment[1] + 1 < following:
                segment[1] += 1
            elif segment[0] - 1 > preceding:
                segment[0] -= 1

    if budget_cs >= minimum:
        budget = duration_cs / 100 * BUDGET_FRACTION
        while (sum(end - start for start, end, _ in merged) > budget_cs
               or sum(_span(start, end) for start, end, _ in merged) > budget):
            trimmable = [s for s in merged if _span(s[0], s[1] - 1) >= MIN_HIGHLIGHT_SEC]
            if trimmable:
                max(trimmable, key=lambda segment: segment[1] - segment[0])[1] -= 1
            elif len(merged) > 1:
                merged.remove(min(merged, key=lambda segment: segment[2]))
            else:
                raise ValueError("Rounded highlight segments exceed the 15% duration budget.")

    if not merged:
        raise ValueError("Inference produced no highlight segments.")
    if not any(_span(start, end) >= MIN_HIGHLIGHT_SEC for start, end, _ in merged):
        raise ValueError("No highlight is at least 4.0 seconds; output cannot satisfy the student-kit rules.")
    return [[start / 100, end / 100] for start, end, _ in merged]


def _video_label(analysis, dataset: str | None, labels: set[str]) -> dict:
    if dataset == "epic_kitchens_100":
        return {"name": "cooking", "score": 1.0}
    if dataset == "fpvsum":
        # FPVSum ground truth is always one of its 14 activity categories.
        allowed = labels - {"cooking", "other"}
    elif dataset is None:
        allowed = labels
    else:
        raise ValueError(f"Unknown manifest dataset: {dataset!r}")
    ranked = [
        (name, float(score))
        for name, score in getattr(analysis, "video_categories", ())
        if name in allowed and math.isfinite(float(score))
    ]
    if not ranked:
        # A wrong label only costs label accuracy; a missing file scores 0 on every metric.
        print("Track 04 warning: no video-category scores; writing label 'other'.", file=sys.stderr)
        return {"name": "other", "score": 0.0}
    name, score = max(ranked, key=lambda item: item[1])
    return {"name": name, "score": max(0.0, min(1.0, score))}


def resolve_video_id(video_path: Path, output_path: Path, manifest: dict) -> str:
    """Prefer the input stem, then the output stem, so organiser re-runs need no renaming."""
    for candidate in (video_path.stem, output_path.stem):
        if candidate in manifest:
            return candidate
    return video_path.stem


def build_submission_payload(
    analysis,
    *,
    team_id: str,
    video_id: str,
    dataset: str | None,
    manifest_duration: float | None,
    supported_classes: dict,
) -> dict:
    report = analysis.report
    duration = float(report.duration_sec)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Inference returned an invalid video duration.")
    if manifest_duration is not None and abs(duration - manifest_duration) > 0.5:
        raise ValueError(
            f"Decoded duration {duration:.3f}s differs from manifest "
            f"duration {manifest_duration:.3f}s by more than 0.5s."
        )

    labels = set(supported_classes.get("labels", []))
    verbs = set(supported_classes.get("verbs", []))
    nouns = set(supported_classes.get("nouns", []))
    if not labels or not verbs or not nouns:
        raise ValueError("supported_classes.json is missing official labels, verbs or nouns.")

    video_label = _video_label(analysis, dataset, labels)

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

    duration_cs = round(duration * 100)
    highlights = finalize_highlights(report.highlights, duration_cs)

    return {
        "team_id": team_id,
        "video_id": video_id,
        "duration_sec": duration_cs / 100,
        "label": video_label,
        "actions": actions,
        "highlights": [{"segment": segment} for segment in highlights],
    }


def run(video_path: Path, output_path: Path) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    manifest = load_manifest()
    video_id = resolve_video_id(video_path, output_path, manifest)
    row = manifest.get(video_id)
    if row is None:
        print(f"Track 04 note: {video_id!r} is not in manifest.csv; using the decoded duration "
              "and the predicted video category.", file=sys.stderr)
    if output_path.stem != video_id:
        print(f"Track 04 note: the submission expects this file as {video_id}.json.", file=sys.stderr)

    team_id = os.environ.get("HANDSCRIBE_TEAM_ID", "").strip()
    if not team_id:
        raise ValueError(
            "Set HANDSCRIBE_TEAM_ID to your organiser-assigned team ID before running inference."
        )
    if not CLASSES.is_file():
        raise FileNotFoundError(f"Official class file is missing: {CLASSES}")
    supported_classes = json.loads(CLASSES.read_text(encoding="utf-8"))

    # allow_short keeps a best clip for videos under 26.67 s, whose 15% budget is below
    # 4.0 s; finalize_highlights widens it to the 4.0 s minimum the validator requires.
    options = Options(
        mode="semantic",
        weights=PACKAGE_ROOT / "weights",
        vocabulary=CLASSES,
        device=os.environ.get("HANDSCRIBE_DEVICE", "auto"),
        allow_short=True,
    )
    result = analyze(video_path, video_id, options)
    payload = build_submission_payload(
        result,
        team_id=team_id,
        video_id=video_id,
        dataset=row["dataset"] if row else None,
        manifest_duration=float(row["duration_sec"]) if row else None,
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
