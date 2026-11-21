"""Track 04 single-video entry point and official JSON exporter.

The exporter deliberately refuses to fabricate a valid-looking result when the
pipeline cannot satisfy the organiser's minimum-highlight rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from .contracts import ROOT, write_json
from .pipeline import Options, analyze
from .recognition import ModelSetupError
from .contracts import Report


KIT = ROOT / "track04_student_kit"
MANIFEST = KIT / "manifest.csv"
VOCABULARY = KIT / "supported_classes.json"


def load_manifest(path: Path = MANIFEST) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Track 04 manifest: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"video_id", "duration_sec", "dataset"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifest.csv must contain video_id,duration_sec,dataset")
    result = {}
    for row in rows:
        video_id = row["video_id"].strip()
        if not video_id or video_id in result:
            raise ValueError(f"Invalid or duplicate manifest video_id: {video_id!r}")
        result[video_id] = row
    return result


def official_payload(report: Report, *, team_id: str, dataset: str) -> dict[str, Any]:
    if not team_id.strip():
        raise ValueError("Set TEAM_ID to the exact organiser-assigned team ID before running.")
    if dataset not in {"epic_kitchens_100", "fpvsum"}:
        raise ValueError(f"Unsupported manifest dataset: {dataset}")

    allowed_verbs: set[str] | None = None
    allowed_nouns: set[str] | None = None
    if VOCABULARY.is_file():
        vocab = json.loads(VOCABULARY.read_text(encoding="utf-8"))
        allowed_verbs = set(vocab.get("verbs", []))
        allowed_nouns = set(vocab.get("nouns", []))

    actions = []
    for action in report.actions:
        verb, noun = action.label.verb, action.label.noun
        if allowed_verbs is not None and verb not in allowed_verbs:
            continue
        if allowed_nouns is not None and noun not in allowed_nouns:
            continue
        actions.append({
            "segment": [round(action.start_ms / 1000, 2), round(action.end_ms / 1000, 2)],
            "verb": verb,
            "noun": noun,
            "score": round(action.confidence, 4),
        })

    highlights = [
        {"segment": [round(item.start_ms / 1000, 2), round(item.end_ms / 1000, 2)]}
        for item in report.highlights
    ]
    if not any(end - start >= 4.0 for item in highlights for start, end in [item["segment"]]):
        raise ValueError(
            f"{report.video_id}: no selected highlight is at least 4.0 seconds. "
            "The organiser's stated minimum-highlight rule cannot be satisfied for this output."
        )

    if dataset == "epic_kitchens_100":
        label = {"name": "cooking", "score": 1.0}
    else:
        # The current model has no FPVSum category classifier. 'other' is a
        # schema-valid conservative fallback, not a meaningful category prediction.
        label = {"name": "other", "score": 0.0}

    return {
        "team_id": team_id,
        "video_id": report.video_id,
        "duration_sec": round(report.duration_sec, 2),
        "label": label,
        "actions": actions[:2000],
        "highlights": highlights,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track 04 inference: ./run.sh <video.mp4> <out.json>"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        team_id = os.environ.get("TEAM_ID", "").strip()
        if not team_id:
            raise ValueError("TEAM_ID is unset. Set it to the exact team ID assigned by the organisers.")
        video_id = args.video.stem
        manifest = load_manifest()
        if video_id not in manifest:
            raise ValueError(f"{video_id!r} is not listed in the official Track 04 manifest.")
        if args.output.name != f"{video_id}.json":
            raise ValueError(f"Output filename must be {video_id}.json to match the input video ID.")

        row = manifest[video_id]
        options = Options(
            mode="semantic",
            weights=ROOT / "weights",
            vocabulary=VOCABULARY,
            device=os.environ.get("XR_DEVICE", "auto"),
        )
        analysis = analyze(args.video, video_id, options)
        expected_duration = float(row["duration_sec"])
        if abs(analysis.report.duration_sec - expected_duration) > 0.5:
            raise ValueError(
                f"Decoded duration {analysis.report.duration_sec:.2f}s differs from manifest "
                f"{expected_duration:.2f}s by more than 0.5s."
            )
        payload = official_payload(analysis.report, team_id=team_id, dataset=row["dataset"])
        write_json(args.output, payload)
        print(
            f"Wrote {args.output} | {len(payload['actions'])} actions | "
            f"{len(payload['highlights'])} highlights"
        )
        if row["dataset"] == "fpvsum":
            print("WARNING: FPVSum label uses the schema-valid 'other' fallback; category model is not implemented.", file=sys.stderr)
        return 0
    except (OSError, ValueError, ModelSetupError) as exc:
        print(f"Track 04 submission error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
