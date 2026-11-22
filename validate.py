#!/usr/bin/env python3
"""Track 04 output validator (team-written stand-in for the organiser's validate.py).

The Student Package lists validate.py but the copy we received did not include it.
This script implements every check that TEAM_README.md sections 3-4 and the
Track 04 submission slides describe, using only the Python standard library:

* all six required top-level keys, their types, and finite numeric ranges;
* 0 <= start < end <= duration_sec + 0.5 for every action and highlight;
* highlights chronologically sorted, non-overlapping, and at least one >= 4.0 s;
* warnings for highlights < 4.0 s (ignored in scoring) and for the 15% budget;
* duration within +/-0.5 s of manifest.csv, and one file per manifest video (--dir);
* label/verb/noun membership in supported_classes.json.

Default severities follow the README. --strict also fails avoidable budget
overruns, vocabulary mismatches, and unknown keys; the packager uses --strict.
Run the organiser's script as well once it is supplied.

    python validate.py --dir outputs/ [--strict]
    python validate.py --file examples/P02_109.json -v
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED_KEYS = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")
MAX_ACTIONS = 2000
MIN_HIGHLIGHT_SEC = 4.0
BUDGET_FRACTION = 0.15
DURATION_TOLERANCE_SEC = 0.5

# "error" always fails; "strict" fails only with --strict; "warning" never fails.
Issue = tuple[str, str]


def load_manifest(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["video_id"].strip(): float(row["duration_sec"]) for row in csv.DictReader(stream)}


def load_classes(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: set(data[key]) for key in ("labels", "verbs", "nouns")}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _segment(item, where: str, duration, issues: list[Issue]) -> tuple[float, float] | None:
    segment = item.get("segment") if isinstance(item, dict) else None
    if not isinstance(segment, list) or len(segment) != 2 or not all(_number(x) for x in segment):
        issues.append(("error", f"{where}: 'segment' must be a list of two finite numbers"))
        return None
    start, end = segment
    upper = duration + DURATION_TOLERANCE_SEC if _number(duration) else math.inf
    if not 0 <= start < end <= upper:
        issues.append(("error", f"{where}: segment {segment} violates 0 <= start < end <= duration_sec + 0.5"))
        return None
    if any(abs(x * 100 - round(x * 100)) > 1e-6 for x in segment):
        issues.append(("warning", f"{where}: timestamps should use at most 2 decimal places"))
    return float(start), float(end)


def _score(container: dict, where: str, issues: list[Issue]) -> None:
    value = container.get("score")
    if not _number(value) or not 0 <= value <= 1:
        issues.append(("error", f"{where}: 'score' must be a finite number in [0, 1]"))


def check_payload(
    data, *, stem: str | None = None,
    manifest: dict[str, float] | None = None, classes: dict[str, set[str]] | None = None,
) -> list[Issue]:
    """Return every issue found in one decoded output JSON."""
    if not isinstance(data, dict):
        return [("error", "top level must be a JSON object")]
    issues: list[Issue] = [("error", f"missing required key '{key}'") for key in REQUIRED_KEYS if key not in data]
    issues += [("strict", f"unexpected top-level key '{key}'") for key in data if key not in REQUIRED_KEYS]

    if "team_id" in data and not _text(data["team_id"]):
        issues.append(("error", "'team_id' must be a non-empty string"))
    video_id = data.get("video_id")
    if "video_id" in data and not _text(video_id):
        issues.append(("error", "'video_id' must be a non-empty string"))
    elif stem is not None and video_id is not None and video_id != stem:
        issues.append(("error", f"'video_id' {video_id!r} does not match the filename {stem}.json"))

    duration = data.get("duration_sec")
    if "duration_sec" in data and (not _number(duration) or duration <= 0):
        issues.append(("error", "'duration_sec' must be a finite number > 0"))
        duration = None
    if manifest is not None and _text(video_id):
        if video_id not in manifest:
            issues.append(("strict", f"video_id {video_id!r} is not listed in manifest.csv"))
        elif duration is not None and abs(duration - manifest[video_id]) > DURATION_TOLERANCE_SEC:
            issues.append(("error", f"duration_sec {duration} differs from manifest {manifest[video_id]} by > 0.5 s"))

    label = data.get("label")
    if "label" in data:
        if not isinstance(label, dict):
            issues.append(("error", "'label' must be an object {'name': str, 'score': number}"))
        else:
            if not _text(label.get("name")):
                issues.append(("error", "label: 'name' must be a non-empty string"))
            elif classes is not None and label["name"] not in classes["labels"]:
                issues.append(("strict", f"label {label['name']!r} is not in supported_classes.json"))
            _score(label, "label", issues)

    actions = data.get("actions")
    if "actions" in data:
        if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
            issues.append(("error", f"'actions' must be an array of 0-{MAX_ACTIONS} items"))
        else:
            for index, action in enumerate(actions):
                where = f"actions[{index}]"
                if not isinstance(action, dict):
                    issues.append(("error", f"{where}: must be an object"))
                    continue
                _segment(action, where, duration, issues)
                for key in ("verb", "noun"):
                    if not _text(action.get(key)):
                        issues.append(("error", f"{where}: '{key}' must be a non-empty string"))
                    elif classes is not None and action[key] not in classes[key + "s"]:
                        issues.append(("strict", f"{where}: {key} {action[key]!r} is not in supported_classes.json"))
                _score(action, where, issues)

    highlights = data.get("highlights")
    if "highlights" in data:
        if not isinstance(highlights, list) or not highlights:
            issues.append(("error", "'highlights' must be a non-empty array"))
        else:
            segments = [_segment(item, f"highlights[{i}]", duration, issues) for i, item in enumerate(highlights)]
            if all(segments):
                issues += _check_highlights(segments, duration)
    return issues


def _check_highlights(segments: list[tuple[float, float]], duration) -> list[Issue]:
    issues: list[Issue] = []
    for index in range(1, len(segments)):
        (prev_start, prev_end), (start, _) = segments[index - 1], segments[index]
        if start < prev_start:
            issues.append(("error", f"highlights[{index}] is not in chronological order"))
        elif start < prev_end:
            issues.append(("error", f"highlights[{index - 1}] and highlights[{index}] overlap"))
        elif start == prev_end:
            issues.append(("strict", f"highlights[{index - 1}] and highlights[{index}] touch; merge them"))
    lengths = [end - start for start, end in segments]
    if not any(length >= MIN_HIGHLIGHT_SEC for length in lengths):
        issues.append(("error", "no highlight segment is at least 4.0 s"))
    issues += [
        ("warning", f"highlights[{i}] is {length:.2f} s (< 4.0 s, ignored during scoring)")
        for i, length in enumerate(lengths) if length < MIN_HIGHLIGHT_SEC
    ]
    if _number(duration):
        total, budget = sum(lengths), duration * BUDGET_FRACTION
        if total > budget + 1e-6:
            message = f"highlights total {total:.2f} s exceeds the 15% budget of {budget:.2f} s"
            if budget >= MIN_HIGHLIGHT_SEC:
                issues.append(("strict", message))
            else:
                issues.append(("warning", message + " (unavoidable: the budget is shorter than 4.0 s)"))
    return issues


def check_file(
    path: Path, manifest: dict[str, float] | None = None, classes: dict[str, set[str]] | None = None,
) -> tuple[dict | None, list[Issue]]:
    def reject(constant: str):
        raise ValueError(f"non-standard JSON constant {constant}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, [("error", f"unreadable JSON: {exc}")]
    return data, check_payload(data, stem=path.stem, manifest=manifest, classes=classes)


def check_directory(
    folder: Path, manifest: dict[str, float] | None = None,
    classes: dict[str, set[str]] | None = None, team_id: str | None = None,
) -> dict[str, list[Issue]]:
    """Map each file name (and '' for directory-level problems) to its issues."""
    results: dict[str, list[Issue]] = {"": []}
    team_ids: dict[str, list[str]] = {}
    for path in sorted(folder.glob("*.json")):
        data, issues = check_file(path, manifest, classes)
        results[path.name] = issues
        if isinstance(data, dict) and isinstance(data.get("team_id"), str):
            team_ids.setdefault(data["team_id"], []).append(path.name)
    if team_id is not None:
        for other, names in team_ids.items():
            if other != team_id:
                results[""].append(("error", f"{len(names)} file(s) use team_id {other!r}, expected {team_id!r}"))
    elif len(team_ids) > 1:
        results[""].append(("error", f"team_id differs across files: {sorted(team_ids)}"))
    if not results.keys() - {""}:
        results[""].append(("error", f"no .json files found in {folder}"))
    if manifest is not None:
        missing = sorted(set(manifest) - {Path(name).stem for name in results})
        if missing:
            shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            results[""].append(("error", f"{len(missing)} manifest video(s) have no output JSON: {shown}"))
    return results


def failures(issues: list[Issue], strict: bool) -> list[str]:
    return [message for level, message in issues if level == "error" or (strict and level == "strict")]


def _default(name: str, given: str | None) -> Path | None:
    if given is not None:
        return Path(given) if given else None
    for candidate in (HERE / name, Path.cwd() / name):
        if candidate.is_file():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Track 04 output JSON files.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dir", type=Path, help="Folder with one <video_id>.json per video")
    target.add_argument("--file", type=Path, help="A single <video_id>.json file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print warnings as well as errors")
    parser.add_argument("--strict", action="store_true",
                        help="Also fail on avoidable budget overruns, unknown classes, and unknown keys")
    parser.add_argument("--manifest", help="manifest.csv path ('' to skip; default: next to this script)")
    parser.add_argument("--classes", help="supported_classes.json path ('' to skip; default: next to this script)")
    parser.add_argument("--team-id", help="Require this exact team_id in every file")
    args = parser.parse_args(argv)

    manifest_path, classes_path = _default("manifest.csv", args.manifest), _default(
        "supported_classes.json", args.classes)
    manifest = load_manifest(manifest_path) if manifest_path else None
    classes = load_classes(classes_path) if classes_path else None
    if args.file is not None:
        data, issues = check_file(args.file, manifest, classes)
        if args.team_id is not None and isinstance(data, dict) and data.get("team_id") != args.team_id:
            issues.append(("error", f"team_id is not {args.team_id!r}"))
        results = {args.file.name: issues}
    else:
        results = check_directory(args.dir, manifest, classes, args.team_id)

    failed = 0
    for name, issues in results.items():
        errors = failures(issues, args.strict)
        warnings = [message for level, message in issues if message not in errors]
        if not name:
            for message in errors:
                print(f"ERROR {message}")
            continue
        failed += bool(errors)
        suffix = f" ({len(warnings)} warning{'s' * (len(warnings) != 1)})" if warnings else ""
        print(f"{'FAIL' if errors else 'PASS'}  {name}{suffix}")
        for message in errors:
            print(f"      error: {message}")
        for message in warnings if args.verbose else ():
            print(f"      warning: {message}")

    checked = len(results) - ("" in results)
    directory_errors = failures(results.get("", []), args.strict)
    print(f"\n{checked} file(s) checked, {checked - failed} passed, {failed} failed"
          + (f", {len(directory_errors)} directory-level error(s)" if directory_errors else "")
          + ("" if manifest else " [manifest checks skipped]")
          + ("" if classes else " [class checks skipped]"))
    return 1 if failed or directory_errors else 0


if __name__ == "__main__":
    sys.exit(main())
