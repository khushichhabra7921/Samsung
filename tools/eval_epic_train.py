"""Development check: score output JSONs against EPIC-KITCHENS-100 *train-split* labels.

    python tools/eval_epic_train.py --outputs runs/new [--annotations data/epic_annotations]

Approximates the Track 04 EPIC metrics: Action mAP (verb-noun classes, tIoU 0.1-0.5, pooled
over videos) and Action coverage (share of distinct ground-truth pairs with an instance at
least half inside the highlights). Videos outside the train split are refused so the scored
validation videos are never used for tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def seconds(stamp: str) -> float:
    hours, minutes, secs = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(secs)


def load_ground_truth(folder: Path) -> dict[str, list[tuple[float, float, tuple[str, str]]]]:
    def keys(name: str) -> list[str]:
        with (folder / name).open(encoding="utf-8", newline="") as stream:
            return [row["key"] for row in csv.DictReader(stream)]

    verbs, nouns = keys("EPIC_100_verb_classes.csv"), keys("EPIC_100_noun_classes.csv")
    truth: dict[str, list] = defaultdict(list)
    with (folder / "EPIC_100_train.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            truth[row["video_id"]].append((
                seconds(row["start_timestamp"]), seconds(row["stop_timestamp"]),
                (verbs[int(row["verb_class"])], nouns[int(row["noun_class"])]),
            ))
    return truth


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def average_precision(predictions, truth, threshold: float, key) -> float:
    """Mean over ground-truth classes of all-point interpolated AP."""
    by_class_truth: dict = defaultdict(list)
    for video, items in truth.items():
        for start, end, label in items:
            by_class_truth[key(label)].append((video, start, end))
    by_class_pred: dict = defaultdict(list)
    for video, items in predictions.items():
        for start, end, label, score in items:
            by_class_pred[key(label)].append((score, video, start, end))
    aps = []
    for cls, gts in by_class_truth.items():
        used = [False] * len(gts)
        hits = []
        for _, video, start, end in sorted(by_class_pred.get(cls, []), key=lambda item: -item[0]):
            best, best_index = threshold, -1
            for index, (gt_video, gt_start, gt_end) in enumerate(gts):
                if gt_video == video and not used[index]:
                    overlap = tiou((start, end), (gt_start, gt_end))
                    if overlap >= best:
                        best, best_index = overlap, index
            if best_index >= 0:
                used[best_index] = True
            hits.append(best_index >= 0)
        if not hits:
            aps.append(0.0)
            continue
        tp = np.cumsum(hits)
        precision = tp / np.arange(1, len(hits) + 1)
        recall = tp / len(gts)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        aps.append(float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision)))
    return float(np.mean(aps)) if aps else 0.0


def coverage(highlights, items) -> float:
    pairs = {label for _, _, label in items}
    captured = set()
    for start, end, label in items:
        inside = sum(max(0.0, min(end, h_end) - max(start, h_start)) for h_start, h_end in highlights)
        if end > start and inside >= 0.5 * (end - start):
            captured.add(label)
    return len(captured) / len(pairs) if pairs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Score outputs on EPIC-100 train-split videos.")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "epic_annotations")
    args = parser.parse_args()
    truth_all = load_ground_truth(args.annotations)
    predictions, truth, covered = {}, {}, []
    for path in sorted(args.outputs.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        video = data["video_id"]
        if video not in truth_all:
            print(f"skip {video}: not an EPIC-100 train-split video")
            continue
        truth[video] = truth_all[video]
        predictions[video] = [
            (*action["segment"], (action["verb"], action["noun"]), action["score"]) for action in data["actions"]
        ]
        covered.append(coverage([tuple(h["segment"]) for h in data["highlights"]], truth[video]))
        n_other = sum(label[0] == "other" for *_, label, _ in predictions[video])
        print(f"{video}: {len(predictions[video])} actions ({n_other} verb=other), "
              f"{len(truth[video])} ground-truth, coverage {covered[-1]:.3f}")
    if not truth:
        print("No train-split outputs found.")
        return 1
    for name, key in (("action", lambda label: label), ("verb", lambda label: label[0]),
                      ("noun", lambda label: label[1])):
        scores = [average_precision(predictions, truth, t, key) for t in THRESHOLDS]
        print(f"{name:>6} mAP@0.1-0.5: {np.mean(scores) * 100:.2f}  "
              + " ".join(f"{t}:{s * 100:.1f}" for t, s in zip(THRESHOLDS, scores, strict=True)))
    print(f"  mean coverage: {np.mean(covered):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
