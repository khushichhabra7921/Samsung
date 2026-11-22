"""Build the verb-given-noun prior from the public EPIC-KITCHENS-100 *train* split.

    python tools/build_action_prior.py --annotations data/epic_annotations

Expects EPIC_100_train.csv, EPIC_100_verb_classes.csv and EPIC_100_noun_classes.csv from
https://github.com/epic-kitchens/epic-kitchens-100-annotations. Only aggregate verb-noun
co-occurrence counts are written; no timestamps or per-video information. The Track 04
EPIC videos are all in the validation split; the script refuses to run if any manifest
video appears in the input.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUTPUT = CODE / "handscribe_xr" / "assets" / "epic100_train_prior.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the EPIC-100 train-split verb-noun prior.")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data" / "epic_annotations")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    def rows(name: str) -> list[dict[str, str]]:
        with (args.annotations / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    verbs = [row["key"] for row in rows("EPIC_100_verb_classes.csv")]
    nouns = [row["key"] for row in rows("EPIC_100_noun_classes.csv")]
    classes = json.loads((CODE / "supported_classes.json").read_text(encoding="utf-8"))
    if verbs != [v for v in classes["verbs"] if v != "other"] or nouns != [n for n in classes["nouns"] if n != "other"]:
        raise SystemExit("EPIC class keys do not match supported_classes.json.")

    train = rows("EPIC_100_train.csv")
    with (CODE / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        evaluated = {row["video_id"] for row in csv.DictReader(stream)}
    leaked = evaluated & {row["video_id"] for row in train}
    if leaked:
        raise SystemExit(f"Refusing to build: manifest videos found in the input: {sorted(leaked)[:5]}")

    counts = Counter((int(row["verb_class"]), int(row["noun_class"])) for row in train)
    payload = {
        "source": "EPIC-KITCHENS-100 annotations, EPIC_100_train.csv (train split only)",
        "actions": len(train),
        "verbs": verbs,
        "nouns": nouns,
        "pair_counts": sorted([verb, noun, count] for (verb, noun), count in counts.items()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} from {len(train)} train actions, {len(counts)} verb-noun pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
