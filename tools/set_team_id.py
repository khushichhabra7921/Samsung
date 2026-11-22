"""Switch every output JSON and run.sh's default to the confirmed team ID (no re-processing).

    python tools/set_team_id.py <ID> [--outputs outputs]

team_id is metadata, not a model output, so re-running ./run.sh with the new ID produces the
same files. The script refuses IDs with characters other than letters, digits, '_', '-', '.'.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Set team_id in all outputs and run.sh.")
    parser.add_argument("team_id")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.team_id):
        parser.error("team ID may contain only letters, digits, '_', '-', '.'")

    changed = 0
    for path in sorted(args.outputs.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("team_id") != args.team_id:
            data["team_id"] = args.team_id
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            changed += 1
    launcher = ROOT / "run.sh"
    text = launcher.read_text(encoding="utf-8")
    line = f"DEFAULT_TEAM_ID={shlex.quote(args.team_id)}"
    updated = re.sub(r"^DEFAULT_TEAM_ID=.*$", lambda _: line, text, count=1, flags=re.MULTILINE)
    launcher.write_text(updated, encoding="utf-8")
    print(f"team_id -> {args.team_id}: {changed} output file(s) updated; run.sh default "
          f"{'updated' if updated != text else 'already set'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
