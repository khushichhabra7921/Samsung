"""Build the TEAM_ID/ folder and TEAM_ID.zip for upload.

    python tools/build_package.py --team-id <ID> [--destination dist]

The repository root already follows the required layout. This tool runs
tools/validate.py --strict over outputs/ (all 187 manifest videos, consistent team_id),
checks the local model files, then copies the six required entries into TEAM_ID/. The
packaged run.sh defaults to the given team ID, so the organisers' re-run needs no
environment variables.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate import check_directory, failures, load_classes, load_manifest

REQUIRED_WEIGHTS = ("hand_landmarker.task", "owlv2/config.json", "siglip/config.json")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".gitkeep", ".cache", "*.lock")


def check_outputs(outputs: Path, team_id: str) -> None:
    results = check_directory(
        outputs, load_manifest(ROOT / "code" / "manifest.csv"),
        load_classes(ROOT / "code" / "supported_classes.json"), team_id,
    )
    problems = [
        f"{name or outputs.name}: {message}"
        for name, issues in results.items() for message in failures(issues, strict=True)
    ]
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(f"{len(problems)} validation problem(s):\n  {shown}{more}")


def build(team_id: str, outputs: Path, destination: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", team_id):
        raise ValueError("Provide the exact organiser-assigned TEAM_ID (letters, digits, '_', '-', '.').")
    check_outputs(outputs, team_id)
    missing = [name for name in REQUIRED_WEIGHTS if not (ROOT / "weights" / name).is_file()]
    if missing:
        raise ValueError("Required model files missing under weights/: " + ", ".join(missing))

    package = destination / team_id
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    shutil.copytree(ROOT / "code", package / "code", ignore=IGNORE)
    shutil.copytree(ROOT / "weights", package / "weights", ignore=IGNORE)
    (package / "outputs").mkdir()
    for path in sorted(outputs.glob("*.json")):
        shutil.copy2(path, package / "outputs" / path.name)
    for name in ("requirements.txt", "README.md"):
        shutil.copy2(ROOT / name, package / name)

    launcher = (ROOT / "run.sh").read_text(encoding="utf-8")
    if 'DEFAULT_TEAM_ID=""' not in launcher:
        raise ValueError('run.sh no longer contains the DEFAULT_TEAM_ID="" placeholder.')
    launcher = launcher.replace('DEFAULT_TEAM_ID=""', f"DEFAULT_TEAM_ID={shlex.quote(team_id)}", 1)
    (package / "run.sh").write_text(launcher, encoding="utf-8")
    os.chmod(package / "run.sh", 0o755)

    archive = destination / f"{team_id}.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(destination))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate outputs and build the TEAM_ID archive.")
    parser.add_argument("--team-id", default=os.environ.get("HANDSCRIBE_TEAM_ID", ""))
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--destination", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    try:
        archive = build(args.team_id.strip(), args.outputs.resolve(), args.destination.resolve())
    except (OSError, ValueError) as exc:
        print(f"Packaging failed: {exc}", file=sys.stderr)
        return 2
    print(f"Created {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
