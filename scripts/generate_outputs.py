"""Generate outputs/<video_id>.json for every manifest video found under a folder.

    export HANDSCRIBE_TEAM_ID=<organiser-assigned-id>
    python scripts/generate_outputs.py --videos /data/track04 [--outputs outputs] [--force]

Videos are matched by file name anywhere under --videos, in any letter case, so the
EPIC downloader layout (EPIC-KITCHENS/P01/videos/P01_11.MP4) and a flat FPVSum folder
both work. Each video runs through the same exporter as ./run.sh with the models loaded
once. Outputs that already pass validate.py --strict are skipped, so an interrupted run
can resume. The script finishes with a strict check of the whole outputs folder.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "BiosHackathon" / "src")]

from egovision.ingest import VideoError

from handscribe_xr.recognition import ModelSetupError
from handscribe_xr.track04 import run
from validate import check_directory, check_file, failures, load_classes, load_manifest


def find_videos(folder: Path, wanted: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() == ".mp4" and path.stem in wanted and path.is_file():
            if path.stem in found:
                print(f"note: using {found[path.stem]}, ignoring duplicate {path}", file=sys.stderr)
                continue
            found[path.stem] = path
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Track 04 exporter for all manifest videos.")
    parser.add_argument("--videos", type=Path, required=True, help="Folder containing the downloaded MP4s")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--team-id", default=os.environ.get("HANDSCRIBE_TEAM_ID", ""))
    parser.add_argument("--force", action="store_true", help="Regenerate outputs that already validate")
    args = parser.parse_args()
    if not args.team_id.strip():
        parser.error("set HANDSCRIBE_TEAM_ID or pass --team-id")
    os.environ["HANDSCRIBE_TEAM_ID"] = args.team_id.strip()

    manifest = load_manifest(ROOT / "manifest.csv")
    classes = load_classes(ROOT / "supported_classes.json")
    videos = find_videos(args.videos, set(manifest))
    missing = sorted(set(manifest) - set(videos))
    failed: dict[str, str] = {}
    skipped = 0
    for number, video_id in enumerate(sorted(videos), 1):
        output = args.outputs / f"{video_id}.json"
        if not args.force and output.is_file() and not failures(check_file(output, manifest, classes)[1], True):
            skipped += 1
            continue
        print(f"[{number}/{len(videos)}] {video_id}", flush=True)
        started = time.monotonic()
        try:
            run(videos[video_id], output)
        except ModelSetupError as exc:
            print(f"Model setup failed; stopping: {exc}", file=sys.stderr)
            return 2
        except (OSError, ValueError, VideoError) as exc:
            failed[video_id] = str(exc)
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        print(f"  done in {time.monotonic() - started:.0f} s", flush=True)

    print(f"\n{len(videos)} videos found, {skipped} already valid, {len(failed)} failed, "
          f"{len(missing)} manifest videos not found under {args.videos}")
    for video_id, reason in failed.items():
        print(f"  failed  {video_id}: {reason}")
    for video_id in missing:
        print(f"  missing {video_id} ({manifest[video_id]:.2f} s)")

    results = check_directory(args.outputs, manifest, classes, args.team_id.strip())
    problems = sum(bool(failures(issues, strict=True)) for issues in results.values())
    print(f"validate.py --strict: {'all outputs valid' if not problems else f'{problems} problem group(s)'}"
          f" — run `python validate.py --dir {args.outputs} --strict -v` for details")
    return 0 if not (failed or missing or problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
