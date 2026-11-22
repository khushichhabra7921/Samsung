"""Generate outputs/<video_id>.json for every manifest video.

    export HANDSCRIBE_TEAM_ID=<organiser-assigned-id>
    python tools/generate_outputs.py --videos /data/track04            # videos already downloaded
    python tools/generate_outputs.py --fetch /data/tmp-videos          # download -> process -> delete

--videos matches <video_id>.mp4 anywhere under the folder, in any letter case (the EPIC
downloader layout and a flat FPVSum folder both work). --fetch downloads each missing video
from its official source (see download_videos.py), processes it, then deletes it, so the
whole manifest fits on a small disk; the next video downloads while the current one is
processed whenever there is room. Videos are processed shortest first. Each video runs
through the same exporter as ./run.sh with the models loaded once. Outputs that already pass
tools/validate.py --strict are skipped, so an interrupted run can resume. The script ends
with a strict check of the whole outputs folder.
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "tools")]

import download_videos as dl
from egovision.ingest import VideoError
from handscribe_xr.recognition import ModelSetupError
from handscribe_xr.track04 import run
from validate import check_directory, check_file, failures, load_classes, load_manifest

DISK_MARGIN = 1.5e9  # bytes kept free while fetching
FPVSUM_SIZE_GUESS = 0.15e9


def find_videos(folder: Path, wanted: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() == ".mp4" and path.stem in wanted and path.is_file():
            if path.stem in found:
                print(f"note: using {found[path.stem]}, ignoring duplicate {path}", file=sys.stderr)
                continue
            found[path.stem] = path
    return found


def assign_part(order: list[str], part: str) -> list[str]:
    """Deal videos (already sorted by length) round-robin into N parts and return part K."""
    if not part:
        return order
    try:
        k, n = (int(x) for x in part.split("/"))
    except ValueError:
        raise SystemExit(f"--part must look like 1/2, not {part!r}") from None
    if not 1 <= k <= n:
        raise SystemExit(f"--part {part}: K must be between 1 and N")
    return order[k - 1::n]


class Fetcher(threading.Thread):
    """Downloads queued videos ahead of processing while the disk has room."""

    def __init__(self, video_ids: list[str], datasets: dict[str, str], folder: Path, connections: int):
        super().__init__(daemon=True)
        self.ids, self.datasets, self.folder, self.connections = video_ids, datasets, folder, connections
        folder.mkdir(parents=True, exist_ok=True)
        self.ready: queue.Queue = queue.Queue(maxsize=2)
        self.on_disk = 0  # bytes downloaded and not yet deleted
        self.lock = threading.Lock()
        epic = [v for v in video_ids if datasets[v] == "epic_kitchens_100"]
        self.epic = dl.epic_sources(epic) if epic else {}
        fpv = [v for v in video_ids if datasets[v] == "fpvsum"]
        self.youtube = {}
        if fpv:
            self.youtube = dl.fpvsum_youtube_ids()
        self.ytdlp = dl.youtube_downloader() if fpv else None

    def release(self, nbytes: int) -> None:
        with self.lock:
            self.on_disk -= nbytes

    def run(self) -> None:
        for video_id in self.ids:
            self.ready.put(self._fetch(video_id))
        self.ready.put(None)

    def _fetch(self, video_id: str) -> tuple[str, Path | None, int, str]:
        epic = self.datasets[video_id] == "epic_kitchens_100"
        try:
            if epic:
                if video_id not in self.epic:
                    return video_id, None, 0, "no official EPIC URL"
                url, md5 = self.epic[video_id]
                size = dl.remote_size(url)
            else:
                if video_id not in self.youtube or self.ytdlp is None:
                    return video_id, None, 0, "needs yt-dlp and an FPVSum_videolist.csv entry"
                size = FPVSUM_SIZE_GUESS
            while shutil.disk_usage(self.folder).free - size < DISK_MARGIN:
                with self.lock:
                    idle = self.on_disk == 0
                if idle:
                    need = (size + DISK_MARGIN) / 1e9
                    return video_id, None, 0, f"needs {need:.1f} GB free disk (EPIC file {size / 1e9:.1f} GB)"
                time.sleep(5)
            with self.lock:
                self.on_disk += size
            sub = self.folder / ("epic" if epic else "fpvsum")
            sub.mkdir(parents=True, exist_ok=True)
            ok = (dl.download_epic(video_id, url, md5, sub, self.connections) if epic
                  else dl.download_fpvsum(video_id, self.youtube[video_id], sub, self.ytdlp, 720))
            path = sub / f"{video_id}.{'MP4' if epic else 'mp4'}"
            if not ok or not path.is_file():
                self.release(size)
                return video_id, None, 0, "download failed (re-run to resume)"
            actual = path.stat().st_size
            self.release(size - actual)
            return video_id, path, actual, ""
        except OSError as exc:
            return video_id, None, 0, f"download failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Track 04 exporter for all manifest videos.")
    parser.add_argument("--videos", type=Path, help="Folder containing already-downloaded MP4s")
    parser.add_argument("--fetch", type=Path, help="Download missing videos here, process, then delete them")
    parser.add_argument("--keep", action="store_true", help="With --fetch, keep videos after processing")
    parser.add_argument("--connections", type=int, default=dl.CONNECTIONS, help="Parallel connections per EPIC file")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--team-id", default=os.environ.get("HANDSCRIBE_TEAM_ID", ""))
    parser.add_argument("--force", action="store_true", help="Regenerate outputs that already validate")
    parser.add_argument("--dataset", choices=("all", "epic", "fpvsum"), default="all",
                        help="Only process one dataset (e.g. split the work across two machines)")
    parser.add_argument("--part", default="", metavar="K/N",
                        help="Only process part K of N (e.g. 1/2 on one machine, 2/2 on another). Parts are fixed "
                             "and balanced by video length, so machines never repeat each other's videos.")
    args = parser.parse_args()
    if not args.team_id.strip():
        parser.error("set HANDSCRIBE_TEAM_ID or pass --team-id")
    if args.videos is None and args.fetch is None:
        parser.error("pass --videos, --fetch, or both")
    os.environ["HANDSCRIBE_TEAM_ID"] = args.team_id.strip()

    manifest = load_manifest(ROOT / "code" / "manifest.csv")
    classes = load_classes(ROOT / "code" / "supported_classes.json")
    with (ROOT / "code" / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        datasets = {row["video_id"]: row["dataset"] for row in csv.DictReader(stream)}

    def valid(video_id: str) -> bool:
        output = args.outputs / f"{video_id}.json"
        return output.is_file() and not failures(check_file(output, manifest, classes)[1], True)

    local = find_videos(args.videos, set(manifest)) if args.videos else {}
    wanted = {"all": set(datasets.values()), "epic": {"epic_kitchens_100"}, "fpvsum": {"fpvsum"}}[args.dataset]
    order = sorted((v for v in manifest if datasets[v] in wanted), key=lambda v: (datasets[v] != "fpvsum", manifest[v]))
    mine = set(assign_part(order, args.part))
    todo = [v for v in order if v in mine and (args.force or not valid(v))]
    skipped = len(mine) - len(todo)
    to_fetch = [v for v in todo if v not in local] if args.fetch else []
    missing = sorted(v for v in todo if v not in local and v not in to_fetch)
    print(f"{len(mine)} {args.dataset} manifest videos{f' (part {args.part})' if args.part else ''}: "
          f"{skipped} already valid, {len(todo)} to do "
          f"({len(todo) - len(to_fetch) - len(missing)} local, {len(to_fetch)} to fetch, {len(missing)} not found)")

    fetcher = Fetcher(to_fetch, datasets, args.fetch, args.connections) if to_fetch else None
    if fetcher:
        fetcher.start()
    jobs = [(v, local[v], 0, "") for v in todo if v in local]
    failed: dict[str, str] = {}
    done = 0
    total = len(todo) - len(missing)
    while True:
        if jobs:
            video_id, path, nbytes, error = jobs.pop(0)
        elif fetcher:
            item = fetcher.ready.get()
            if item is None:
                break
            video_id, path, nbytes, error = item
        else:
            break
        done += 1
        if path is None:
            failed[video_id] = error
            print(f"[{done}/{total}] {video_id}  SKIPPED: {error}", file=sys.stderr, flush=True)
            continue
        print(f"[{done}/{total}] {video_id} ({manifest[video_id]:.0f} s video)", flush=True)
        started = time.monotonic()
        try:
            run(path, args.outputs / f"{video_id}.json")
            print(f"  done in {time.monotonic() - started:.0f} s", flush=True)
        except ModelSetupError as exc:
            print(f"Model setup failed; stopping: {exc}", file=sys.stderr)
            return 2
        except (OSError, ValueError, VideoError) as exc:
            failed[video_id] = str(exc)
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
        finally:
            if nbytes:
                if not args.keep:
                    path.unlink(missing_ok=True)
                fetcher.release(nbytes)

    print(f"\n{skipped} already valid, {done - len(failed)} generated, {len(failed)} failed/skipped, "
          f"{len(missing)} not found")
    for video_id, reason in failed.items():
        print(f"  failed  {video_id}: {reason}")
    for video_id in missing:
        print(f"  missing {video_id} ({manifest[video_id]:.2f} s) — pass --fetch DIR to download it")

    results = check_directory(args.outputs, manifest, classes, args.team_id.strip())
    problems = sum(bool(failures(issues, strict=True)) for issues in results.values())
    print(f"validate.py --strict: {'all outputs valid' if not problems else f'{problems} problem group(s)'}"
          f" — run `python tools/validate.py --dir {args.outputs} --strict -v` for details")
    return 0 if not (failed or missing or problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
