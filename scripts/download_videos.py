"""Download the 187 Track 04 videos listed in manifest.csv from their official sources.

    python scripts/download_videos.py --dest /path/with/200GB/free              # everything
    python scripts/download_videos.py --dest DIR --dataset fpvsum               # 49 FPVSum videos only
    python scripts/download_videos.py --dest DIR --ids P26_30,Ski_-Z5a4UQmZLs   # selected videos
    python scripts/download_videos.py --dest DIR --list                         # print URLs only

Sources (TEAM_README section 2):
* EPIC-KITCHENS-100 (138 videos, ~184 GB): the official data.bris.ac.uk endpoint. URLs and MD5
  checksums come from the official epic-kitchens-download-scripts repository. All 138 manifest
  videos are EPIC-55-era recordings stored under videos/test/. Downloads resume and are MD5-checked.
* FPVSum (49 videos): YouTube, via yt-dlp. The FPVSum names are not always the YouTube ID
  (e.g. BikePolo_5yf797Lh4Oc_1 is 6oOSbkUiJY0), so the official FPVSum_videolist.csv is used.

Files are written as DEST/epic/<video_id>.MP4 and DEST/fpvsum/<video_id>.mp4, so
`python scripts/generate_outputs.py --videos DEST` finds them. Every file's duration is checked
against manifest.csv (+/-0.5 s) at the end.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPIC_55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
EPIC_100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
EPIC_MD5 = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/master/data/md5.csv"
FPVSUM_LIST = "https://raw.githubusercontent.com/azuxmioy/fpvsum/master/crawler/FPVSum_videolist.csv"
CHUNK = 8 << 20


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def epic_sources(ids: list[str]) -> dict[str, tuple[str, str]]:
    """video_id -> (url, md5) from the official checksum list."""
    found = {}
    for md5, path, version in csv.reader(io.StringIO(fetch_text(EPIC_MD5))):
        name = path.rsplit("/", 1)[-1]
        if "videos/" in path and name.endswith(".MP4") and name[:-4] in ids:
            found[name[:-4]] = (f"{EPIC_55 if version == '55' else EPIC_100}/{path}", md5)
    return found


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download_epic(video_id: str, url: str, md5: str, folder: Path) -> bool:
    target = folder / f"{video_id}.MP4"
    if target.is_file() and md5sum(target) == md5:
        print(f"ok      {video_id} (already downloaded)")
        return True
    partial = target.with_suffix(".MP4.part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, \
                    partial.open("ab" if response.status == 206 else "wb") as stream:
                total = offset + int(response.headers.get("Content-Length", 0))
                print(f"fetch   {video_id} {total / 1e9:.2f} GB" + (" (resuming)" if response.status == 206 else ""))
                shutil.copyfileobj(response, stream, CHUNK)
        except OSError as exc:
            print(f"retry   {video_id}: {exc}", file=sys.stderr)
            continue
        if md5sum(partial) == md5:
            partial.replace(target)
            print(f"ok      {video_id}")
            return True
        print(f"bad md5 {video_id}; downloading again", file=sys.stderr)
        partial.unlink()
    return False


def youtube_downloader() -> list[str] | None:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "yt_dlp"]


def download_fpvsum(video_id: str, youtube_id: str, folder: Path, command: list[str], height: int) -> bool:
    target = folder / f"{video_id}.mp4"
    if target.is_file():
        print(f"ok      {video_id} (already downloaded)")
        return True
    # H.264 video-only MP4 needs no ffmpeg merge; format 18 (360p MP4) is the fallback.
    selector = f"bv*[vcodec^=avc1][ext=mp4][height<={height}]/b[ext=mp4][height<={height}]/b[ext=mp4]"
    result = subprocess.run([
        *command, "--quiet", "--no-warnings", "--no-playlist", "-f", selector,
        "-o", str(target), f"https://www.youtube.com/watch?v={youtube_id}",
    ], check=False)
    ok = result.returncode == 0 and target.is_file()
    print(f"{'ok' if ok else 'FAILED':<7} {video_id} (YouTube {youtube_id})")
    return ok


def check_durations(dest: Path, manifest: dict[str, dict[str, str]]) -> list[str]:
    try:
        import av
    except ImportError:
        print("PyAV not installed; skipping the duration check.")
        return []
    problems = []
    for path in sorted(dest.glob("*/*")):
        if path.stem in manifest and path.suffix.lower() == ".mp4":
            with av.open(str(path)) as container:
                seconds = (container.duration or 0) / av.time_base
            expected = float(manifest[path.stem]["duration_sec"])
            if abs(seconds - expected) > 0.5:
                problems.append(f"{path.stem}: {seconds:.2f} s, manifest says {expected:.2f} s")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Track 04 manifest videos.")
    parser.add_argument("--dest", type=Path, required=True, help="Download folder (needs ~190 GB for all)")
    parser.add_argument("--dataset", choices=("all", "epic", "fpvsum"), default="all")
    parser.add_argument("--ids", help="Comma-separated video IDs to fetch (default: all in the dataset)")
    parser.add_argument("--max-height", type=int, default=720, help="FPVSum resolution cap (default 720)")
    parser.add_argument("--list", action="store_true", help="Print the source URLs without downloading")
    args = parser.parse_args()

    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["video_id"]: row for row in csv.DictReader(stream)}
    wanted = set(args.ids.split(",")) if args.ids else set(manifest)
    unknown = wanted - set(manifest)
    if unknown:
        parser.error(f"not in manifest.csv: {', '.join(sorted(unknown))}")
    epic = sorted(v for v in wanted if manifest[v]["dataset"] == "epic_kitchens_100" and args.dataset != "fpvsum")
    fpv = sorted(v for v in wanted if manifest[v]["dataset"] == "fpvsum" and args.dataset != "epic")

    sources = epic_sources(epic) if epic else {}
    youtube = {}
    if fpv:
        rows = csv.DictReader(io.StringIO(fetch_text(FPVSUM_LIST)))
        youtube = {row["video_name"]: row["youtube_id"] for row in rows}
    missing_source = [v for v in epic if v not in sources] + [v for v in fpv if v not in youtube]
    if args.list:
        for video_id in epic:
            print(video_id, sources.get(video_id, ("<no official URL>",))[0])
        for video_id in fpv:
            print(video_id, f"https://www.youtube.com/watch?v={youtube.get(video_id, '<unknown>')}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.dest).free / 1e9
    print(f"{len(epic)} EPIC + {len(fpv)} FPVSum videos -> {args.dest} ({free_gb:.0f} GB free)")
    if epic and free_gb < 1.4 * len(epic):
        print("warning: EPIC videos average ~1.3 GB each; this disk may fill up.", file=sys.stderr)

    failed = list(missing_source)
    if epic:
        (args.dest / "epic").mkdir(exist_ok=True)
        failed += [v for v in epic if v in sources and not download_epic(v, *sources[v], args.dest / "epic")]
    if fpv:
        command = youtube_downloader()
        if command is None:
            print("FPVSum needs yt-dlp: python -m pip install yt-dlp", file=sys.stderr)
            failed += fpv
        else:
            (args.dest / "fpvsum").mkdir(exist_ok=True)
            failed += [v for v in fpv if v in youtube and not download_fpvsum(
                v, youtube[v], args.dest / "fpvsum", command, args.max_height)]

    mismatched = check_durations(args.dest, manifest)
    print(f"\n{len(epic) + len(fpv) - len(failed)} downloaded or present, {len(failed)} failed")
    for video_id in failed:
        print(f"  failed   {video_id}")
    for line in mismatched:
        print(f"  duration {line}")
    return 1 if failed or mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
