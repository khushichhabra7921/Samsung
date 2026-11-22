"""Build the TEAM_ID archive after all 187 official outputs and weights are ready."""
from __future__ import annotations
import argparse, csv, json, math, os, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"team_id", "video_id", "duration_sec", "label", "actions", "highlights"}

def check_outputs(outputs: Path, team_id: str) -> None:
    with (ROOT / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {row["video_id"]: row for row in csv.DictReader(stream)}
    files = list(outputs.glob("*.json"))
    if {p.stem for p in files} != set(manifest):
        raise ValueError("outputs must contain exactly one JSON for each of the 187 manifest IDs.")
    classes = json.loads((ROOT / "supported_classes.json").read_text(encoding="utf-8"))
    labels, verbs, nouns = set(classes["labels"]), set(classes["verbs"]), set(classes["nouns"])
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != KEYS:
            raise ValueError(f"{path.name}: incorrect six-field schema.")
        if data["team_id"] != team_id or data["video_id"] != path.stem:
            raise ValueError(f"{path.name}: team_id/video_id mismatch.")
        duration = data["duration_sec"]
        if isinstance(duration, bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{path.name}: invalid duration.")
        if abs(duration-float(manifest[path.stem]["duration_sec"])) > 0.5:
            raise ValueError(f"{path.name}: duration differs from manifest by more than 0.5s.")
        label=data["label"]
        if not isinstance(label,dict) or label.get("name") not in labels:
            raise ValueError(f"{path.name}: unsupported label.")
        score=label.get("score")
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{path.name}: invalid label score.")
        if not isinstance(data["actions"],list) or len(data["actions"]) > 2000:
            raise ValueError(f"{path.name}: invalid actions array.")
        for action in data["actions"]:
            if not isinstance(action,dict) or not {"segment","verb","noun","score"} <= set(action):
                raise ValueError(f"{path.name}: malformed action.")
            seg=action["segment"]
            if not isinstance(seg,list) or len(seg)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in seg) or not 0 <= seg[0] < seg[1] <= duration+0.5:
                raise ValueError(f"{path.name}: invalid action segment.")
            if action["verb"] not in verbs or action["noun"] not in nouns:
                raise ValueError(f"{path.name}: unsupported verb/noun.")
            value=action["score"]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path.name}: invalid action score.")
        highlights=data["highlights"]
        if not isinstance(highlights,list) or not highlights:
            raise ValueError(f"{path.name}: highlights must be non-empty.")
        previous,total,longest=-1.0,0.0,0.0
        for item in highlights:
            seg=item.get("segment") if isinstance(item,dict) else None
            if not isinstance(seg,list) or len(seg)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in seg) or not 0 <= seg[0] < seg[1] <= duration+0.5:
                raise ValueError(f"{path.name}: invalid highlight segment.")
            start,end=seg
            if start < previous: raise ValueError(f"{path.name}: highlights overlap or are unsorted.")
            previous=end; total+=end-start; longest=max(longest,end-start)
        if longest < 4.0: raise ValueError(f"{path.name}: no highlight is at least 4.0 seconds.")
        if total > duration*0.15+1e-9: raise ValueError(f"{path.name}: highlights exceed 15% budget.")

def build(team_id: str, outputs: Path, destination: Path) -> Path:
    if not team_id.strip() or "/" in team_id or "\\" in team_id:
        raise ValueError("Provide the exact organiser-assigned TEAM_ID.")
    check_outputs(outputs,team_id)
    required=[ROOT/"weights"/"hand_landmarker.task",ROOT/"weights"/"owlv2"/"config.json",ROOT/"weights"/"siglip"/"config.json"]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing: raise ValueError("Required model assets missing: "+", ".join(missing))
    package=destination/team_id
    if package.exists(): shutil.rmtree(package)
    (package/"code").mkdir(parents=True)
    shutil.copytree(ROOT/"handscribe_xr",package/"code"/"handscribe_xr",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(ROOT/"BiosHackathon"/"src"/"egovision",package/"code"/"egovision",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(ROOT/"weights",package/"weights",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(outputs,package/"outputs")
    for name in ("manifest.csv","supported_classes.json"): shutil.copy2(ROOT/name,package/name)
    base=(ROOT/"requirements.txt").read_text(encoding="utf-8").rstrip()
    semantic=(ROOT/"requirements-semantic.txt").read_text(encoding="utf-8").splitlines()
    semantic=[line for line in semantic if line.strip() and not line.lstrip().startswith("#")]
    (package/"requirements.txt").write_text(base+"\n"+"\n".join(semantic)+"\n",encoding="utf-8")
    (package/"requirements-torch.txt").write_text(
        "# Install a PyTorch wheel compatible with the target device.\n"
        "# Tested CPU example:\n"
        "python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu torchvision==0.26.0+cpu\n",encoding="utf-8")
    (package/"run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'root="$(cd -- "$(dirname -- "$0")" && pwd)"\n'
        "python=python3\n"
        'if [ -n "$XR_PYTHON" ]; then python="$XR_PYTHON"; fi\n'
        'if [ -z "$XR_PYTHON" ] && [ -x "$root/.venv/bin/python" ]; then python="$root/.venv/bin/python"; fi\n'
        'export PYTHONPATH="$root/code:$PYTHONPATH"\n'
        'exec "$python" -m handscribe_xr.track04 "$@"\n',encoding="utf-8")
    os.chmod(package/"run.sh", 0o755)
    os.chmod(package/"run.sh", 0o755)
    (package/"README.md").write_text(
        "HandScribe XR — Track 04 submission\n\n"
        f"Team ID: {team_id}\n\n"
        "Install Python 3.12 dependencies from requirements.txt and install a matching PyTorch wheel "
        "using requirements-torch.txt. Local MediaPipe, OWLv2 and SigLIP assets are included under weights/.\n\n"
        "Set HANDSCRIBE_TEAM_ID to the assigned ID, then run:\n\n"
        f"export HANDSCRIBE_TEAM_ID={team_id}\n"
        "./run.sh /path/to/VIDEO_ID.mp4 outputs/VIDEO_ID.json\n\n"
        "Input and output filename stems must match manifest.csv. Run the organiser-provided "
        "python validate.py --dir outputs/ when supplied.\n",encoding="utf-8")
    archive=destination/(team_id+".zip")
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for path in package.rglob("*"):
            if path.is_file(): bundle.write(path,path.relative_to(destination))
    return archive

def main() -> int:
    parser=argparse.ArgumentParser(description="Preflight and build the Track 04 TEAM_ID archive.")
    parser.add_argument("--team-id",default=os.environ.get("HANDSCRIBE_TEAM_ID",""))
    parser.add_argument("--outputs",type=Path,default=ROOT/"outputs")
    parser.add_argument("--destination",type=Path,default=ROOT)
    args=parser.parse_args()
    try:
        archive=build(args.team_id,args.outputs.resolve(),args.destination.resolve())
        print(f"Created {archive}")
        return 0
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        print(f"Packaging failed: {exc}",file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
