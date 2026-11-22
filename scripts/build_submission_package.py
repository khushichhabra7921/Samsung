"""Build the TEAM_ID archive after all 187 official outputs and weights are ready."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate import check_directory, failures, load_classes, load_manifest


def check_outputs(outputs: Path, team_id: str) -> None:
    """Run validate.py in strict mode; every manifest video needs a clean JSON."""
    results = check_directory(
        outputs, load_manifest(ROOT / "manifest.csv"), load_classes(ROOT / "supported_classes.json"), team_id,
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
    check_outputs(outputs,team_id)
    required=[ROOT/"weights"/"hand_landmarker.task",ROOT/"weights"/"owlv2"/"config.json",ROOT/"weights"/"siglip"/"config.json"]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise ValueError("Required model assets missing: " + ", ".join(missing))
    package=destination/team_id
    if package.exists():
        shutil.rmtree(package)
    (package/"code").mkdir(parents=True)
    shutil.copytree(ROOT/"handscribe_xr",package/"code"/"handscribe_xr",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(ROOT/"BiosHackathon"/"src"/"egovision",package/"code"/"egovision",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(ROOT/"weights",package/"weights",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(outputs,package/"outputs")
    for name in ("manifest.csv", "supported_classes.json"):
        shutil.copy2(ROOT / name, package / name)
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
        'if [ -n "$(printenv XR_PYTHON 2>/dev/null || true)" ]; then python="$(printenv XR_PYTHON)"; fi\n'
        'if [ -z "$(printenv XR_PYTHON 2>/dev/null || true)" ] && [ -x "$root/.venv/bin/python" ]; then python="$root/.venv/bin/python"; fi\n'
        'export PYTHONPATH="$root/code"\n'
        'if [ -z "$(printenv HANDSCRIBE_TEAM_ID 2>/dev/null || true)" ]; then '
        f'export HANDSCRIBE_TEAM_ID={shlex.quote(team_id)}; fi\n'
        'exec "$python" -m handscribe_xr.track04 "$@"\n',encoding="utf-8")
    os.chmod(package/"run.sh", 0o755)
    readme = (ROOT / "docs" / "SUBMISSION_README.md").read_text(encoding="utf-8")
    (package / "README.md").write_text(readme.replace("{TEAM_ID}", team_id), encoding="utf-8")
    archive=destination/(team_id+".zip")
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for path in package.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(destination))
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
