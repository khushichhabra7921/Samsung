#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${XR_PYTHON:-python3}"
if [[ -z "${XR_PYTHON:-}" && -x "$root/.venv/bin/python" ]]; then
  python="$root/.venv/bin/python"
fi
export PYTHONPATH="$root:$root/BiosHackathon/src${PYTHONPATH:+:$PYTHONPATH}"

# Track 04's required interface is exactly: ./run.sh <video.mp4> <out.json>
if [[ $# -eq 2 && "$1" != -* && "$2" != -* ]]; then
  exec "$python" -m handscribe_xr.track04 "$@"
fi

# Backward-compatible development CLI for the Studio/project workflow.
exec "$python" -m handscribe_xr summarize "$@"
