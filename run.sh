#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Organiser-assigned team ID, used when HANDSCRIBE_TEAM_ID is not set. Fill this in once the
# ID is confirmed so the organisers' re-run of `./run.sh <video.mp4> <out.json>` needs no
# environment variables. (tools/build_package.py --team-id also writes it into the package.)
DEFAULT_TEAM_ID=""

python="${XR_PYTHON:-python3}"
if [[ -z "${XR_PYTHON:-}" && -x "$root/.venv/bin/python" ]]; then
  python="$root/.venv/bin/python"
fi
if [[ -z "${HANDSCRIBE_TEAM_ID:-}" && -n "$DEFAULT_TEAM_ID" ]]; then
  export HANDSCRIBE_TEAM_ID="$DEFAULT_TEAM_ID"
fi
export PYTHONPATH="$root/code${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m handscribe_xr.track04 "$@"
