#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${XR_PYTHON:-python3}"
if [[ -z "${XR_PYTHON:-}" && -x "$root/.venv/bin/python" ]]; then
  python="$root/.venv/bin/python"
fi
export PYTHONPATH="$root:$root/BiosHackathon/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m handscribe_xr summarize "$@"
