#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${XR_PYTHON:-python3}"
if [[ -z "${XR_PYTHON:-}" && -x "$root/.venv/bin/python" ]]; then
  python="$root/.venv/bin/python"
fi
export PYTHONPATH="$root:$root/BiosHackathon/src${PYTHONPATH:+:$PYTHONPATH}"

# Submission contract: exactly two positional arguments use the official Track 04
# exporter. Calls with flags or additional arguments retain the legacy dev CLI.
if [[ "$#" -eq 2 ]]; then
  exec "$python" -m handscribe_xr.track04 "$1" "$2"
fi
exec "$python" -m handscribe_xr summarize "$@"
