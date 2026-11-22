"""Download the three local model assets into weights/ (pinned, verified).

    python tools/download_weights.py

* hand_landmarker.task: MediaPipe Hand Landmarker float16 (Apache-2.0), SHA-256 checked.
* owlv2/: google/owlv2-base-patch16-ensemble at a pinned revision (Apache-2.0).
* siglip/: google/siglip-base-patch16-224 at a pinned revision (Apache-2.0).

Pinned revisions keep every machine on identical weights. Re-running skips files that verify.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
            "hand_landmarker.task")
HAND_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
SNAPSHOTS = {
    "owlv2": ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
    "siglip": ("google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"),
}
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    WEIGHTS.mkdir(exist_ok=True)
    hand = WEIGHTS / "hand_landmarker.task"
    if not (hand.is_file() and sha256(hand) == HAND_SHA256):
        print(f"fetch {HAND_URL}")
        urllib.request.urlretrieve(HAND_URL, hand)
        if sha256(hand) != HAND_SHA256:
            print("hand_landmarker.task checksum mismatch", file=sys.stderr)
            return 1
    print("ok    hand_landmarker.task")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install the requirements first: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    for folder, (repo, revision) in SNAPSHOTS.items():
        snapshot_download(repo, revision=revision, local_dir=WEIGHTS / folder, allow_patterns=PATTERNS)
        print(f"ok    {folder}/ ({repo}@{revision[:7]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
