# HandScribe XR — Samsung PRISM 2026, Track 04

Action-aware, multi-evidence highlights from egocentric video. This directory follows the organiser's required package layout.

## Package layout

```
TEAM_ID/
├── run.sh
├── code/                 # inference source
├── weights/              # place local model assets here
├── requirements.txt
├── outputs/              # one <video_id>.json per manifest video
└── README.md
```

The official `manifest.csv` and `supported_classes.json` are stored in `code/` alongside the inference source, keeping the package root limited to the six required entries.

## Setup

Use Python 3.12 in a virtual environment. Install a PyTorch/torchvision build compatible with your machine first, then:

```bash
python -m pip install -r requirements.txt
```

Place the local assets in `weights/`:
- `hand_landmarker.task`
- `owlv2/` model snapshot
- `siglip/` model snapshot

The OWLv2 and SigLIP snapshots are not included in this repository; add the exact offline model files before running semantic inference. No cloud API is used.

## Run

Set the organiser-assigned team ID (do not use the sample ID from the student kit):

```bash
export HANDSCRIBE_TEAM_ID="<assigned-team-id>"
./run.sh /path/to/VIDEO_ID.mp4 outputs/VIDEO_ID.json
```

The input filename stem and output filename stem must exactly match a `video_id` in `manifest.csv`.

## Validate

When the organiser supplies the missing validator file, run:

```bash
python validate.py --dir outputs/
```

The student-kit guide references `validate.py`, but it was absent from the ZIP shared with the team. The repository's local tests are not a replacement for the official validator.

## Dataset and output status

Raw EPIC-KITCHENS-100 and FPVSum videos are not bundled. Download the source videos separately and process them using their exact manifest IDs. The `outputs/` directory is intentionally a placeholder until real inference is run; no synthetic predictions are included.

Three manifest entries have a 15% budget shorter than the stated 4-second minimum highlight. These conflicting constraints require organiser clarification before those entries can be finalized.
