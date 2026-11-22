# HandScribe XR — Samsung PRISM 2026, Track 04

Action-aware, multi-evidence highlights from egocentric video. HandScribe XR is built by Team 404NotFound at Thapar Institute of Engineering & Technology. This directory follows the organiser's required package layout.

## Package layout

```
TEAM_ID/
├── run.sh                # ./run.sh <video.mp4> <out.json>
├── code/                 # inference source (handscribe_xr, egovision) + manifest.csv, supported_classes.json
├── weights/              # local model files: hand_landmarker.task, owlv2/, siglip/
├── requirements.txt      # exact Python dependencies
├── outputs/              # one <video_id>.json per manifest video
└── README.md
```

The official `manifest.csv` and `supported_classes.json` are stored in `code/` alongside the inference source, so the package root holds only the six required entries. In the repository, `presentation/`, `demo-video/` and `tools/` are team material and are not part of the uploaded package. See [tools/README.md](tools/README.md).

## Setup

Use Python 3.12 in a virtual environment. On Debian or Ubuntu, first install the native libraries that MediaPipe and OpenCV need:

```bash
sudo apt-get install ffmpeg libegl1 libgl1 libgles2 libportaudio2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.11.0` and `torchvision==0.26.0`. For CPU-only or a specific CUDA build, install that wheel first, e.g. `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu torchvision==0.26.0+cpu`. Pip accepts it as meeting the pins.

In a fresh clone, fetch the pinned model files with `python tools/download_weights.py`. Inference loads only the local files in `weights/` (`hand_landmarker.task`, `owlv2/`, `siglip/`) with Hugging Face offline mode on. It needs no network access, cloud API or account.

## Run

```bash
./run.sh /path/to/P01_11.MP4 outputs/P01_11.json
```

- **Team ID:** set `HANDSCRIBE_TEAM_ID`. The `DEFAULT_TEAM_ID` line in `run.sh` holds the organiser-assigned ID, so re-runs need no environment variables.
- **Device:** `HANDSCRIBE_DEVICE=cpu|cuda|auto` chooses the device. The default is `auto`, which uses CUDA when available.
- **Video ID:** this is the input file name without its extension. If that isn't in `code/manifest.csv`, the output file name is tried. If neither matches, the video is still processed, using its decoded duration and a label predicted from all 16 classes.
- **Output name:** the output file can have any name, but the submitted file must be named `<video_id>.json`.

## Model architecture

No network is trained or fine-tuned. The pipeline combines three public pretrained models, a verb–noun co-occurrence prior counted from the EPIC-KITCHENS-100 train split, and deterministic temporal logic.

| Stage | Component | What it does |
|---|---|---|
| Decode | PyAV | Decodes the MP4 with timestamps. Every second frame is analysed, and a keyframe is taken every 330 ms. |
| Quality | OpenCV | Checks each keyframe for blur and a covered lens, and measures frame-difference motion. |
| Hands | MediaPipe Hand Landmarker (float16) | Gives 21 landmarks per hand. Hands are then tracked by ID, voted left or right, smoothed with a One-Euro filter, re-identified within 3 s, and bridged across brief dropouts. The output is wrist motion per keyframe. |
| Objects | OWLv2 (`google/owlv2-base-patch16-ensemble`) | Detects the 300 non-`other` nouns from the open vocabulary. A box counts as hand contact when two or more fingertips fall inside it. |
| Verb, noun and category scores | SigLIP (`google/siglip-base-patch16-224`) | Compares each keyframe embedding with text prompts for the 98 verbs, 301 nouns and 16 video labels. A noun's score combines SigLIP (0.60), OWLv2 detection (0.25) and hand contact (0.15). |
| Verb choice | Train-split prior | SigLIP verb log-odds are centred per video, which removes scene-wide bias such as `cook` on every kitchen frame. They are then combined with P(verb \| noun), counted from the EPIC-KITCHENS-100 **train** annotations. The posterior's top verb is used. `other` is never emitted, because it cannot match EPIC labels. |
| Actions | Temporal grouping | Consecutive keyframes with the same verb–noun pair form one action. Its edges move to the nearest hand-motion minimum within ±1.2 s. Confidence combines visual score (0.55), contact (0.22), hand motion (0.08) and temporal consistency (0.15), and is reduced when a different object is in contact. Each segment is written with its five most likely verbs for the detected noun. Each alternative is scored by its posterior relative to the top verb, so it always ranks below it. |
| Highlights | Greedy budgeted selection | Candidate windows of about 4–8 s snap to motion minima. Each is scored on action confidence, novelty, contact, motion and image quality. The selector then adds windows under the 15% budget, favouring verb–noun pairs not yet covered and visually distinct footage. |
| Video label | Zero-shot SigLIP | EPIC-KITCHENS-100 videos get `cooking`, as the Student Package prescribes. FPVSum videos get the most likely of the 14 FPVSum categories, averaged over all keyframes. |

### Output rules enforced before writing

- Times are in seconds, rounded to 2 decimals. Every highlight lasts at least 4.0 s even after float rounding. Highlights are in time order, and touching segments are merged.
- Total highlight time stays within 15% of `duration_sec`, and `duration_sec` is within ±0.5 s of `manifest.csv`.
- Every label, verb and noun comes from `supported_classes.json`. Anything outside it maps to `other`.
- FPVSum videos get `"actions": []`, which the student kit allows. Actions are scored only on EPIC-KITCHENS, and the verb/noun vocabulary is kitchen-specific.
- **Videos shorter than 26.67 s** (`P03_26`, `P26_30` and `P28_23` in the manifest): their 15% budget is under 4.0 s, so no highlight can satisfy both rules. The Student Package treats the 4.0 s minimum as a hard check and the budget as a warning with a soft scoring penalty. These videos therefore get a single 4.0 s highlight centred on their best-scoring moment.

## Data used

- **Inputs:** the 187 videos in `code/manifest.csv`, downloaded by the team from the official sources. These are 138 EPIC-KITCHENS-100 validation videos and 49 FPVSum videos. Video files are not included in this package.
- **Pretrained weights:** MediaPipe Hand Landmarker (Apache-2.0), OWLv2 and SigLIP (Apache-2.0, from Hugging Face). They are used as released, with no fine-tuning.
- **Vocabulary:** `code/supported_classes.json` from the Student Package.
- **EPIC-KITCHENS-100 train-split annotations** (public, `EPIC_100_train.csv`): used only to count how often each verb occurs with each noun. The counts are stored in `code/handscribe_xr/assets/epic100_train_prior.json` (3,568 pairs from 67,217 train actions) and rebuilt by `tools/build_action_prior.py`. That script refuses to run if any manifest video is in its input. None of the 138 manifest videos is in the train split; all are validation videos.
- **Evaluation-set annotations:** none. No EPIC-KITCHENS-100 validation or FPVSum ground truth was used for training, tuning, or inference. All other thresholds and weights are fixed heuristics in `code/handscribe_xr/`.
- **Manifest use:** at inference time `manifest.csv` is used only to check the decoded duration and to tell an EPIC-KITCHENS video from an FPVSum one. The code contains no video IDs or per-video outputs.

## Validate

```bash
python tools/validate.py --dir outputs/ --strict -v
```

The Student Package lists `validate.py`, but it was missing from the ZIP shared with the team. `tools/validate.py` is the team's standard-library version of every check that the student-kit guide and submission slides describe. When the organiser supplies the official script, run it as well.
