# HandScribe XR

First-person video highlights for **Samsung XR Hackathon, Track 04**. Built by Team 404NotFound at Thapar Institute of Engineering & Technology. The Studio includes the team's original kitchen clip and its computed highlight, motion trace, and 21-point MediaPipe tracks.

## Run the Studio

Use Python 3.12 and FFmpeg with H.264 encoding. On Debian or Ubuntu, install the native libraries first.

```sh
sudo apt-get install ffmpeg libegl1 libgl1 libgles2 libportaudio2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./serve.sh --port 4173
```

Open `http://127.0.0.1:4173`. Play the sample, switch between **Source** and **Highlights**, click an interval to seek to it, and download the report or silent highlight MP4. Uploads run through the same engine and support MP4s up to 512 MiB, 35 minutes, and 4K. H.264 works in most browsers. **Remove uploaded video** deletes the current upload and its exports.

The app runs one analysis at a time. It keeps up to four finished jobs in a private temporary directory, expires old results on subsequent requests, and removes the directory on shutdown. The server binds to loopback by default. For a static demo, serve `web/` independently. The bundled sample works in static mode.

### Analyze from the command line

```sh
./run.sh input.mp4 outputs/result.json --highlight outputs/highlights.mp4
./run.sh web/assets/kitchen-demo.mp4 outputs/sample.json --allow-short \
  --highlight outputs/sample.mp4 --trace outputs/sample.trace.json
```

The report uses integer milliseconds for intervals and budgets. The selector enforces a hard 15% cap, selects chronological non-overlapping cuts, and requires each full-length highlight to be at least four seconds. A source shorter than 26.667 seconds cannot meet both constraints. The CLI reports `ineligible_short` with no intervals by default. The Studio and `--allow-short` can show a shorter cut as `short_clip_preview`, with the 15% cap preserved. An unreadable MP4 exits with an error and no newly written result.

### Container

```sh
docker build -t handscribe-xr .
docker run --rm -p 127.0.0.1:4173:8000 handscribe-xr
```

The image runs the CPU preview. To use MediaPipe in the preview image, mount the verified model as `/app/weights/hand_landmarker.task` read-only. If your build environment uses a TLS proxy, pass its trusted CA bundle with `docker build --secret id=pip_ca,src=/path/to/ca-bundle.pem -t handscribe-xr .`. The Studio processes videos without cloud APIs, account credentials, or outbound downloads at runtime.

## Analysis modes

| Mode | Evidence and output |
| --- | --- |
| **CPU preview** | Blur and lens checks, frame motion, and visual novelty. With a local MediaPipe task file, it also uses tracked wrist motion and can display real hand landmarks. Reports contain no semantic action labels. |
| **Semantic** | MediaPipe tracks, OWLv2 object detections, and SigLIP verb and noun scores. Two or more fingertips in the box count as contact. Missing hand evidence is omitted and weights are renormalized. Contradictory contact reduces confidence. A video-level pair aggregates visual scores across the clip. |

The Samsung pipeline reuses the BIOS hand detector, ID tracking, side voting, One-Euro smoothing, and three-second re-identification. It bridges brief dropouts between same-ID observations. It refines semantic action boundaries within ±1.2 seconds of hand-motion minima and recomputes temporal evidence. A deterministic greedy selection step favors new verb–noun pairs, reduces redundant intervals, and checks the budget again before writing the report. Rendered highlights use source footage and are checked against the duration cap before delivery.

### Local semantic assets

Install the appropriate PyTorch wheels for your device. These versions were checked on Linux CPU. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```sh
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.11.0+cpu torchvision==0.26.0+cpu
.venv/bin/python -m pip install -r requirements-semantic.txt
.venv/bin/python BiosHackathon/scripts/download_weights.py
```

The last command downloads and verifies the MediaPipe task file. It is also the only model needed to add hand tracks to the CPU preview. On a connected machine, download both model snapshots with their processor, tokenizer, and Safetensors files. [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) · [SigLIP](https://huggingface.co/google/siglip-base-patch16-224)

```text
weights/
├── hand_landmarker.task
├── owlv2/
└── siglip/
```

Pass a normalized vocabulary with the Track 04 label strings as `{"verbs": [...], "nouns": [...]}`. The loader accepts up to 98 verbs and 301 nouns. If the vocabulary includes `other`, weak predictions can map to that class. Use the official kit labels for submission integration.

```sh
./run.sh input.mp4 outputs/result.json --mode semantic \
  --weights weights --vocabulary path/to/track04-labels.json
./serve.sh --port 4173 --mode semantic \
  --weights weights --vocabulary path/to/track04-labels.json
```

Model loading uses local files only. `--device auto` chooses CUDA when available. Use `--device cpu` to force CPU or `--device cuda` to require a GPU. Semantic mode reports a setup error if required assets are missing.

## Track 04 submission status

The official student ZIP is now inspected and the Track 04 adapter is implemented at `handscribe_xr/track04.py`. Root `run.sh` accepts the required two-argument form `./run.sh <video.mp4> <out.json>` and routes it through semantic inference and the official-schema exporter. The previous development CLI remains available for invocations with flags or additional arguments.

Before running, extract the organiser ZIP and set:
```sh
export TEAM_ID="<exact organiser-assigned team ID>"
export TRACK04_KIT_DIR="/path/to/Student package_track04"
./run.sh P02_109.mp4 outputs/P02_109.json
```

**Not yet fully submission-ready:** the uploaded ZIP omitted the promised `validate.py`; the current system has no FPVSum video-category classifier (the adapter explicitly falls back to `other` with score 0.0 for those 49 videos); and some manifest videos are too short to satisfy both the 4-second minimum highlight and the 15% duration cap. The exact team ID is also not confirmed. See [Track 04 submission readiness](docs/TRACK04_SUBMISSION_READINESS.md) for details. Do not claim official validator compliance until the organiser's actual validator has been run.


**Important:** the current `handscribe.xr/v1` JSON is a development report, not yet the organiser's final submission schema. The official student kit (`manifest.csv`, `supported_classes.json`, `validate.py`, `TEAM_README.md`, and sample JSONs) is required before the export adapter and final package can be completed. Do not submit the development JSON files as official predictions.

The intended package contains `run.sh`, `code/`, `weights/`, `requirements.txt`, and `outputs/` (one official-schema JSON per video), plus a reproducibility README. See [Track 04 submission readiness](docs/TRACK04_SUBMISSION_READINESS.md) for the exact blockers, current CLI invocation, packaging checklist, and presentation claims that need reconciliation.

## Validation and submission

Run `.venv/bin/python -m pip install -r requirements-dev.txt`, then `.venv/bin/python -m pytest`. The tests exercise evidence fusion, gap filling, boundary refinement, coverage, strict budget handling, deterministic processing, and a real upload-to-export flow.

The bundled kitchen clip produces a **3.499 s source interval** from **23.333 s** of video. The rendered MP4 is **3.467 s** after frame quantization. The report and track are generated with the project's CLI. Action accuracy and dataset-wide performance require evaluation with the production checkpoints and labeled clips.

The JSON is a versioned development report, `handscribe.xr/v1`. To submit, obtain the official student-kit schema, `supported_classes.json`, manifest, and validator. Map the normalized vocabulary and report to that contract, then run the organizer's validation and benchmark. Semantic confidence weights and thresholds still require calibration with labeled data.

The original project is available under `BiosHackathon/`. [BIOS HandScribe workflow](BiosHackathon/README.md)
