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

## Track 04 submission interface

The official Student Package files `manifest.csv` and `supported_classes.json` are at the repository root. When `run.sh` gets exactly two arguments it runs the six-field Track 04 exporter. With flags or more arguments it keeps the development CLI.

```sh
export HANDSCRIBE_TEAM_ID=T07  # replace with the exact organiser-assigned ID if different
./run.sh /path/to/VIDEO_ID.mp4 outputs/VIDEO_ID.json
```

The video ID is the input file name without its extension, for example `P01_11.MP4` gives `P01_11`. If that isn't in `manifest.csv`, the output file name is tried. This lets an organiser re-run like `./run.sh sample.mp4 out.json` still work. A video outside the manifest is processed with its decoded duration and a label predicted from all 16 classes. The exporter runs semantic inference and writes `team_id`, `video_id`, `duration_sec`, `label`, `actions`, and `highlights`. EPIC-KITCHENS videos use the Student Package's prescribed `cooking` label. FPVSum videos get the top SigLIP zero-shot score among the 14 FPVSum categories. Its accuracy still needs checking on labelled data.

Before writing, the exporter checks or enforces:
- duration within ±0.5 s of the manifest, and every class from the official vocabulary;
- highlights in time order with touching segments merged;
- every highlight at least 4.0 s *after* rounding to two decimals (0.02 → 4.02 is 3.9999999999999996 s as a float);
- a total within the 15% budget.

Install base and semantic dependencies as described above. Semantic mode needs local assets at `weights/hand_landmarker.task`, `weights/owlv2/`, and `weights/siglip/`. They aren't committed to Git because of their size. Set `HANDSCRIBE_DEVICE=cpu` or `cuda` to choose a device.

**Videos shorter than 26.67 s** (`P03_26`, `P26_30`, `P28_23` in the manifest) cannot meet both the 4.0 s minimum and the 15% budget. The Student Package treats "at least one highlight ≥ 4.0 s" as a hard check: a file that fails it is invalid and scores 0. Exceeding the budget only produces a warning and a soft penalty on coverage. These videos therefore get one 4.0 s highlight centred on their best-scoring moment.

### Generate all 187 outputs

Download the videos as described in `TEAM_README.md`, then run:

```sh
python scripts/generate_outputs.py --videos /path/to/downloads
```

The script finds `<video_id>.mp4` in any letter case anywhere under the folder, including the EPIC downloader's `P01/videos/P01_11.MP4` layout. It loads the models once and skips outputs that already pass validation, so an interrupted run can resume. At the end it lists failures, any manifest videos it could not find, and a strict validation result.

### Validate

The Student Package ZIP we received did not include `validate.py`. The root `validate.py` is our standard-library version of every check that `TEAM_README.md` §3–4 and the submission slides describe. It covers the six keys, types, finite ranges, segment bounds, order and overlap, and the 4.0 s minimum. It also checks manifest duration and coverage, a consistent `team_id`, and vocabulary membership.

```sh
python validate.py --dir outputs/            # README severities: budget overrun and unknown classes warn
python validate.py --dir outputs/ --strict   # also fail avoidable budget overruns, unknown classes and keys
python validate.py --file examples/P02_109.json -v
```

The organiser's sample `skiing_03.json` uses `skiing`, which is not in `supported_classes.json` (`ski`). Default mode warns about this, and `--strict` fails it. When the official script arrives, run it as well.

### Build the final TEAM_ID archive

After generating all 187 outputs and placing the model assets under `weights/`, run:

```sh
python scripts/build_submission_package.py --team-id T07
```

Replace `T07` with the confirmed organiser-assigned ID. The packager runs `validate.py --strict` over the outputs, then creates `TEAM_ID/` and `TEAM_ID.zip`. These contain `run.sh` (with the team ID as its default), `code/`, `weights/`, `requirements.txt` with exact pins including torch, `outputs/`, and `README.md`. The README is built from [docs/SUBMISSION_README.md](docs/SUBMISSION_README.md) and covers setup, model architecture, and data used. The packager refuses to build if any output is missing or invalid, or if a required model file is missing.

Run the regression suite with `.venv/bin/python -m pip install -r requirements-dev.txt` followed by `.venv/bin/python -m pytest`. The original project is available under `BiosHackathon/`. [BIOS HandScribe workflow](BiosHackathon/README.md)
