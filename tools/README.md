# Team tools (not part of the uploaded package)

These scripts take the repository from "no videos" to `TEAM_ID.zip`. Run every command from the repository root. Install the tool dependencies with `python -m pip install -r tools/requirements.txt`.

## Workflow

| Step | Command | Notes |
|---|---|---|
| 0. Model files | `python tools/download_weights.py` | MediaPipe hand model (SHA-256 checked) plus OWLv2 and SigLIP at pinned revisions, so every machine runs identical weights. About 1.3 GB. |
| 1+2. Download and generate | `HANDSCRIBE_TEAM_ID=404NotFound python tools/generate_outputs.py --fetch /tmp/track04-videos` | Downloads each missing manifest video, writes `outputs/<video_id>.json`, then deletes the video. The next video downloads while the current one is processed. A few GB of free disk is enough; videos too big for the free space are listed at the end. Add `--dataset epic` or `--dataset fpvsum` to split the work across machines. Re-run it until it reports 0 failed: finished outputs are skipped and downloads resume. |
| (alt.) Download first | `python tools/download_videos.py --dest /data/track04` then `generate_outputs.py --videos /data/track04` | Needs about 190 GB. EPIC files come from `data.bris.ac.uk` over 8 parallel connections (~0.9 MB/s each, ~5 MB/s total measured), MD5-checked. FPVSum comes from YouTube via `yt-dlp`, using the official `FPVSum_videolist.csv`. |
| 3. Validate | `python tools/validate.py --dir outputs/ --strict -v` | Without `--strict`, it follows the student-kit README: budget overruns and unknown classes only warn. `--strict` fails them too. |
| 4. Team ID | `python tools/set_team_id.py <ID>` | Outputs are generated as `404NotFound` until the organiser ID is confirmed. This rewrites `team_id` in every output and the `DEFAULT_TEAM_ID` line in `run.sh`, with no re-processing. |
| 5. Package | `python tools/build_package.py --team-id <ID>` | Validates strictly, checks `weights/`, and writes `dist/<ID>/` and `dist/<ID>.zip`. |

### Speed

OWLv2 object detection runs on every keyframe (3 per second of video) and is about 95% of the cost.

| Machine | Measured or estimated speed | All 15.6 h of video |
|---|---|---|
| Apple M4 (CPU; its GPU was not faster) | ~5.5 s per second of video (measured) | ~3.5 days |
| NVIDIA GPU with `HANDSCRIBE_DEVICE=cuda` (OWLv2 and SigLIP in fp16) | not measured; expected several times faster | well under a day |

### Running EPIC on a free Colab GPU (any laptop, including Windows)

Open [tools/colab_epic.ipynb](colab_epic.ipynb) in Google Colab: **File → Open notebook → GitHub**, then paste `https://github.com/khushichhabra7921/Samsung/blob/main/tools/colab_epic.ipynb`.

1. Choose a **T4 GPU** runtime.
2. Set `PART`; each person runs a different one.
3. **Run all.**

Outputs go to `My Drive/track04_outputs` as they finish. After a disconnect, **Run all** again to resume. `--part` splits EPIC into fixed parts balanced by video length: 2 parts are 6.5 h and 6.8 h of video, and 3 parts are about 4.3–4.6 h each. Two or three Google accounts can work in parallel without repeating videos.

Colab speed has not been measured. Each video prints `done in … s`, so check the first few lines to see how long a part will take.

### Running on any machine with an NVIDIA GPU

```bash
git pull
python3.12 -m venv .venv          # Windows: py -3.12 -m venv .venv, then use .venv\Scripts\python
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r tools/requirements.txt
.venv/bin/python tools/download_weights.py
HANDSCRIBE_DEVICE=cuda HANDSCRIBE_TEAM_ID=404NotFound \
  .venv/bin/python tools/generate_outputs.py --dataset epic [--part K/N] --fetch <folder with ~10 GB free>
```

## Checks

```bash
python -m pytest -q tools/tests          # 51 tests, no model weights needed
python -m ruff check tools && python -m ruff check --line-length 110 code/handscribe_xr
```

## Verb prior and evaluation (EPIC train split only)

Previously, about 80% of actions were written with verb `other`, so Action mAP was 0. Single-frame SigLIP verb scores fell below the 0.12 cut-off, and `cook` came top on almost every kitchen frame. The fix has three parts:
- verb log-odds are centred per video;
- they are combined with P(verb | noun), counted from the public EPIC-100 **train** annotations;
- each segment is written with its five most likely verbs.

All 138 manifest EPIC videos are in the validation split. `build_action_prior.py` refuses to run if any manifest video is in its input, and `eval_epic_train.py` refuses to score any non-train video.

```bash
# annotation CSVs from https://github.com/epic-kitchens/epic-kitchens-100-annotations -> data/epic_annotations/
python tools/build_action_prior.py
python tools/eval_epic_train.py --outputs <folder of outputs for train-split videos>
```

Measured on three train-split videos (P02_08, P04_103, P30_112; 121 ground-truth actions):

| Version | Action mAP (tIoU 0.1–0.5) | Verb mAP | Action coverage |
|---|---:|---:|---:|
| Before | 0.00 | 0.00 | 0.217 |
| Prior + centring | 1.41 | 0.84 | 0.276 |
| + 5 verb hypotheses (current) | **2.83** | 2.81 | **0.276** |

Absolute scores are low, as expected for a zero-shot system without a trained action model. Three videos is a small sample.

## Decisions to know about

- **Videos under 26.67 s** (`P03_26`, `P26_30`, `P28_23`) can't meet both the ≥ 4.0 s and ≤ 15% rules. Each gets one 4.0 s highlight, because the ≥ 4.0 s rule invalidates the file while a budget overrun is only a warning with a soft penalty.
- **Sample label:** `tools/examples/skiing_03.json` (the organiser's sample) uses `skiing`, but `supported_classes.json` lists `ski`. We emit `ski`.
- **Schema:** the slides show a simplified schema. We follow the six-key format from the student-kit README and the example files.
- **FPVSum labels:** these come from zero-shot SigLIP over the 14 categories, never `other` or `cooking`. Accuracy is unmeasured; one test video, `Skate`, was labelled `longboarding`.

## Still to do

1. Generate the outputs: FPVSum on the Mac, EPIC on Colab (`--part` split across accounts; see above).
2. Confirm the organiser-assigned team ID, then run `python tools/set_team_id.py <ID>`.
3. Commit `outputs/*.json` if the repository itself is submitted. Upload `dist/<ID>.zip` if an archive is required.
