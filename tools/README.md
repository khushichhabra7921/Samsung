# Team tools (not part of the uploaded package)

These scripts take the repository from "no videos" to `TEAM_ID.zip`. Run every command from the repository root. Install the tool dependencies with `python -m pip install -r tools/requirements.txt`.

## Workflow

| Step | Command | Notes |
|---|---|---|
| 1. Download the 187 videos | `python tools/download_videos.py --dest /data/track04` | EPIC: 138 videos, about 180 GB, from the official `data.bris.ac.uk` endpoint. Downloads are MD5-checked and resume after interruptions. FPVSum: 49 videos, about 2.5 GB, from YouTube via `yt-dlp`. These use the official `FPVSum_videolist.csv`, because names like `BikePolo_5yf797Lh4Oc_1` map to a different YouTube ID. Durations are checked against the manifest. Use a machine or drive with about 200 GB free. `--dataset fpvsum`, `--ids A,B` and `--list` are also available. |
| 2. Generate outputs | `HANDSCRIBE_TEAM_ID=<ID> python tools/generate_outputs.py --videos /data/track04` | Finds `<video_id>.mp4` anywhere under the folder, in any letter case. Loads the models once and skips outputs that already pass validation, so you can re-run it until it reports 0 failed and 0 missing. |
| 3. Validate | `python tools/validate.py --dir outputs/ --strict -v` | Without `--strict`, it follows the student-kit README: budget overruns and unknown classes only warn. `--strict` fails them too. |
| 4. Package | `python tools/build_package.py --team-id <ID>` | Validates strictly, checks `weights/`, and writes `dist/<ID>/` and `dist/<ID>.zip`. The packaged `run.sh` has `DEFAULT_TEAM_ID` filled in. |

Before submitting, also put the confirmed ID into the `DEFAULT_TEAM_ID=""` line of the root `run.sh`.

## Checks

```bash
python -m pytest -q tools/tests          # 47 tests, no model weights needed
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

1. Confirm the organiser-assigned team ID, then fill `DEFAULT_TEAM_ID` in `run.sh`.
2. Steps 1–4 above, on a machine with about 200 GB free.
3. Commit `outputs/*.json` if the repository itself is submitted. Upload `dist/<ID>.zip` if an archive is required.
