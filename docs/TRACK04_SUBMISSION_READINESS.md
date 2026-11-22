# Track 04 submission readiness

Checked against the Student Package `TEAM_README.md` and the Track 04 submission slides (package layout, output JSON, evaluation).

## Requirements status

| Requirement | Status |
|---|---|
| `run.sh <video.mp4> <out.json>` | Done. Two arguments run the Track 04 exporter. The video ID comes from the input stem, then the output stem. Videos outside the manifest still produce output. The packaged `run.sh` has the team ID built in, so a re-run needs no environment variables. |
| One `<video_id>.json` per manifest video, six keys | Done in the exporter, but not yet generated. Use `scripts/generate_outputs.py` once the videos are downloaded. |
| `label` | EPIC videos get `cooking`. FPVSum videos get a SigLIP zero-shot pick over the 14 FPVSum categories, never `other` or `cooking`. The prompts use plain phrases such as "rock climbing" and "bike polo". |
| `actions` (Action mAP) | Verb–noun segments from MediaPipe, OWLv2 and SigLIP. Verbs are chosen from per-video-centred SigLIP scores combined with an EPIC-100 **train-split** P(verb \| noun) prior, and are never `other`. See *Verb fix* below. |
| `highlights` (Action coverage, Highlight F1) | Chronological and non-overlapping, with touching segments merged. Each is at least 4.0 s after float rounding, and the total stays within the 15% budget. |
| Videos shorter than 26.67 s (`P03_26`, `P26_30`, `P28_23`) | Resolved. Each gets one 4.0 s highlight, so the file stays valid. The ≥ 4.0 s rule is a hard check. A budget overrun is only a warning with a soft penalty. |
| `validate.py` | Team-written stand-in at the repository root. It implements every documented check and adds `--strict`. The packager runs it before building. |
| `code/`, `weights/`, `requirements.txt`, `README.md` | Done by `scripts/build_submission_package.py`. Requirements are fully pinned, including torch. The README covers setup, model architecture and data used, built from `docs/SUBMISSION_README.md`. |
| No hardcoding | The code contains no video IDs or per-video outputs. `manifest.csv` is used only for the dataset type and the duration check. No ground truth is used. |

## What is left for the team

1. **Team ID.** Confirm the organiser-assigned ID. The Student Package sample uses `T07`, but that is not proof.
2. **Download the 187 videos** with `python scripts/download_videos.py --dest <folder>` (about 180 GB for EPIC plus 2.5 GB for FPVSum, so use a machine or drive with around 200 GB free). Also get the model weights; see `README.md` → *Local semantic assets*.
3. **Generate the outputs:** `python scripts/generate_outputs.py --videos <download folder>`. Re-run it until it reports 0 failed and 0 missing.
4. **Validate:** `python validate.py --dir outputs/ --strict -v`. If the official `validate.py` arrives, run it as well.
5. **Package:** `python scripts/build_submission_package.py --team-id <ID>`, then upload `<ID>.zip`.
6. Add the presentation and demo video.

## Verb fix (Action mAP)

Before the fix, SigLIP's single-frame verb scores were almost all below the 0.12 cut-off, and `cook` came top on nearly every kitchen frame. As a result about 80% of actions were written with verb `other`. Action mAP was **0**, because EPIC labels never use `other`. The fix has three parts:
1. Verb log-odds are centred per video, which removes scene-wide bias.
2. They are combined with P(verb | noun) from the public EPIC-100 **train** split (`scripts/build_action_prior.py`). All 138 manifest videos are validation videos, and the script refuses to run if any manifest video is in its input.
3. The five most likely verbs are written per segment.

Checked with `scripts/eval_epic_train.py` on three train-split videos (P02_08, P04_103, P30_112; 121 ground-truth actions). The evaluator refuses non-train videos.

| Version | Action mAP (tIoU 0.1–0.5) | Verb mAP | Action coverage |
|---|---:|---:|---:|
| Before | 0.00 | 0.00 | 0.217 |
| Prior + centring | 1.41 | 0.84 | 0.276 |
| + 5 verb hypotheses (submitted) | **2.83** | 2.81 | **0.276** |

Absolute scores stay low, as expected for a zero-shot system without a trained action model. Three videos are a small sample, so treat these as a direction, not a precise estimate.

## Open questions for the organisers

- `examples/skiing_03.json` uses `skiing`, but `supported_classes.json` lists `ski`. We emit `ski` as the class list requires.
- The slides show a simplified schema (`label` as a string, `start`/`end` keys, highlight `importance`). The TEAM_README and both example files use the six-key format with `segment` arrays. We follow the TEAM_README, which matches the slide's "all six required schema keys".
- The TEAM_README says `validate.py` only *warns* about the 15% budget, while the slide says "strict adherence". We stay within budget everywhere it is achievable. Only the three sub-26.67 s videos exceed it, because they must.
