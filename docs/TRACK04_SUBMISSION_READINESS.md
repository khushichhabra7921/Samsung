# Track 04 submission readiness

Checked against the Student Package `TEAM_README.md` and the Track 04 submission slides (package layout, output JSON, evaluation).

## Requirements status

| Requirement | Status |
|---|---|
| `run.sh <video.mp4> <out.json>` | Done. Two arguments run the Track 04 exporter. The video ID comes from the input stem, then the output stem. Videos outside the manifest still produce output. The packaged `run.sh` has the team ID built in, so a re-run needs no environment variables. |
| One `<video_id>.json` per manifest video, six keys | Done in the exporter, but not yet generated. Use `scripts/generate_outputs.py` once the videos are downloaded. |
| `label` | EPIC videos get `cooking`. FPVSum videos get a SigLIP zero-shot pick over the 14 FPVSum categories, never `other` or `cooking`. The prompts use plain phrases such as "rock climbing" and "bike polo". |
| `actions` (Action mAP) | Verb–noun segments from MediaPipe, OWLv2 and SigLIP, limited to the official vocabulary. |
| `highlights` (Action coverage, Highlight F1) | Chronological and non-overlapping, with touching segments merged. Each is at least 4.0 s after float rounding, and the total stays within the 15% budget. |
| Videos shorter than 26.67 s (`P03_26`, `P26_30`, `P28_23`) | Resolved. Each gets one 4.0 s highlight, so the file stays valid. The ≥ 4.0 s rule is a hard check. A budget overrun is only a warning with a soft penalty. |
| `validate.py` | Team-written stand-in at the repository root. It implements every documented check and adds `--strict`. The packager runs it before building. |
| `code/`, `weights/`, `requirements.txt`, `README.md` | Done by `scripts/build_submission_package.py`. Requirements are fully pinned, including torch. The README covers setup, model architecture and data used, built from `docs/SUBMISSION_README.md`. |
| No hardcoding | The code contains no video IDs or per-video outputs. `manifest.csv` is used only for the dataset type and the duration check. No ground truth is used. |

## What is left for the team

1. **Team ID.** Confirm the organiser-assigned ID. The Student Package sample uses `T07`, but that is not proof.
2. **Download the 187 videos** (about 184 GB for EPIC-KITCHENS) and the model weights. See `README.md` → *Local semantic assets*.
3. **Generate the outputs:** `python scripts/generate_outputs.py --videos <download folder>`. Re-run it until it reports 0 failed and 0 missing.
4. **Validate:** `python validate.py --dir outputs/ --strict -v`. If the official `validate.py` arrives, run it as well.
5. **Package:** `python scripts/build_submission_package.py --team-id <ID>`, then upload `<ID>.zip`.
6. Add the presentation and demo video.

## Open questions for the organisers

- `examples/skiing_03.json` uses `skiing`, but `supported_classes.json` lists `ski`. We emit `ski` as the class list requires.
- The slides show a simplified schema (`label` as a string, `start`/`end` keys, highlight `importance`). The TEAM_README and both example files use the six-key format with `segment` arrays. We follow the TEAM_README, which matches the slide's "all six required schema keys".
- The TEAM_README says `validate.py` only *warns* about the 15% budget, while the slide says "strict adherence". We stay within budget everywhere it is achievable. Only the three sub-26.67 s videos exceed it, because they must.
