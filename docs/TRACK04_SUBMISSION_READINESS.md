# Track 04 submission readiness

Team: **404NotFound**  
Track: **04 — Action-Aware Video Summarization**  
Repository: https://github.com/khushichhabra7921/Samsung

## Current state

The root-level HandScribe XR implementation is the active Track 04 application. Its CLI is:

```sh
./run.sh INPUT_VIDEO OUTPUT_REPORT.json --mode semantic \
  --weights weights --vocabulary path/to/track04-labels.json \
  --highlight outputs/highlights.mp4
```

The command-line entry point is `python -m handscribe_xr summarize VIDEO OUTPUT`. The current output is the project's development contract, `handscribe.xr/v1`, not the organiser's final submission schema.

The current development report uses:
- `schema_version`, `team_id`, `video_id`
- millisecond duration/budget fields
- `highlights` with `start_ms`, `end_ms`, `score`, and labels
- `actions` with nested `label`, confidence, verification and evidence
- optional `video_label` and diagnostics

This is structurally different from the submission example in the Track 04 slides, which expects `video_id`, `label`, `actions[{verb,noun,start,end}]`, and `highlights[{start,end,importance}]`. Do not rename fields mechanically without checking the official kit's exact schema and validator.

## Required organiser files — not yet available in this repository

Obtain these exact files from the organisers before implementing the final adapter:
- `manifest.csv`
- `supported_classes.json`
- `validate.py`
- `TEAM_README.md`
- `examples/` containing sample output JSONs

The official manifest determines which videos and IDs are expected. The official class file determines allowed labels. The validator and examples determine required fields, types, ranges, and constraints. No guessed schema or fabricated sample predictions should be submitted.

## Final package target

After receiving the official kit, create the team-ID package with:
- `run.sh` — one documented inference command matching the organiser's invocation exactly
- `code/` — all source needed for inference and official-schema export
- `weights/` — all required local model assets, subject to organiser size/licensing rules
- `requirements.txt` — pinned runtime dependencies
- `outputs/` — exactly one official-schema JSON per required video
- `README.md` — reproducible setup, execution, model/data provenance and limitations

The required `TEAM_ID` is the organiser-assigned identifier, not the team name `404NotFound`. Replace the package directory name only after confirming that identifier.

## Safe pre-submission checks

1. Run the repository test suite: `.venv/bin/python -m pip install -r requirements-dev.txt && .venv/bin/python -m pytest`.
2. Run semantic inference using the official vocabulary once received.
3. Confirm all model assets are local and that offline execution succeeds.
4. Implement an explicit adapter from `handscribe.xr/v1` to the official schema.
5. Generate one JSON per manifest video using the manifest's exact video IDs.
6. Run the organiser's `validate.py --dir outputs/` and retain its complete report.
7. Do not report a validation pass count unless it comes from the exact organiser-provided validator and the same output directory.

## Presentation claims to reconcile

The submitted deck currently says **142/187 outputs pass validate.py** (slide 6), while the root README describes the present JSON as a development format awaiting the official student-kit schema, manifest, class list and validator. Re-run the stated count with the exact official validator or revise the slide to identify which validator produced it.

The deck also describes approximately **35 FPS** as measured in BIOS V2 and explicitly marks Samsung re-measurement as `[FILL]` (slide 7). Do not attribute that speed to this Samsung pipeline without a fresh benchmark.

The slide 3 demo command should be shown as the real repository command above, or the CLI should be deliberately adapted and tested to match the organisers' required invocation.

## Current limitations

This branch intentionally does not add:
- a guessed official JSON schema or label vocabulary
- fake `outputs/*.json` predictions
- placeholder model weight files
- an untested replacement `run.sh`
- claims of official validator compliance

These require the organiser student kit and a real inference/validation run.
