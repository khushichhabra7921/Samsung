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


## Student kit received (2026-11-22)

The uploaded ZIP was inspected. It contains:
- `manifest.csv` (187 rows: 138 EPIC-KITCHENS-100, 49 FPVSum)
- `supported_classes.json` (16 video labels, 98 verbs, 301 nouns)
- `TEAM_README.md`
- two sample JSONs: `P02_109.json` and `skiing_03.json`

**The uploaded ZIP does not contain `validate.py` or an `examples/` directory**, despite the TEAM_README describing both. The two sample JSONs are at the ZIP root. Obtain the actual validator from the organiser; do not substitute a guessed script and call it official.

The Track 04 adapter was added at `handscribe_xr/track04.py`. It accepts the required two positional arguments, loads the manifest, runs semantic inference, and exports the six required top-level fields. Root `run.sh` now routes exactly two positional arguments through this adapter while retaining the previous development CLI for other invocations.

Set these environment variables before invoking it:
```sh
export TEAM_ID="<exact organiser-assigned team ID>"
export TRACK04_KIT_DIR="/path/to/Student package_track04"
./run.sh P02_109.mp4 outputs/P02_109.json
```

On Windows PowerShell:
```powershell
$env:TEAM_ID = "<exact organiser-assigned team ID>"
$env:TRACK04_KIT_DIR = "D:\path\to\Student package_track04"
.\run.sh P02_109.mp4 outputs/P02_109.json
```

### Known blockers — do not call this fully submission-ready yet

1. **FPVSum category classifier is missing.** The current HandScribe model predicts verb/noun actions, not one of the 14 FPVSum video categories. The adapter currently emits the schema-valid `other` label with score `0.0` for FPVSum as an explicit fallback. This is not a meaningful prediction and will score poorly on Label Accuracy (10%). A trained/integrated video-category classifier is required.
2. **Short-video highlight constraint conflict.** The manifest includes EPIC clips such as `P26_30` (15.28 s) and `P28_23` (25.16 s). Their 15% budgets are below 4 seconds, while TEAM_README requires at least one highlight of at least 4 seconds. Both conditions cannot be met simultaneously. Ask the organisers which rule takes precedence for these videos.
3. **Team ID is not confirmed.** `T07` appears in sample JSONs, but sample files are examples and do not establish this team's assigned ID. The adapter therefore requires the actual ID via `TEAM_ID`.
4. **Official validator absent from upload.** The adapter has not been verified against the organiser's `validate.py`; do not claim official validation until that exact file is obtained and run.

The adapter intentionally errors instead of writing an output when it cannot produce a highlight meeting the stated 4-second minimum. This prevents silently generating a file known to violate the documented rule.
