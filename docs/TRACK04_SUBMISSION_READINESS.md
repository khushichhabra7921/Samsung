# Track 04 submission readiness

## Implemented on main

- Added the exact 187-row Student Package manifest (138 EPIC-KITCHENS-100 and 49 FPVSum IDs).
- Added the official supported class vocabulary (16 video labels, 98 verbs, 301 nouns).
- Added the two supplied example JSON files under examples/.
- Added a separate Track 04 exporter that writes the six required top-level fields without changing the existing handscribe.xr/v1 report contract.
- Added zero-shot SigLIP video-category scores for the FPVSum label field.
- Updated run.sh so exactly two arguments invoke the submission exporter; legacy calls with flags or more arguments continue to invoke the existing development CLI.
- Added tests for the submission payload, manifest counts, duration check, and highlight budget.
- Added a packager that preflights the 187 JSON files and builds the TEAM_ID directory/archive.

## Run one video

Use the exact video ID as the input filename stem and output filename stem:

```sh
export HANDSCRIBE_TEAM_ID=<organiser-assigned-team-id>
./run.sh /path/to/P02_109.mp4 outputs/P02_109.json
```

Semantic inference requires local model assets under weights/:
- hand_landmarker.task
- owlv2/
- siglip/

The model weights are not committed to Git because of their size. The exporter expects the root supported_classes.json and manifest.csv.

## Team ID

The sample JSON in the Student Package uses T07, but that sample alone does not prove the team's assigned ID. The exporter therefore requires HANDSCRIBE_TEAM_ID instead of silently hardcoding a possibly incorrect ID.

## Validation status

The uploaded Student Package ZIP includes manifest.csv, supported_classes.json, TEAM_README.md, and two JSON examples, but does not include validate.py despite the guide listing it. The supervisor has been contacted. The packager's own preflight is not a replacement for the organiser validator.

## Contradictory highlight constraints

The guide requires total highlight duration <=15% and at least one highlight >=4.0s. The manifest includes three videos for which those conditions cannot both be true:

| Video ID | Duration | 15% budget |
|---|---:|---:|
| P03_26 | 11.13s | 1.67s |
| P26_30 | 15.28s | 2.29s |
| P28_23 | 25.16s | 3.77s |

The exporter deliberately stops for these videos rather than writing a knowingly invalid file. Ask the organisers how they want these manifest entries handled.

## Category-label sample discrepancy

The supplied supported_classes.json lists the FPVSum class as ski, while the supplied skiing_03.json sample uses skiing. The implementation follows the supported class dictionary and emits ski. Confirm this discrepancy with the organisers or the returned validator before final output generation.

## Not yet complete

- The 187 source videos are not included in the Student Package ZIP and must be downloaded separately.
- Local OWLv2, SigLIP, and MediaPipe weights must be installed.
- Outputs for all 187 videos have not been generated.
- The official validate.py is pending from the supervisor.
- FPVSum zero-shot category scores and action confidence thresholds still need evaluation on labelled data.
- The final archive cannot be produced until the team ID, weights, all 187 outputs, and organiser clarification for the three short videos are available.
