# Samsung PRISM Hackathon 2026 — Track 04 Student Package Notes

The official manifest and class vocabulary supplied by the organisers are mirrored in this repository as manifest.csv and supported_classes.json. There are 187 target videos: 138 EPIC-KITCHENS-100 and 49 FPVSum.

## Required prediction format

Generate one JSON file named VIDEO_ID.json per manifest video. Every file must contain exactly these six top-level fields:

- team_id: the same non-empty organiser-assigned ID in every file
- video_id: exact manifest ID and matching filename stem
- duration_sec: positive duration, within 0.5 seconds of the manifest
- label: object with name and score in [0,1]
- actions: array of at most 2,000 objects with segment [start,end], verb, noun, score
- highlights: chronological, non-overlapping objects with segment [start,end]

Timestamps are seconds. Action timestamps must be within the video (allowing the documented 0.5 second tolerance). Highlight total duration must not exceed 15% of the video and at least one highlight must be at least 4.0 seconds.

Use cooking for EPIC-KITCHENS-100 labels. FPVSum labels are selected from the supplied class dictionary.

## Inference

Set HANDSCRIBE_TEAM_ID to the exact ID supplied by the organisers. Semantic inference requires local model assets under weights/ and the official vocabulary:

```sh
export HANDSCRIBE_TEAM_ID=<assigned-id>
./run.sh /path/to/VIDEO_ID.mp4 outputs/VIDEO_ID.json
```

## Validation

The Student Package guide specifies:

```sh
python validate.py --dir outputs/
```

The ZIP provided to the team did not contain validate.py, although the guide lists it. Use the organiser-provided validator when the supervisor supplies it. The repository's packaging preflight is not a substitute for that validator.

## Important edge case

Three manifest durations are shorter than 26.6667 seconds, so their 15% budget is shorter than the required 4.0-second minimum highlight. The two constraints cannot both be met for those videos. Do not invent an exception; request written organiser clarification.
