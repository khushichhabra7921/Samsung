from types import SimpleNamespace

import pytest

from handscribe_xr.contracts import ROOT
from handscribe_xr.track04 import build_submission_payload, load_manifest, resolve_video_id
from validate import check_payload, failures
from validate import load_classes as load_validator_classes
from validate import load_manifest as load_validator_manifest


def fake_analysis(*, duration=100.0, categories=()):
    action = SimpleNamespace(
        start_ms=12_400,
        end_ms=18_900,
        label=SimpleNamespace(verb="open", noun="fridge"),
        confidence=0.82,
    )
    highlight = SimpleNamespace(start_ms=20_000, end_ms=25_000)
    report = SimpleNamespace(duration_sec=duration, actions=[action], highlights=[highlight])
    return SimpleNamespace(report=report, video_categories=categories)


def classes():
    import json
    return json.loads((ROOT / "supported_classes.json").read_text(encoding="utf-8"))


def strict_failures(payload, stem):
    return failures(check_payload(
        payload, stem=stem, manifest=load_validator_manifest(ROOT / "manifest.csv"),
        classes=load_validator_classes(ROOT / "supported_classes.json"),
    ), strict=True)


def test_manifest_contains_official_187_videos():
    manifest = load_manifest()
    assert len(manifest) == 187
    assert sum(row["dataset"] == "epic_kitchens_100" for row in manifest.values()) == 138
    assert sum(row["dataset"] == "fpvsum" for row in manifest.values()) == 49


def test_epic_output_matches_six_field_contract():
    payload = build_submission_payload(
        fake_analysis(), team_id="T07", video_id="P02_109",
        dataset="epic_kitchens_100", manifest_duration=100.0,
        supported_classes=classes(),
    )
    assert set(payload) == {
        "team_id", "video_id", "duration_sec", "label", "actions", "highlights"
    }
    assert payload["label"] == {"name": "cooking", "score": 1.0}
    assert payload["actions"][0] == {
        "segment": [12.4, 18.9], "verb": "open", "noun": "fridge", "score": 0.82
    }
    assert payload["highlights"] == [{"segment": [20.0, 25.0]}]


def test_fpvsum_uses_video_level_category_scores():
    payload = build_submission_payload(
        fake_analysis(categories=(("ski", 0.91), ("other", 0.09))),
        team_id="T07", video_id="Ski_sample",
        dataset="fpvsum", manifest_duration=100.0,
        supported_classes=classes(),
    )
    assert payload["label"] == {"name": "ski", "score": 0.91}


def test_duration_mismatch_is_rejected():
    with pytest.raises(ValueError, match="manifest duration"):
        build_submission_payload(
            fake_analysis(duration=101.0), team_id="T07", video_id="P02_109",
            dataset="epic_kitchens_100", manifest_duration=100.0,
            supported_classes=classes(),
        )


def payload_for(analysis, *, duration=100.0, dataset="epic_kitchens_100", video_id="P02_109"):
    return build_submission_payload(
        analysis, team_id="T07", video_id=video_id, dataset=dataset,
        manifest_duration=duration, supported_classes=classes(),
    )


def spans(payload):
    return [end - start for start, end in (item["segment"] for item in payload["highlights"])]


def test_highlight_budget_is_enforced():
    analysis = fake_analysis()
    analysis.report.highlights = [
        SimpleNamespace(start_ms=0, end_ms=8000, score=0.4),
        SimpleNamespace(start_ms=10_000, end_ms=18_000, score=0.9),
    ]
    lengths = spans(payload_for(analysis))
    assert sum(lengths) <= 15.0
    assert all(length >= 4.0 for length in lengths)


def test_rounded_four_second_highlight_stays_at_least_four_seconds():
    # 0.02 -> 4.02 evaluates to 3.9999999999999996 s in floating point.
    analysis = fake_analysis()
    analysis.report.highlights = [SimpleNamespace(start_ms=20, end_ms=4020)]
    assert spans(payload_for(analysis)) == [pytest.approx(4.01)]
    assert min(spans(payload_for(analysis))) >= 4.0


def test_touching_highlights_are_merged():
    analysis = fake_analysis()
    analysis.report.highlights = [
        SimpleNamespace(start_ms=10_000, end_ms=14_000), SimpleNamespace(start_ms=14_000, end_ms=18_000),
    ]
    assert payload_for(analysis)["highlights"] == [{"segment": [10.0, 18.0]}]


@pytest.mark.parametrize("video_id,duration", [("P03_26", 11.13), ("P26_30", 15.28), ("P28_23", 25.16)])
def test_short_manifest_videos_get_one_minimum_length_highlight(video_id, duration):
    analysis = fake_analysis(duration=duration)
    analysis.report.actions = []
    analysis.report.highlights = [SimpleNamespace(start_ms=5_000, end_ms=6_600, score=0.5)]
    payload = payload_for(analysis, duration=duration, video_id=video_id)
    [(start, end)] = [item["segment"] for item in payload["highlights"]]
    assert end - start >= 4.0 and end - start < 4.02
    assert 0 <= start < 5.0 and 6.6 < end <= duration
    assert not strict_failures(payload, video_id)


def test_short_video_without_candidates_is_centred():
    analysis = fake_analysis(duration=11.13)
    analysis.report.actions, analysis.report.highlights = [], []
    [item] = payload_for(analysis, duration=11.13, video_id="P03_26")["highlights"]
    assert item["segment"][0] == pytest.approx(3.56) and item["segment"][1] - item["segment"][0] >= 4.0


def test_fpvsum_never_predicts_other_or_cooking():
    analysis = fake_analysis(categories=(("other", 0.5), ("cooking", 0.3), ("surfing", 0.2)))
    assert payload_for(analysis, dataset="fpvsum")["label"] == {"name": "surfing", "score": 0.2}


def test_missing_category_scores_still_write_a_valid_file():
    assert payload_for(fake_analysis(), dataset="fpvsum")["label"] == {"name": "other", "score": 0.0}


def test_video_outside_manifest_uses_predicted_label_and_decoded_duration():
    analysis = fake_analysis(duration=42.0, categories=(("cooking", 0.7), ("ski", 0.2)))
    payload = build_submission_payload(
        analysis, team_id="T07", video_id="clip", dataset=None,
        manifest_duration=None, supported_classes=classes(),
    )
    assert payload["label"] == {"name": "cooking", "score": 0.7}
    assert payload["duration_sec"] == 42.0


def test_video_id_comes_from_input_then_output_name(tmp_path):
    manifest = load_manifest()
    assert resolve_video_id(tmp_path / "P01_11.MP4", tmp_path / "out.json", manifest) == "P01_11"
    assert resolve_video_id(tmp_path / "video.mp4", tmp_path / "P01_11.json", manifest) == "P01_11"
    assert resolve_video_id(tmp_path / "video.mp4", tmp_path / "out.json", manifest) == "video"


def test_payload_passes_strict_validator():
    payload = payload_for(fake_analysis(duration=561.53), duration=561.53, video_id="P01_11")
    assert not strict_failures(payload, "P01_11")
