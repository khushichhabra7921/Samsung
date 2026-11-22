from types import SimpleNamespace

import pytest

from handscribe_xr.contracts import ROOT
from handscribe_xr.track04 import build_submission_payload, load_manifest


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


def test_highlight_budget_is_enforced():
    analysis = fake_analysis()
    analysis.report.highlights = [
        SimpleNamespace(start_ms=0, end_ms=8000),
        SimpleNamespace(start_ms=10_000, end_ms=18_000),
    ]
    with pytest.raises(ValueError, match="15%"):
        build_submission_payload(
            analysis, team_id="T07", video_id="P02_109",
            dataset="epic_kitchens_100", manifest_duration=100.0,
            supported_classes=classes(),
        )
