import copy
import json

import pytest

from handscribe_xr.contracts import ROOT
from validate import check_directory, check_file, check_payload, failures, load_classes, load_manifest, main

MANIFEST = load_manifest(ROOT / "manifest.csv")
CLASSES = load_classes(ROOT / "supported_classes.json")
VALID = {
    "team_id": "T07",
    "video_id": "P01_12",
    "duration_sec": 173.81,
    "label": {"name": "cooking", "score": 1.0},
    "actions": [{"segment": [12.4, 18.9], "verb": "open", "noun": "fridge", "score": 0.82}],
    "highlights": [{"segment": [10.0, 18.0]}, {"segment": [40.0, 52.0]}],
}


def errors(data, strict=False):
    return failures(check_payload(data, stem=data.get("video_id"), manifest=MANIFEST, classes=CLASSES), strict)


def edited(**changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


@pytest.mark.parametrize("name", ["P02_109.json", "skiing_03.json"])
def test_organiser_examples_pass_default_checks(name):
    _, issues = check_file(ROOT / "examples" / name, MANIFEST, CLASSES)
    assert failures(issues, strict=False) == []


def test_strict_mode_flags_sample_label_outside_class_list():
    _, issues = check_file(ROOT / "examples" / "skiing_03.json", MANIFEST, CLASSES)
    assert any("'skiing' is not in supported_classes" in message for message in failures(issues, strict=True))


def test_valid_payload_passes_strict_checks():
    assert errors(VALID, strict=True) == []


@pytest.mark.parametrize("data,expected", [
    ({k: v for k, v in VALID.items() if k != "label"}, "missing required key 'label'"),
    (edited(team_id=""), "'team_id' must be a non-empty string"),
    (edited(duration_sec=True), "'duration_sec' must be a finite number"),
    (edited(duration_sec=180.0), "differs from manifest"),
    (edited(label={"name": "cooking", "score": 1.2}), "label: 'score'"),
    (edited(actions=[{"segment": [9.0, 3.0], "verb": "open", "noun": "fridge", "score": 0.5}]), "violates"),
    (edited(actions=[{"segment": [1.0, 3.0], "verb": "", "noun": "fridge", "score": 0.5}]), "'verb'"),
    (edited(highlights=[]), "non-empty array"),
    (edited(highlights=[{"segment": [40.0, 52.0]}, {"segment": [10.0, 18.0]}]), "chronological"),
    (edited(highlights=[{"segment": [10.0, 18.0]}, {"segment": [17.0, 22.0]}]), "overlap"),
    (edited(highlights=[{"segment": [10.0, 13.0]}, {"segment": [20.0, 23.9]}]), "at least 4.0 s"),
    (edited(highlights=[{"segment": [0.02, 4.02]}]), "at least 4.0 s"),
    (edited(highlights=[{"segment": [170.0, 174.5]}]), "violates"),
])
def test_invalid_payloads_fail(data, expected):
    assert any(expected in message for message in errors(data)), errors(data)


def test_budget_overrun_is_strict_only_when_avoidable():
    over = edited(highlights=[{"segment": [0.0, 30.0]}])
    assert errors(over) == []
    assert any("15% budget" in message for message in errors(over, strict=True))
    short = edited(video_id="P03_26", duration_sec=11.13, actions=[], highlights=[{"segment": [3.0, 7.0]}])
    assert errors(short, strict=True) == []


def test_touching_highlights_only_fail_in_strict_mode():
    touching = edited(highlights=[{"segment": [10.0, 14.0]}, {"segment": [14.0, 18.0]}])
    assert errors(touching) == []
    assert any("touch" in message for message in errors(touching, strict=True))


def test_nan_is_rejected(tmp_path):
    path = tmp_path / "P01_12.json"
    path.write_text(json.dumps(VALID).replace("0.82", "NaN"), encoding="utf-8")
    _, issues = check_file(path, MANIFEST, CLASSES)
    assert any("NaN" in message for message in failures(issues, strict=False))


def test_directory_checks_coverage_and_team_id(tmp_path):
    (tmp_path / "P01_12.json").write_text(json.dumps(VALID), encoding="utf-8")
    other = edited(video_id="P01_13", duration_sec=93.76, team_id="T08")
    (tmp_path / "P01_13.json").write_text(json.dumps(other), encoding="utf-8")
    directory = failures(check_directory(tmp_path, MANIFEST, CLASSES)[""], strict=False)
    assert any("team_id differs" in message for message in directory)
    assert any("185 manifest video(s) have no output JSON" in message for message in directory)


def test_cli_exit_codes(tmp_path, capsys):
    good = tmp_path / "P01_12.json"
    good.write_text(json.dumps(VALID), encoding="utf-8")
    assert main(["--file", str(good), "-v"]) == 0
    bad = tmp_path / "P01_13.json"
    bad.write_text(json.dumps(edited(video_id="P01_13", highlights=[])), encoding="utf-8")
    assert main(["--file", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out
