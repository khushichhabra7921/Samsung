from dataclasses import replace

import numpy as np
import pytest
from egovision.hands.postprocess import bridge_dropouts
from egovision.schema import FrameHands, Hand, ObjectDet

from handscribe_xr.contracts import Diagnostics, Highlight, Label, Report, budget_ms, write_json
from handscribe_xr.recognition import Predictions, SemanticModels, Vocabulary, load_verb_prior
from handscribe_xr.selection import Candidate, Sample, Timeline, select


def test_selector_rewards_distinct_actions_and_avoids_overlap():
    cut = Label(verb="cut", noun="pepper")
    wash = Label(verb="wash", noun="plate")
    candidates = [
        Candidate(Highlight(start_ms=0, end_ms=4000, score=0.96, labels=(cut,)), np.array([1.0, 0.0])),
        Candidate(Highlight(start_ms=2000, end_ms=6000, score=0.90, labels=(cut,)), np.array([1.0, 0.0])),
        Candidate(Highlight(start_ms=4000, end_ms=8000, score=0.94, labels=(cut,)), np.array([1.0, 0.0])),
        Candidate(Highlight(start_ms=10000, end_ms=14000, score=0.60, labels=(wash,)), np.array([0.0, 1.0])),
    ]
    selected = select(candidates, 8000)
    assert selected == (candidates[0].clip, candidates[-1].clip)
    assert select(list(reversed(candidates)), 8000) == selected
    assert select(candidates, 3999) == ()


def test_missing_hand_evidence_is_renormalized_and_conflict_penalizes():
    label = Label(verb="cut", noun="pepper")
    samples = [Sample(t, np.ones(2), 0.5, label=label, visual_score=0.8) for t in range(0, 1650, 330)]
    clean, = Timeline(samples, 1650).actions()
    assert clean.evidence.contact is None
    assert clean.confidence == pytest.approx((0.55 * 0.8 + 0.15) / 0.70, abs=0.0001)
    assert not clean.verified

    matching_samples = [replace(sample, contact=0.8) for sample in samples]
    supported, = Timeline(matching_samples, 1650).actions()
    contradictory, = Timeline([replace(sample, conflict=0.9) for sample in matching_samples], 1650).actions()
    assert supported.verified
    assert contradictory.confidence < supported.confidence
    assert not contradictory.verified


def test_action_boundaries_snap_to_hand_motion_valleys():
    label = Label(verb="wash", noun="plate")
    samples = [
        Sample(i * 330, np.ones(2), 0.4, label=label if 3 <= i < 7 else None,
               visual_score=0.8, hand_motion=motion)
        for i, motion in enumerate((0.9, 0.6, 0.05, 0.7, 0.5, 0.8, 0.6, 0.7, 0.05, 0.6))
    ]
    action, = Timeline(samples, 3300).actions()
    assert (action.start_ms, action.end_ms) == (660, 2640)
    assert action.evidence.temporal == pytest.approx(4 / 6)


def test_short_same_id_dropout_is_interpolated():
    hand = Hand(id=1, side="left", score=0.9, bbox=[0, 0, 10, 10], kp2d=[[0, 0]] * 21)
    moved = hand.model_copy(update={"kp2d": [[4, 8]] * 21})
    frames = [
        FrameHands(frame=0, t=0, hands=[hand]),
        FrameHands(frame=2, t=0.1),
        FrameHands(frame=4, t=0.2, hands=[moved]),
    ]
    recovered, = bridge_dropouts(frames, 6)[1].hands
    assert recovered.interp and recovered.id == 1
    assert recovered.kp2d[0] == [2, 4]
    assert not bridge_dropouts(frames, 3)[1].hands
    assert not frames[1].hands


def test_recognizer_uses_labels_and_contact_with_real_landmark_geometry():
    from egovision.config import ObjectsConfig

    recognizer = SemanticModels.__new__(SemanticModels)
    recognizer.vocab = Vocabulary(("cut", "rinse"), ("pepper", "plate"))
    recognizer.objects = ObjectsConfig(contact_margin=0)
    recognizer.verb_prior = load_verb_prior(recognizer.vocab)
    assert recognizer.verb_prior is None  # custom vocabularies keep the threshold path
    prediction = Predictions(
        np.array([1]), np.array([0.70, 0.2]), np.array([0.51, 0.59]),
        [ObjectDet(label="pepper", kind="portable", score=0.81, bbox=[0, 0, 10, 10])],
    )
    hand = Hand(id=1, side="left", score=0.9, bbox=[0, 0, 10, 10], kp2d=[[5, 5]] * 21)
    observed = Sample(0, np.array([1]), 0.5)
    recognizer.label(observed, prediction, [hand])
    assert observed.label == Label(verb="cut", noun="pepper")
    assert observed.contact == pytest.approx(0.9)
    hidden = Sample(0, np.array([1]), 0.5)
    recognizer.label(hidden, prediction, [])
    assert hidden.label is not None
    assert hidden.contact is None and hidden.conflict is None


def test_report_rejects_overlaps_overruns_and_nonfinite_scores(tmp_path):
    base = {
        "video_id": "fixture",
        "duration_ms": 60_000,
        "duration_sec": 60,
        "budget_ms": 9000,
        "mode": "preview",
        "device": "cpu",
        "status": "complete",
        "actions": (),
        "diagnostics": Diagnostics(sampled_frames=10, keyframes=5, blur_pct=0, covered_pct=0),
    }
    first = Highlight(start_ms=0, end_ms=4000, score=0.7)
    second = Highlight(start_ms=2000, end_ms=6000, score=0.6)
    with pytest.raises(ValueError, match="non-overlapping"):
        Report(**base, highlights=(first, second), highlight_duration_ms=8000)
    with pytest.raises(ValueError, match="15% budget"):
        Report(**base, highlights=(Highlight(start_ms=0, end_ms=9001, score=0.7),), highlight_duration_ms=9001)
    assert budget_ms(26_666) == 3999
    assert budget_ms(26_667) == 4000

    target = tmp_path / "report.json"
    target.write_text("existing")
    with pytest.raises(ValueError):
        write_json(target, {"score": float("nan")})
    assert target.read_text() == "existing"
