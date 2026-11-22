import numpy as np
import pytest

from egovision.config import ObjectsConfig
from egovision.schema import ObjectDet
from handscribe_xr.contracts import ROOT, Label
from handscribe_xr.recognition import Predictions, SemanticModels, Vocabulary, load_verb_prior
from handscribe_xr.selection import Sample

VOCAB = Vocabulary.load(ROOT / "supported_classes.json")


def recognizer():
    model = SemanticModels.__new__(SemanticModels)
    model.vocab = VOCAB
    model.objects = ObjectsConfig(contact_margin=0)
    model.verb_prior = load_verb_prior(VOCAB)
    model.real_nouns = np.array([noun != "other" for noun in VOCAB.nouns])
    return model


def prediction(noun: str, verb_scores: dict[str, float], base: float = 0.01) -> Predictions:
    verbs = np.full(len(VOCAB.verbs), base)
    for verb, score in verb_scores.items():
        verbs[VOCAB.verbs.index(verb)] = score
    nouns = np.full(len(VOCAB.nouns), 0.02)
    nouns[VOCAB.nouns.index(noun)] = 0.6
    nouns[VOCAB.nouns.index("other")] = 0.9  # 'other' must never be chosen
    box = ObjectDet(label=noun, kind="portable", score=0.8, bbox=[0, 0, 10, 10])
    return Predictions(np.array([1.0]), verbs, nouns, [box])


def test_train_prior_is_a_distribution_without_other():
    prior = np.exp(load_verb_prior(VOCAB))
    assert prior.shape == (len(VOCAB.nouns), len(VOCAB.verbs))
    assert np.allclose(prior.sum(axis=1), 1.0)
    assert prior[:, VOCAB.verbs.index("other")].max() == 0.0
    tap = prior[VOCAB.nouns.index("tap")]
    assert {VOCAB.verbs[i] for i in np.argsort(-tap)[:2]} == {"turn-on", "turn-off"}


def test_scene_bias_verb_is_removed_and_never_other():
    model = recognizer()
    frames = [prediction("tap", {"cook": 0.18, "eat": 0.05}) for _ in range(5)]
    frames.append(prediction("tap", {"cook": 0.18, "eat": 0.05, "turn-on": 0.03}))
    offset = model.verb_offsets(frames)
    sample = Sample(0, np.array([1.0]), 0.5)
    model.label(sample, frames[-1], [], offset)
    assert sample.label == Label(verb="turn-on", noun="tap")
    assert 0 < sample.visual_score <= 1


def test_weak_object_evidence_leaves_frame_unlabelled():
    model = recognizer()
    weak = prediction("tap", {})
    weak.noun_scores[:] = 0.01
    weak.boxes.clear()
    sample = Sample(0, np.array([1.0]), 0.5)
    model.label(sample, weak, [], None)
    assert sample.label is None


def test_prior_requires_official_class_keys():
    assert load_verb_prior(Vocabulary(("cut", "rinse"), ("pepper", "plate"))) is None


@pytest.mark.parametrize("noun", ["fridge", "onion", "knife"])
def test_label_without_visual_verb_evidence_follows_prior(noun):
    model = recognizer()
    frame = prediction(noun, {})
    sample = Sample(0, np.array([1.0]), 0.5)
    model.label(sample, frame, [], model.verb_offsets([frame]))
    top = VOCAB.verbs[int(np.argmax(model.verb_prior[VOCAB.nouns.index(noun)]))]
    assert sample.label == Label(verb=top, noun=noun)
