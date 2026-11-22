"""Optional local OWLv2 and SigLIP recognition using the BIOS model adapters."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from egovision.config import ObjectsConfig, VocabItem
from egovision.objects.contact import associate
from egovision.objects.detector import OpenVocabDetector
from egovision.schema import Hand, ObjectDet
from egovision.steps.features import SiglipEmbedder

from .contracts import Label
from .selection import Sample, blend


class ModelSetupError(RuntimeError):
    """Local model assets or dependencies are unavailable."""


PRIOR_PATH = Path(__file__).resolve().parent / "assets" / "epic100_train_prior.json"
PRIOR_STRENGTH = 5.0  # Dirichlet pseudo-counts pulling rare nouns toward the overall verb frequency
VERB_TEMPERATURE = 1.0
MIN_NOUN_SCORE = 0.12


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def load_verb_prior(vocab: Vocabulary, path: Path = PRIOR_PATH) -> np.ndarray | None:
    """log P(verb | noun) as a (nouns x verbs) matrix in vocabulary order.

    Built by scripts/build_action_prior.py from EPIC-KITCHENS-100 train-split counts. Returns
    None for a vocabulary that does not contain the EPIC class keys. 'other' is never chosen
    as a verb because it cannot match EPIC ground truth.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    verb_index = {key: i for i, key in enumerate(vocab.verbs)}
    noun_index = {key: i for i, key in enumerate(vocab.nouns)}
    if not set(data["verbs"]) <= verb_index.keys() or not set(data["nouns"]) <= noun_index.keys():
        return None
    counts = np.zeros((len(vocab.nouns), len(vocab.verbs)))
    for verb, noun, count in data["pair_counts"]:
        counts[noun_index[data["nouns"][noun]], verb_index[data["verbs"][verb]]] += count
    known = np.zeros(len(vocab.verbs), dtype=bool)
    known[[verb_index[verb] for verb in data["verbs"]]] = True
    marginal = counts.sum(axis=0) + known
    marginal /= marginal.sum()
    prior = (counts + PRIOR_STRENGTH * marginal) / (counts.sum(axis=1, keepdims=True) + PRIOR_STRENGTH)
    with np.errstate(divide="ignore"):
        return np.where(known, np.log(prior), -np.inf)


# Plain-language phrases for the official video labels. Compound class keys such as
# "rockclimbing" or "bikepolo" otherwise reach the SigLIP text encoder as unfamiliar words.
CATEGORY_PHRASES = {
    "bike": "riding a bicycle",
    "bikepolo": "playing bike polo",
    "boxing": "boxing",
    "cooking": "cooking in a kitchen",
    "horse": "riding a horse",
    "jumping": "jumping",
    "longboarding": "riding a longboard",
    "motor": "riding a motorcycle",
    "parkour": "doing parkour",
    "plane": "flying in a plane",
    "rockclimbing": "rock climbing",
    "scuba": "scuba diving underwater",
    "skate": "skating",
    "ski": "skiing",
    "surfing": "surfing",
    "other": "an everyday activity",
}


@dataclass(frozen=True)
class Vocabulary:
    verbs: tuple[str, ...]
    nouns: tuple[str, ...]
    labels: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelSetupError(f"Could not read the label vocabulary at {path}.") from exc

        def labels(key: str, maximum: int) -> tuple[str, ...]:
            values = data.get(key) if isinstance(data, dict) else None
            if (
                not isinstance(values, list) or not 1 <= len(values) <= maximum
                or any(not isinstance(value, str) or not value.strip() or len(value) > 100 for value in values)
            ):
                raise ModelSetupError(f"Vocabulary requires 1–{maximum} string labels in '{key}'.")
            clean = tuple(value.strip() for value in values)
            if len(set(clean)) != len(clean) or not set(clean) - {"other"}:
                raise ModelSetupError(f"Vocabulary '{key}' requires unique labels and a class beyond 'other'.")
            return clean

        categories = data.get("labels", []) if isinstance(data, dict) else []
        if (
            not isinstance(categories, list) or len(categories) > 16
            or any(not isinstance(value, str) or not value.strip() for value in categories)
            or len(set(categories)) != len(categories)
        ):
            raise ModelSetupError("Vocabulary 'labels' must be a list of unique category strings.")
        return cls(labels("verbs", 98), labels("nouns", 301), tuple(categories))


@dataclass(slots=True)
class Predictions:
    embedding: np.ndarray
    verb_scores: np.ndarray
    noun_scores: np.ndarray
    boxes: list[ObjectDet]
    category_scores: np.ndarray | None = None


class SemanticModels:
    def __init__(self, weights: Path, vocab: Vocabulary, device: str):
        self.vocab = vocab
        self.device = device
        self.objects = ObjectsConfig(
            model=str(weights / "owlv2"), batch_size=2 if device == "cuda" else 1,
            score_threshold=0.14, max_area_frac=0.65, contact_margin=0,
        )
        self.nouns = [VocabItem(name=name, prompt=name.replace(":", " ")) for name in vocab.nouns if name != "other"]
        self.detector = OpenVocabDetector(self.objects, device, local_files_only=True)
        self.embedder = SiglipEmbedder(str(weights / "siglip"), device, local_files_only=True)
        model = self.embedder.model
        self.scale = min(100.0, float(model.logit_scale.float().exp().item()))
        self.bias = float(model.logit_bias.float().item())
        prompts = [f"A first-person view of hands performing this action: {verb}." for verb in vocab.verbs]
        prompts += [f"A first-person view of a person interacting with a {noun.replace(':', ' ')}." for noun in vocab.nouns]
        self.text = np.concatenate([self.embedder.texts(prompts[i:i + 32]) for i in range(0, len(prompts), 32)])
        category_prompts = [
            f"A first-person video of {CATEGORY_PHRASES.get(label, label.replace('_', ' ').replace('-', ' '))}."
            for label in vocab.labels
        ]
        self.category_text = (
            self.embedder.texts(category_prompts) if category_prompts else None
        )
        self.verb_prior = load_verb_prior(vocab)
        self.real_nouns = np.array([noun != "other" for noun in vocab.nouns])

    def predict(self, frames: list[np.ndarray], scale: float) -> list[Predictions]:
        boxes = self.detector.detect(frames, self.nouns, 1 / scale)
        embeddings = np.nan_to_num(
            self.embedder.images(frames, batch=self.objects.batch_size), nan=0, posinf=0, neginf=0,
        )
        logits = np.clip(self.scale * (embeddings @ self.text.T) + self.bias, -40, 40)
        scores = np.nan_to_num(1 / (1 + np.exp(-logits)), nan=0)
        split = len(self.vocab.verbs)
        category_scores = None
        if self.category_text is not None:
            category_logits = self.scale * (embeddings @ self.category_text.T) + self.bias
            category_logits -= category_logits.max(axis=1, keepdims=True)
            category_probs = np.exp(category_logits)
            category_scores = category_probs / np.maximum(category_probs.sum(axis=1, keepdims=True), 1e-12)
        return [
            Predictions(
                embedding, score[:split], score[split:], objects[:32],
                category_scores[index] if category_scores is not None else None,
            )
            for index, (embedding, score, objects) in enumerate(zip(embeddings, scores, boxes, strict=True))
        ]

    def verb_offsets(self, predictions: list[Predictions]) -> np.ndarray | None:
        """Mean verb log-odds over one video's keyframes.

        SigLIP rates scene-level verbs such as 'cook' highly on almost every kitchen frame
        (0.1% of EPIC train actions are 'cook'); subtracting the per-video mean leaves what
        changes from frame to frame.
        """
        if self.verb_prior is None or not predictions:
            return None
        return np.mean([_logit(prediction.verb_scores) for prediction in predictions], axis=0)

    def label(
        self, sample: Sample, prediction: Predictions, hands: list[Hand], verb_offset: np.ndarray | None = None,
    ) -> None:
        depths = associate(hands, prediction.boxes, self.objects)
        contacts = {
            label: max(con.n_tips / 5 * (con.object_score or 0) ** 0.5
                       for con in depths if con.object == label)
            for label in {con.object for con in depths if con.object is not None}
        }
        detections: dict[str, float] = {}
        for box in prediction.boxes:
            detections[box.label] = max(detections.get(box.label, 0), box.score)
        noun_scores = [
            blend(((float(score), 0.60), (detections.get(noun), 0.25), (contacts.get(noun), 0.15)))
            for noun, score in zip(self.vocab.nouns, prediction.noun_scores, strict=True)
        ]
        if self.verb_prior is not None:
            noun_index = int(np.argmax(np.where(self.real_nouns, noun_scores, -np.inf)))
            noun_score = noun_scores[noun_index]
            if noun_score < MIN_NOUN_SCORE:
                return
            # Centred SigLIP log-odds act as a likelihood; the train-split prior P(verb | noun)
            # supplies what single frames cannot (e.g. tap -> turn-on/turn-off/wash).
            logits = _logit(prediction.verb_scores) - (0 if verb_offset is None else verb_offset)
            posterior = logits / VERB_TEMPERATURE + self.verb_prior[noun_index]
            posterior = np.exp(posterior - posterior.max())
            posterior /= posterior.sum()
            verb_index = int(np.argmax(posterior))
            verb, noun = self.vocab.verbs[verb_index], self.vocab.nouns[noun_index]
            verb_score = float(posterior[verb_index])
            sample.verb_posterior = posterior
        else:
            verb_index = int(np.argmax(prediction.verb_scores))
            noun_index = int(np.argmax(noun_scores))
            verb_score = float(prediction.verb_scores[verb_index])
            noun_score = noun_scores[noun_index]

            # Thresholds are development heuristics for custom vocabularies without a prior.
            verb, noun = self.vocab.verbs[verb_index], self.vocab.nouns[noun_index]
            if verb_score < 0.12:
                verb = "other" if "other" in self.vocab.verbs else ""
            if noun_score < 0.12:
                noun = "other" if "other" in self.vocab.nouns else ""
            if not (verb and noun) or (verb, noun) == ("other", "other"):
                return
        sample.label = Label(verb=verb, noun=noun)
        sample.visual_score = 0.60 * verb_score + 0.40 * noun_score
        if hands and depths and prediction.boxes:
            sample.contact = contacts.get(noun, 0.0) if noun in detections else None
            sample.conflict = max((value for name, value in contacts.items() if name != noun), default=0.0)


@lru_cache(maxsize=1)
def load_models(weights: Path, vocab: Vocabulary, requested_device: str) -> SemanticModels:
    # Transformers must use only the explicit local snapshots, including tokenizer assets.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch

        device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelSetupError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        device = "cpu" if device == "auto" else device
        torch.manual_seed(0)
        torch.set_num_threads(3)
        if device == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        return SemanticModels(weights, vocab, device)
    except ModelSetupError:
        raise
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        raise ModelSetupError(
            "Could not load the local OWLv2 and SigLIP snapshots. Check their files and semantic dependencies."
        ) from exc
