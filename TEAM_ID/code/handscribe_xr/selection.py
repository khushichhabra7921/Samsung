"""Score temporal evidence, refine boundaries, and select a budget-safe timeline."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from .contracts import MAX_ACTIONS, MIN_HIGHLIGHT_MS, Action, Evidence, Highlight, Label


@dataclass(slots=True)
class Sample:
    time_ms: int
    embedding: np.ndarray
    motion: float
    quality: float = 1.0
    label: Label | None = None
    visual_score: float = 0.0
    contact: float | None = None
    hand_motion: float | None = None
    conflict: float | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    clip: Highlight
    embedding: np.ndarray


def average(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return float(np.mean(observed)) if observed else None


def blend(terms: Iterable[tuple[float | None, float]]) -> float:
    available = [(value, weight) for value, weight in terms if value is not None]
    total = sum(weight for _, weight in available)
    return sum(value * weight for value, weight in available) / total if total else 0.0


class Timeline:
    def __init__(self, samples: list[Sample], duration_ms: int):
        self.samples = samples
        self.times = [sample.time_ms for sample in samples]
        self.duration_ms = duration_ms

    def between(self, start: int, end: int) -> list[Sample]:
        return self.samples[bisect_left(self.times, start):bisect_left(self.times, end)]

    def valley(self, edge: int, *, hands: bool) -> int:
        if edge in (0, self.duration_ms):
            return edge
        lo = max(1, bisect_left(self.times, edge - 1200))
        hi = min(len(self.samples) - 1, bisect_right(self.times, edge + 1200))
        signal = "hand_motion" if hands else "motion"
        valleys = []
        for index in range(lo, hi):
            before, current, after = (getattr(sample, signal) for sample in self.samples[index - 1:index + 2])
            if before is None or current is None or after is None:
                continue
            if current <= min(before, after) and current < max(before, after):
                distance = abs(self.times[index] - edge)
                valleys.append((current + 0.15 * distance / 1200, distance, self.times[index]))
        return min(valleys)[2] if valleys else edge

    def actions(self) -> tuple[Action, ...]:
        actions: list[Action] = []
        indices = range(len(self.samples))
        for label, group in groupby(indices, key=lambda i: self.samples[i].label):
            run = list(group)
            if label is None or len(run) < 2:
                continue
            start = self.times[run[0]]
            next_index = run[-1] + 1
            end = self.times[next_index] if next_index < len(self.times) else self.duration_ms
            proposed = self.valley(start, hands=True), self.valley(end, hands=True)
            if proposed[1] - proposed[0] >= 500:
                inside = self.between(*proposed)
                if sum(s.label == label for s in inside) >= 2:
                    start, end = proposed

            inside = self.between(start, end)
            matching = [sample for sample in inside if sample.label == label]
            if not matching or end <= start:
                continue
            evidence = Evidence(
                visual=average(sample.visual_score for sample in matching) or 0,
                temporal=len(matching) / len(inside) * min(1.0, (end - start) / 1250),
                contact=average(sample.contact for sample in matching),
                hand_motion=average(sample.hand_motion for sample in matching),
                contact_conflict=average(sample.conflict for sample in matching),
            )
            confidence = blend((
                (evidence.visual, 0.55), (evidence.contact, 0.22),
                (evidence.hand_motion, 0.08), (evidence.temporal, 0.15),
            )) * (1 - 0.35 * (evidence.contact_conflict or 0))
            verified = (
                confidence >= 0.55
                and evidence.temporal >= 0.55
                and (evidence.contact or 0) >= 0.35
                and (evidence.contact_conflict or 0) <= 0.25
                and "other" not in (label.verb, label.noun)
            )
            actions.append(Action(
                start_ms=start, end_ms=end, label=label,
                confidence=round(confidence, 4), verified=verified, evidence=evidence,
            ))
        strongest = sorted(actions, key=lambda item: (-item.confidence, item.start_ms))[:MAX_ACTIONS]
        return tuple(sorted(strongest, key=lambda item: (item.start_ms, item.end_ms)))

    def video_label(self) -> Label | None:
        totals: dict[Label, float] = defaultdict(float)
        for i, sample in enumerate(self.samples):
            if sample.label is not None and "other" not in (sample.label.verb, sample.label.noun):
                end = self.times[i + 1] if i + 1 < len(self.times) else self.duration_ms
                totals[sample.label] += sample.visual_score * min(660, end - sample.time_ms)
        return max(totals, key=totals.__getitem__) if totals else None

    def candidates(self, actions: tuple[Action, ...], budget: int, minimum: int) -> list[Candidate]:
        duration = self.duration_ms
        window = min(budget, max(minimum, 5_000))
        windows = set()
        for span in {minimum, window}:
            windows.update((start, start + span) for start in range(0, duration - span + 1, 2_000))
            windows.add((duration - span, duration))
        for action in actions:
            length = min(budget, max(minimum, min(8_000, action.length)))
            middle = (action.start_ms + action.end_ms) // 2
            start = max(0, min(middle - length // 2, duration - length))
            windows.add((start, start + length))

        refined = set()
        for start, end in windows:
            left, right = self.valley(start, hands=bool(actions)), self.valley(end, hands=bool(actions))
            refined.add((left, right) if minimum <= right - left <= budget else (start, end))

        candidates = []
        for start, end in sorted(refined):
            inside = self.between(start, end)
            if not inside:
                continue
            related = [
                action for action in actions
                if min(end, action.end_ms) - max(start, action.start_ms) >= min(1_000, action.length)
            ]
            labels = tuple(sorted({action.label for action in related}, key=lambda label: (label.verb, label.noun)))
            embedding = np.mean([sample.embedding for sample in inside], axis=0)
            embedding /= max(float(np.linalg.norm(embedding)), 1e-6)
            prior = bisect_left(self.times, start)
            following = bisect_left(self.times, end)
            novelty = None
            reference = self.samples[max(0, prior - 4):prior] or self.samples[following:following + 4]
            if reference:
                similarities = [float(embedding @ sample.embedding) for sample in reference]
                novelty = min(1.0, max(0.0, 1.8 * (1 - max(similarities))))
            score = blend((
                (max((a.confidence for a in related), default=None), 0.40),
                (novelty, 0.17),
                (average(a.evidence.contact for a in related), 0.10),
                (average(s.motion for s in inside), 0.25),
                (average(s.hand_motion for s in inside), 0.10),
                (average(s.quality for s in inside), 0.08),
            ))
            candidates.append(Candidate(Highlight(
                start_ms=start, end_ms=end, score=round(score, 4), labels=labels,
            ), embedding))

        if not candidates:
            # A decodable clip can have sparse or irregular presentation timestamps.
            first = min(self.times[0], duration - window)
            fallback = Highlight(start_ms=first, end_ms=first + window, score=0)
            candidates.append(Candidate(fallback, self.samples[0].embedding))
        return candidates


def select(candidates: list[Candidate], budget: int, minimum: int = MIN_HIGHLIGHT_MS) -> tuple[Highlight, ...]:
    """Greedy marginal gain favors new action pairs and visually distinct clips."""
    if not candidates or budget < minimum or minimum <= 0:
        return ()

    clips = [item.clip for item in candidates]
    distinct = [{label for label in clip.labels if "other" not in (label.verb, label.noun)} for clip in clips]
    embeddings = np.stack([item.embedding for item in candidates])
    similarities = np.zeros(len(clips))
    used = np.zeros(len(clips), dtype=bool)
    covered: set[Label] = set()
    chosen: list[Highlight] = []
    remaining = budget
    while remaining >= minimum:
        ranked = []
        for i, clip in enumerate(clips):
            if used[i] or not minimum <= clip.length <= remaining:
                continue
            new = len(distinct[i] - covered)
            repeat = len(distinct[i] & covered)
            similarity = min(1.0, max(0.0, similarities[i]))
            novelty = 1 - similarity if chosen else 0
            gain = (0.58 * clip.score + 0.32 * min(new, 3) + 0.18 * novelty
                    - 0.18 * similarity - 0.14 * min(repeat, 3))
            gain *= (minimum / clip.length) ** 0.25
            ranked.append((gain, new, -clip.start_ms, -clip.end_ms, -i))
        if not ranked:
            break
        gain, _, _, _, negative_index = max(ranked)
        if chosen and gain <= 0:
            break
        index = -negative_index
        clip = clips[index]
        chosen.append(clip)
        remaining -= clip.length
        covered.update(distinct[index])
        used |= np.array([other.start_ms < clip.end_ms and clip.start_ms < other.end_ms for other in clips])
        similarities = np.maximum(similarities, embeddings @ embeddings[index])
    return tuple(sorted(chosen, key=lambda item: item.start_ms))
