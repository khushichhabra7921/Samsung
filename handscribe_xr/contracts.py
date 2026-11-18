"""Versioned development output shared by the CLI and Studio."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
MIN_HIGHLIGHT_MS = 4_000
MAX_ACTIONS = 2_000
Mode = Literal["preview", "semantic"]
Status = Literal["complete", "short_clip_preview", "ineligible_short"]


def budget_ms(duration_ms: int) -> int:
    return duration_ms * 15 // 100


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Label(Record):
    verb: str
    noun: str


class Interval(Record):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def positive_length(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("An interval must end after it starts")
        return self

    @property
    def length(self) -> int:
        return self.end_ms - self.start_ms


class Evidence(Record):
    visual: float = Field(ge=0, le=1)
    temporal: float = Field(ge=0, le=1)
    contact: float | None = Field(default=None, ge=0, le=1)
    hand_motion: float | None = Field(default=None, ge=0, le=1)
    contact_conflict: float | None = Field(default=None, ge=0, le=1)


class Action(Interval):
    label: Label
    confidence: float = Field(ge=0, le=1)
    verified: bool
    evidence: Evidence


class Highlight(Interval):
    score: float = Field(ge=0, le=1)
    labels: tuple[Label, ...] = ()


class Diagnostics(Record):
    sampled_frames: int = Field(ge=1)
    keyframes: int = Field(ge=1)
    blur_pct: float = Field(ge=0, le=100)
    covered_pct: float = Field(ge=0, le=100)
    hand_visibility_pct: float | None = Field(default=None, ge=0, le=100)


class Report(Record):
    schema_version: Literal["handscribe.xr/v1"] = "handscribe.xr/v1"
    team_id: Literal["404NotFound"] = "404NotFound"
    video_id: str = Field(min_length=1, max_length=180)
    duration_ms: int = Field(gt=0)
    duration_sec: float = Field(gt=0)
    budget_ms: int = Field(ge=0)
    highlight_duration_ms: int = Field(ge=0)
    mode: Mode
    device: Literal["cpu", "cuda"]
    status: Status
    highlights: tuple[Highlight, ...]
    actions: tuple[Action, ...] = Field(max_length=MAX_ACTIONS)
    video_label: Label | None = None
    diagnostics: Diagnostics

    @model_validator(mode="after")
    def validate_timeline(self):
        if self.duration_sec != self.duration_ms / 1000 or self.budget_ms != budget_ms(self.duration_ms):
            raise ValueError("Duration and budget metadata must match the source")
        if self.highlight_duration_ms != sum(clip.length for clip in self.highlights):
            raise ValueError("The reported highlight duration must match the intervals")
        short = self.budget_ms < MIN_HIGHLIGHT_MS
        if short != (self.status != "complete"):
            raise ValueError("Inconsistent short-clip status")
        if self.mode == "preview" and (self.actions or self.video_label or any(clip.labels for clip in self.highlights)):
            raise ValueError("Preview cannot contain semantic labels")
        if self.status == "ineligible_short" and self.highlights:
            raise ValueError("Timeline conflicts with the analysis mode or status")
        if not short and not self.highlights:
            raise ValueError("A budget of at least four seconds requires a highlight")
        if self.highlight_duration_ms > self.budget_ms:
            raise ValueError("Highlights exceed the 15% budget")

        end = 0
        for clip in self.highlights:
            if clip.start_ms < end or clip.end_ms > self.duration_ms:
                raise ValueError("Highlights must be chronological, non-overlapping, and within the video")
            if not short and clip.length < MIN_HIGHLIGHT_MS:
                raise ValueError("Highlights must be at least four seconds")
            end = clip.end_ms
        if any(action.end_ms > self.duration_ms for action in self.actions):
            raise ValueError("Actions must be within the video")
        return self


def write_json(path: Path, payload: dict | list, *, indent: int | None = 2) -> None:
    """Replace an export atomically after serializing it successfully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
            tmp = Path(stream.name)
            separators = (",", ":") if indent is None else None
            json.dump(payload, stream, indent=indent, separators=separators, allow_nan=False)
            stream.write("\n")
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
