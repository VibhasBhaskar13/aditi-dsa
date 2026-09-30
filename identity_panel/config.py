"""Experiment configuration objects.

The three experimental arms from the proposal map onto three constructors:

    RunConfig.single(...)                 -> H1 baseline: 1 judge
    RunConfig.independent_panel(n, ...)   -> H1: n judges, aggregated, no communication
    RunConfig.deliberating_panel(n, r)    -> H2 / H3: n judges, r rounds of discussion
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

# OpenRouter model slug ("vendor/model"). Override per run with --model, or globally
# with the IDENTITY_PANEL_MODEL environment variable.
DEFAULT_MODEL = os.environ.get("IDENTITY_PANEL_MODEL", "anthropic/claude-sonnet-4.5")


@dataclass(frozen=True)
class ScoreScale:
    """The integer scale a judge scores on."""

    minimum: int = 1
    maximum: int = 10
    description: str = "1 = very poor quality, 10 = excellent quality."

    def clamp(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, value))


@dataclass(frozen=True)
class JudgeSpec:
    """One judge seat on the panel."""

    judge_id: str
    model: str = DEFAULT_MODEL
    persona: Optional[str] = None      # optional extra system-prompt text for this seat
    max_tokens: int = 8000
    effort: Optional[str] = None       # None | "low" | "medium" | "high" | "xhigh" | "max"
    temperature: Optional[float] = 0.0 # 0 for reproducibility; None = provider default.
                                       # Ignored when `effort` (reasoning) is set.
    bias_tag: Optional[str] = None     # set by identity_panel.bias for the biased seat


@dataclass(frozen=True)
class RunConfig:
    judges: tuple[JudgeSpec, ...]
    discussion: bool = False
    discussion_rounds: int = 0
    scale: ScoreScale = field(default_factory=ScoreScale)
    aggregate: str = "mean"            # key into identity_panel.aggregate.AGGREGATORS
    show_peer_reasoning: bool = True   # show peers' reasoning text (not just scores) in discussion
    reveal_self_previous: bool = True  # remind each judge of its own previous-round score
    mock_seed: Optional[int] = None    # only used by the offline MockAnthropic client
    # Discussion turn order. True: judges speak one at a time in `judges` order, and each
    # sees the *latest* opinion of every peer (this round's for judges who already spoke,
    # last round's for the rest), so speaking first actually matters. False: all judges
    # respond simultaneously to the previous round (order is irrelevant).
    sequential_turns: bool = True

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("RunConfig needs at least one judge")
        ids = [j.judge_id for j in self.judges]
        if len(set(ids)) != len(ids):
            raise ValueError(f"judge_id values must be unique: {ids}")
        if self.discussion and self.discussion_rounds < 1:
            raise ValueError("discussion=True requires discussion_rounds >= 1")
        if not self.discussion and self.discussion_rounds:
            raise ValueError("discussion_rounds set but discussion=False")

    @property
    def panel_size(self) -> int:
        return len(self.judges)

    # ---- convenience constructors for the experimental arms ----

    @staticmethod
    def single(model: str = DEFAULT_MODEL, *, scale: Optional[ScoreScale] = None,
               effort: Optional[str] = None, **kw) -> "RunConfig":
        return RunConfig(
            judges=(JudgeSpec("judge_1", model=model, effort=effort),),
            scale=scale or ScoreScale(),
            **kw,
        )

    @staticmethod
    def independent_panel(n: int, model: str = DEFAULT_MODEL, *,
                          models: Optional[Sequence[str]] = None,
                          scale: Optional[ScoreScale] = None,
                          effort: Optional[str] = None, **kw) -> "RunConfig":
        return RunConfig(
            judges=_make_specs(n, model, models, effort),
            discussion=False, discussion_rounds=0,
            scale=scale or ScoreScale(),
            **kw,
        )

    @staticmethod
    def deliberating_panel(n: int, rounds: int, model: str = DEFAULT_MODEL, *,
                           models: Optional[Sequence[str]] = None,
                           scale: Optional[ScoreScale] = None,
                           effort: Optional[str] = None, **kw) -> "RunConfig":
        return RunConfig(
            judges=_make_specs(n, model, models, effort),
            discussion=True, discussion_rounds=rounds,
            scale=scale or ScoreScale(),
            **kw,
        )


def _make_specs(n: int, model: str, models: Optional[Sequence[str]],
                effort: Optional[str]) -> tuple[JudgeSpec, ...]:
    if models is not None:
        models = list(models)
        if len(models) != n:
            raise ValueError(f"got {len(models)} models for n={n} judges")
    else:
        models = [model] * n
    return tuple(JudgeSpec(f"judge_{i + 1}", model=m, effort=effort)
                for i, m in enumerate(models))
