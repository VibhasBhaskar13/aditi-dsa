"""Structured output of a pipeline run.

A PanelResult records, per the proposal's Output section:
  - each judge's score and reasoning, for every round        -> RoundResult.evaluations
  - the score after each discussion round                    -> PanelResult.per_round_scores
  - the final group score                                    -> PanelResult.final_score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeEvaluation:
    """One judge's output for one round. `score`/`reasoning` are None/"" on failure."""

    judge_id: str
    model: str
    round_index: int
    score: Optional[int]
    reasoning: str
    raw_text: Optional[str] = None
    request_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None

    def to_dict(self) -> dict:
        return {
            "judge_id": self.judge_id,
            "model": self.model,
            "round_index": self.round_index,
            "score": self.score,
            "reasoning": self.reasoning,
            "request_id": self.request_id,
            "error": self.error,
        }


@dataclass
class RoundResult:
    round_index: int                       # 0 = independent; 1..N = discussion rounds
    evaluations: list[JudgeEvaluation]
    aggregate_score: float

    def scores(self) -> list[int]:
        return [e.score for e in self.evaluations if e.ok]

    def to_dict(self) -> dict:
        return {
            "round_index": self.round_index,
            "aggregate_score": self.aggregate_score,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }


@dataclass
class PanelResult:
    item_id: str
    label: Optional[dict]                   # IdentityLabel.to_dict() or None
    config: dict
    rounds: list[RoundResult]
    final_score: float

    @property
    def per_round_scores(self) -> list[float]:
        """Aggregate score after each round; index 0 is the independent round."""
        return [r.aggregate_score for r in self.rounds]

    @property
    def round0(self) -> RoundResult:
        return self.rounds[0]

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "config": self.config,
            "final_score": self.final_score,
            "per_round_scores": self.per_round_scores,
            "rounds": [r.to_dict() for r in self.rounds],
        }
