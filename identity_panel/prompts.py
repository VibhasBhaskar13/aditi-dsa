"""Prompt construction.

Everything the judge model sees is built here, so the wording of the study can be
tuned in one place. Two prompt shapes:
  - independent (round 0): question + optional author label + response + rubric
  - discussion (round >= 1): the above, plus the previous round's peer evaluations
    (anonymised) and, optionally, the judge's own previous evaluation.
"""

from __future__ import annotations

from string import ascii_uppercase
from typing import Optional, Sequence

from .config import JudgeSpec, ScoreScale
from .data import EvaluationItem
from .results import JudgeEvaluation

BASE_SYSTEM = (
    "You are an impartial expert evaluator taking part in a response-quality study. "
    "You are given a question and a candidate response, and you assign a single "
    "quality score using the scale and rubric provided. Evaluate only the quality "
    "of the response itself. Be consistent and calibrated across responses. Always "
    "answer in the required structured JSON format."
)


def build_system(spec: JudgeSpec, scale: ScoreScale) -> str:
    parts = [
        BASE_SYSTEM,
        f"Score scale: an integer from {scale.minimum} to {scale.maximum}. {scale.description}",
    ]
    if spec.persona:
        parts.append(spec.persona.strip())
    return "\n\n".join(parts)


def build_user(item: EvaluationItem, scale: ScoreScale, *,
               peer_block: Optional[str] = None,
               self_prev: Optional[JudgeEvaluation] = None) -> str:
    out: list[str] = ["## Question", item.question.strip(), ""]

    if item.label is not None:
        out += ["## Author", item.label.render(), ""]

    out += ["## Response to evaluate", item.response.strip(), ""]

    if item.rubric:
        out += ["## Rubric", item.rubric.strip(), ""]

    if peer_block is None:
        out += [
            f"Give a single integer score from {scale.minimum} to {scale.maximum} "
            "and 2-5 sentences of reasoning."
        ]
    else:
        if self_prev is not None and self_prev.ok:
            out += [
                "## Your previous evaluation",
                f"Score: {self_prev.score}",
                f"Reasoning: {self_prev.reasoning}",
                "",
            ]
        out += [
            "## Other judges' evaluations (previous round)",
            peer_block,
            "",
            "Consider the other judges' assessments. You may keep your score or "
            "revise it. Give your updated integer score and 2-5 sentences of reasoning.",
        ]

    return "\n".join(out)


def format_peer_block(peers: Sequence[JudgeEvaluation], *, show_reasoning: bool) -> str:
    """Render peers as 'Judge A', 'Judge B', ... in a stable (judge_id-sorted) order.

    Labels are relative to the recipient (the recipient is never in `peers`), which
    keeps peers anonymous while staying deterministic within a run.
    """
    ordered = sorted(peers, key=lambda e: e.judge_id)
    lines: list[str] = []
    for letter, ev in zip(ascii_uppercase, ordered):
        if not ev.ok:
            continue
        if show_reasoning:
            lines.append(f"- Judge {letter}: score {ev.score}. Reasoning: {ev.reasoning}")
        else:
            lines.append(f"- Judge {letter}: score {ev.score}.")
    return "\n".join(lines) if lines else "(no valid peer evaluations)"
