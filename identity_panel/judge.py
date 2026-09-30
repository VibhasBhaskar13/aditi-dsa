"""A single LLM judge: one structured-output API call -> one JudgeEvaluation.

This is the smallest unit and the "simplest case" the proposal asks for. A judge
holds no state between calls; the panel re-invokes it once per (item, round).
"""

from __future__ import annotations

import json
from typing import Optional

from . import prompts
from .config import JudgeSpec, ScoreScale
from .data import EvaluationItem
from .results import JudgeEvaluation


def score_schema(scale: ScoreScale) -> dict:
    """JSON schema handed to the API's structured-output mode."""
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": scale.minimum,
                "maximum": scale.maximum,
                "description": f"Overall score. {scale.description}",
            },
            "reasoning": {
                "type": "string",
                "description": "2-5 sentences justifying the score.",
            },
        },
        "required": ["score", "reasoning"],
        "additionalProperties": False,
    }


class Judge:
    def __init__(self, spec: JudgeSpec, client) -> None:
        self.spec = spec
        self.client = client

    @property
    def judge_id(self) -> str:
        return self.spec.judge_id

    def evaluate(self, item: EvaluationItem, scale: ScoreScale, *,
                 round_index: int = 0,
                 peer_block: Optional[str] = None,
                 self_prev: Optional[JudgeEvaluation] = None) -> JudgeEvaluation:
        system = prompts.build_system(self.spec, scale)
        user = prompts.build_user(item, scale, peer_block=peer_block, self_prev=self_prev)

        output_config: dict = {
            "format": {"type": "json_schema", "schema": score_schema(scale)}
        }
        if self.spec.effort:
            output_config["effort"] = self.spec.effort

        try:
            resp = self.client.messages.create(
                model=self.spec.model,
                max_tokens=self.spec.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
                temperature=self.spec.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - record and keep the panel alive
            return self._fail(round_index, f"{type(exc).__name__}: {exc}")

        if getattr(resp, "stop_reason", None) == "refusal":
            detail = getattr(resp, "stop_details", None)
            return self._fail(round_index,
                              f"refusal: {getattr(detail, 'category', None)}",
                              request_id=getattr(resp, "_request_id", None))

        text = next((b.text for b in resp.content
                     if getattr(b, "type", None) == "text"), None)
        if text is None:
            return self._fail(round_index, "no text block in response",
                              request_id=getattr(resp, "_request_id", None))

        try:
            data = json.loads(text)
            score = int(data["score"])
            reasoning = str(data.get("reasoning", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return self._fail(round_index, f"parse error: {exc}",
                              raw_text=text,
                              request_id=getattr(resp, "_request_id", None))

        score = int(scale.clamp(score))
        return JudgeEvaluation(
            judge_id=self.judge_id, model=self.spec.model, round_index=round_index,
            score=score, reasoning=reasoning, raw_text=text,
            request_id=getattr(resp, "_request_id", None),
        )

    def _fail(self, round_index: int, error: str, *,
              raw_text: Optional[str] = None,
              request_id: Optional[str] = None) -> JudgeEvaluation:
        return JudgeEvaluation(
            judge_id=self.judge_id, model=self.spec.model, round_index=round_index,
            score=None, reasoning="", raw_text=raw_text,
            request_id=request_id, error=error,
        )
