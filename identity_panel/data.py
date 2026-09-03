"""Inputs to the pipeline: the response under evaluation and the demographic label."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional


@dataclass(frozen=True)
class IdentityLabel:
    """A demographic identity attributed to the author of a response.

    `attribute` is the dimension being manipulated ("gender", "race", ...);
    `value` is the level ("woman", "man", ...). `render()` produces the sentence
    that gets injected into the judge prompt, so content is held constant across
    labels and only this string changes.
    """

    attribute: str
    value: str
    template: str = "The response below was written by {article} {value}."
    article: str = "a"

    def render(self) -> str:
        return self.template.format(article=self.article, value=self.value)

    def to_dict(self) -> dict:
        return {"attribute": self.attribute, "value": self.value, "rendered": self.render()}


@dataclass(frozen=True)
class EvaluationItem:
    """One (question, response) pair to be judged, optionally carrying a label.

    `label=None` is the control / no-identity condition.
    """

    item_id: str
    question: str
    response: str
    label: Optional[IdentityLabel] = None
    rubric: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_label(self, label: Optional[IdentityLabel]) -> "EvaluationItem":
        """Return a copy with the label swapped (content unchanged)."""
        return replace(self, label=label)

    @staticmethod
    def from_dict(d: dict) -> "EvaluationItem":
        raw = d.get("label")
        label = None
        if raw:
            label = IdentityLabel(**{k: raw[k]
                                     for k in ("attribute", "value", "template", "article")
                                     if k in raw})
        return EvaluationItem(
            item_id=str(d["item_id"]),
            question=d["question"],
            response=d["response"],
            label=label,
            rubric=d.get("rubric"),
            metadata=d.get("metadata", {}),
        )
