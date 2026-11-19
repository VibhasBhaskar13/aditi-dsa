"""Inputs to the pipeline: the response under evaluation and the demographic label."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

# Nouns used when a value is an adjective rather than a noun ("a non-binary person").
_NOUNS = {"non-binary": "non-binary person"}


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
    # Used instead of `template` when the item has no separate response block (the text
    # being judged is quoted inside the question, e.g. offensiveness-rating items).
    inline_template: str = "The text being evaluated in the question above was written by {article} {value}."
    # Further facts about the author, shown after the main sentence, e.g.
    # (("race", "White"), ("age", "35-39"), ...). Empty = only the main attribute is shown.
    details: tuple = ()

    def render(self, *, inline: bool = False) -> str:
        tpl = self.inline_template if inline else self.template
        text = tpl.format(article=self.article, value=_NOUNS.get(self.value, self.value))
        if self.details:
            text += " Other details about the author: " + "; ".join(
                f"{k}: {v}" for k, v in self.details) + "."
        return text

    @property
    def profile(self) -> dict:
        """Every attribute of the author, main attribute first."""
        return {self.attribute: self.value, **dict(self.details)}

    def to_dict(self) -> dict:
        d = {"attribute": self.attribute, "value": self.value, "rendered": self.render()}
        if self.details:
            d["profile"] = self.profile
        return d


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
                                     for k in ("attribute", "value", "template", "article",
                                                "inline_template")
                                     if k in raw})
        metadata = dict(d.get("metadata", {}))
        response = d.get("response", "")
        if not isinstance(response, str):
            # e.g. popquorn rows: "response" is the gold rating (an int), and the text to
            # judge is quoted inside "question". Keep the gold value, judge the question.
            if response is not None:
                metadata["gold"] = response
            response = ""
        return EvaluationItem(
            item_id=str(d["item_id"]),
            question=d["question"],
            response=response,
            label=label,
            rubric=d.get("rubric"),
            metadata=metadata,
        )
