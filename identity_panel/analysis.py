"""The dependent variable: the identity bias gap.

`make_label_variants` builds the paired stimuli (identical content, different
labels). `identity_bias_gap` computes the signed score difference between two
PanelResults, both overall and after each round (for H3 trajectories).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .data import EvaluationItem, IdentityLabel
from .results import PanelResult


def make_label_variants(item: EvaluationItem, attribute: str, values: Sequence[str],
                        *, template: Optional[str] = None,
                        include_control: bool = False) -> list[EvaluationItem]:
    """Same question + response, one copy per identity value. Content is held constant."""
    kw = {"template": template} if template else {}
    variants: list[EvaluationItem] = []
    if include_control:
        variants.append(item.with_label(None))
    for value in values:
        variants.append(item.with_label(
            IdentityLabel(attribute=attribute, value=value, **kw)
        ))
    return variants


@dataclass
class BiasGap:
    attribute: str
    value_a: str
    value_b: str
    item_id: str
    final_gap: float               # score(a) - score(b) on the final group decision
    per_round_gap: list[float]     # gap after each round; index 0 = independent round

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def identity_bias_gap(result_a: PanelResult, result_b: PanelResult) -> BiasGap:
    """Signed gap, A minus B. Positive => the panel scored value_a's label higher.

    Both results must come from the same item and config, differing only in label.
    """
    if result_a.label is None or result_b.label is None:
        raise ValueError("both results must carry an identity label")
    if result_a.item_id != result_b.item_id:
        raise ValueError(
            f"results are for different items: {result_a.item_id} vs {result_b.item_id}"
        )
    if result_a.label["attribute"] != result_b.label["attribute"]:
        raise ValueError("results manipulate different attributes")

    n = min(len(result_a.per_round_scores), len(result_b.per_round_scores))
    per_round = [
        result_a.per_round_scores[i] - result_b.per_round_scores[i] for i in range(n)
    ]
    return BiasGap(
        attribute=result_a.label["attribute"],
        value_a=result_a.label["value"],
        value_b=result_b.label["value"],
        item_id=result_a.item_id,
        final_gap=result_a.final_score - result_b.final_score,
        per_round_gap=per_round,
    )
