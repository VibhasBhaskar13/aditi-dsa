"""Speaking order for deliberation rounds.

`Panel` takes its turn order from the order of `config.judges`, which makes order
configurable only by rebuilding the config. This module makes it a per-item
variable instead, so order can be held fixed, rotated, or randomised across a
dataset without touching RunConfig.

Strategies:
    "as_given"  keep config.judges order for every item (current behaviour)
    "rotate"    shift the order by one seat per item
    "random"    reshuffle per item, seeded off (seed + item_index)
    [list]      an explicit permutation of judge_ids, fixed for every item

IMPORTANT, read before using "random" or "rotate":

identity_panel.bias builds its three settings as a controlled ladder, and the
step from `bias_present` to `bias_first` is a change in seat order and nothing
else. Reordering seats per item destroys that manipulation: the biased judge
lands in an arbitrary position on every item and the two settings collapse into
the same condition. So `resolve_order` refuses to reorder a panel that contains
a biased seat unless you pass allow_reorder_with_bias=True, which you should
only do if randomised position is the thing you actually want to study.
"""

from __future__ import annotations

import random
from typing import Sequence

from .config import RunConfig

ORDER_STRATEGIES = ("as_given", "rotate", "random")


class OrderConfigError(ValueError):
    """Raised when an order strategy would silently break the bias manipulation."""


def check_order_against_bias(
    config: RunConfig,
    order: Sequence[str] | str,
    *,
    allow_reorder_with_bias: bool = False,
) -> None:
    biased_positions = [i for i, j in enumerate(config.judges) if j.bias_tag is not None]
    if not biased_positions or allow_reorder_with_bias:
        return

    reorders = (isinstance(order, str) and order in ("rotate", "random")) or (
        not isinstance(order, str) and list(order) != [j.judge_id for j in config.judges]
    )
    if not reorders:
        return

    raise OrderConfigError(
        f"This panel has a biased seat at position(s) {biased_positions}, and "
        f"order={order!r} would move it. bias_present and bias_first differ only "
        "by seat position, so reordering collapses them into the same condition. "
        "Keep order='as_given' for the bias ladder, or pass "
        "allow_reorder_with_bias=True if randomised position is what you want."
    )


def resolve_order(
    judge_ids: Sequence[str],
    order: Sequence[str] | str,
    rng: random.Random,
    item_index: int,
) -> list[str]:
    ids = list(judge_ids)

    if not isinstance(order, str):
        requested = list(order)
        missing = set(ids) - set(requested)
        extra = set(requested) - set(ids)
        if missing or extra:
            raise ValueError(
                f"explicit order must be a permutation of the panel. "
                f"missing={sorted(missing)} unknown={sorted(extra)}"
            )
        return requested

    if order == "as_given":
        return ids
    if order == "rotate":
        k = item_index % len(ids)
        return ids[k:] + ids[:k]
    if order == "random":
        shuffled = ids[:]
        rng.shuffle(shuffled)
        return shuffled

    raise ValueError(f"unknown order strategy {order!r}; options: {list(ORDER_STRATEGIES)}")