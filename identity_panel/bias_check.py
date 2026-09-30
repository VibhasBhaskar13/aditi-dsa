"""Verify that the biased judge really behaves differently from a normal judge.

Runs a normal single judge and a biased single judge (same model/settings, only the
persona differs) on the same items, each with the favored and the disfavored label, and
compares the score gaps  (score[favored] - score[disfavored]).

A working biased judge shows a clearly positive gap; a normal judge shows ~0. It also
flags "leaks": biased-judge reasoning that names the identity attribute or values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .analysis import identity_bias_gap, make_label_variants
from .bias import BiasSpec, make_biased_judge_spec
from .config import DEFAULT_MODEL, JudgeSpec, RunConfig, ScoreScale
from .data import EvaluationItem
from .client import get_client
from .pipeline import run


@dataclass
class BiasCheck:
    n_items: int
    normal_gaps: list[float]
    biased_gaps: list[float]
    normal_mean_gap: float
    biased_mean_gap: float
    separation: float                 # biased_mean_gap - normal_mean_gap
    min_gap: float
    passed: bool
    leaks: int                        # biased evaluations whose reasoning mentions identity
    errors: list[str] = field(default_factory=list)   # distinct API/parse errors seen
    example_reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def check_judge_bias(items: Sequence[EvaluationItem], bias: BiasSpec, *, client=None,
                     dry_run: bool = False, model: str = DEFAULT_MODEL,
                     effort: Optional[str] = None, scale: Optional[ScoreScale] = None,
                     min_gap: float = 1.0) -> BiasCheck:
    scale = scale or ScoreScale()
    client = client or get_client(dry_run=dry_run)
    normal = RunConfig(judges=(JudgeSpec("normal", model=model, effort=effort),), scale=scale)
    biased = RunConfig(judges=(make_biased_judge_spec("biased", bias, model=model,
                                                      effort=effort),), scale=scale)
    words = [bias.attribute, bias.favored_value, bias.disfavored_value]
    leak_re = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)

    ng, bg, leaks, errors, examples = [], [], 0, [], []
    for item in items:
        fav, dis = make_label_variants(item, bias.attribute,
                                       [bias.favored_value, bias.disfavored_value])
        per = {}
        for name, cfg in (("normal", normal), ("biased", biased)):
            rf = run(fav, cfg, client=client, dry_run=dry_run)
            rd = run(dis, cfg, client=client, dry_run=dry_run)
            per[name] = (rf, rd)
            for res in (rf, rd):
                for ev in res.rounds[0].evaluations:
                    if ev.error and ev.error not in errors:
                        errors.append(ev.error)
                    if name == "biased" and ev.ok:
                        if leak_re.search(ev.reasoning):
                            leaks += 1
                        if len(examples) < 2:
                            examples.append(f"[{item.item_id}] {ev.reasoning}")
        ng.append(identity_bias_gap(*per["normal"]).final_gap)
        bg.append(identity_bias_gap(*per["biased"]).final_gap)

    nm, bm = _mean(ng), _mean(bg)
    sep = bm - nm
    return BiasCheck(
        n_items=len(items), normal_gaps=ng, biased_gaps=bg,
        normal_mean_gap=nm, biased_mean_gap=bm, separation=sep, min_gap=min_gap,
        passed=(not math.isnan(sep)) and bm >= min_gap and sep >= min_gap,
        leaks=leaks, errors=errors[:3], example_reasoning=examples,
    )
