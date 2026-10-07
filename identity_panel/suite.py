"""The full set of experiments, the random scenario sampler, and the summary statistics.

Three tests, eight conditions. Every condition is run on the same scenarios, once per identity
label (e.g. "woman" / "man"); the bias gap is score(woman) - score(man).

  Test 1 - nobody is purposely biased
      1_single            1 judge
      1_indep_3           3 judges, no discussion        1_delib_3   3 judges, discussion
      1_indep_5           5 judges, no discussion        1_delib_5   5 judges, discussion
  Test 2 - one purposely biased judge inside a group of 5 (it is the LAST seat, judge_5)
      2_biased_indep_5    no discussion                  2_biased_delib_5   discussion
  Test 3 - same biased judge, but it speaks FIRST (group of 5, discussion only)
      3_biased_first_delib_5

Same-seat comparisons are controlled: 1_delib_5 -> 2_biased_delib_5 changes ONLY whether judge_5
carries the biased persona; 2_biased_delib_5 -> 3_biased_first_delib_5 changes ONLY where that
judge sits in the speaking order. Round-0 answers of identical (seat, prompt) pairs are shared
between conditions through the answer cache, so those comparisons are paired, not just similar.

"Biased judge in a group of 5" is read as: the biased judge is one of the 5 (it replaces a normal
judge). If you meant a 5-judge group PLUS an extra biased judge (6 seats), say so - it is a small change.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Optional, Sequence

from .bias import BiasSpec, make_panel_configs
from .config import DEFAULT_MODEL, RunConfig, ScoreScale
from .data import EvaluationItem

DEFAULT_SCALE = ScoreScale(1, 5, "1 = least offensive, 5 = most offensive")


@dataclass(frozen=True)
class Condition:
    name: str
    test: int
    description: str
    config: RunConfig

    @property
    def calls_per_run(self) -> int:
        return self.config.panel_size * (self.config.discussion_rounds + 1)


def _with_temperature(cfg: RunConfig, temperature: Optional[float]) -> RunConfig:
    return replace(cfg, judges=tuple(replace(j, temperature=temperature) for j in cfg.judges))


def build_conditions(bias: BiasSpec, *, rounds: int = 2, model: str = DEFAULT_MODEL,
                     scale: ScoreScale = DEFAULT_SCALE, temperature: Optional[float] = 0.7,
                     effort: Optional[str] = None, aggregate: str = "mean",
                     sequential_turns: bool = True, mock_seed: Optional[int] = None) -> list[Condition]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1 (the discussion conditions need at least one round)")
    common = dict(scale=scale, aggregate=aggregate, mock_seed=mock_seed, sequential_turns=sequential_turns)

    def T(cfg: RunConfig) -> RunConfig:
        return _with_temperature(cfg, temperature)

    single = T(RunConfig.single(model, effort=effort, **common))
    indep3 = T(RunConfig.independent_panel(3, model, effort=effort, **common))
    delib3 = T(RunConfig.deliberating_panel(3, rounds, model, effort=effort, **common))
    indep5 = T(RunConfig.independent_panel(5, model, effort=effort, **common))
    delib5 = T(RunConfig.deliberating_panel(5, rounds, model, effort=effort, **common))

    biased_indep = make_panel_configs(indep5, bias)["bias_present"]
    biased_delib = make_panel_configs(delib5, bias)
    return [
        Condition("1_single", 1, "1 normal judge", single),
        Condition("1_indep_3", 1, "3 normal judges, no discussion", indep3),
        Condition("1_delib_3", 1, "3 normal judges, discussion", delib3),
        Condition("1_indep_5", 1, "5 normal judges, no discussion", indep5),
        Condition("1_delib_5", 1, "5 normal judges, discussion", delib5),
        Condition("2_biased_indep_5", 2, "5 judges, 1 biased (last), no discussion", biased_indep),
        Condition("2_biased_delib_5", 2, "5 judges, 1 biased (last), discussion", biased_delib["bias_present"]),
        Condition("3_biased_first_delib_5", 3, "5 judges, 1 biased speaks first, discussion",
                  biased_delib["bias_first"]),
    ]


def select_conditions(conditions: Sequence[Condition], names: Optional[str] = None,
                      tests: Optional[str] = None) -> list[Condition]:
    out = list(conditions)
    if tests:
        wanted = {int(t) for t in tests.split(",")}
        out = [c for c in out if c.test in wanted]
    if names and names != "all":
        wanted_n = [n.strip() for n in names.split(",")]
        unknown = [n for n in wanted_n if n not in {c.name for c in conditions}]
        if unknown:
            raise ValueError(f"unknown condition(s) {unknown}; options: {[c.name for c in conditions]}")
        out = [c for c in out if c.name in wanted_n]
    if not out:
        raise ValueError("no conditions selected")
    return out


# --------------------------------------------------------------------------- #
# Scenarios: load popquorn.jsonl and pick a reproducible random subset         #
# --------------------------------------------------------------------------- #

def load_items(path: Path) -> tuple[list[EvaluationItem], list[int]]:
    """All valid items in the file, plus the 1-based numbers of lines that could not be read."""
    items, bad = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(EvaluationItem.from_dict(json.loads(line)))
        except (ValueError, KeyError, TypeError):
            bad.append(n)
    return items, bad


def sample_items(items: Sequence[EvaluationItem], n: Optional[int], seed: int = 0) -> list[EvaluationItem]:
    """`n` scenarios chosen at random (all of them, shuffled, if n is None / >= len).

    Shuffle-then-take, so the same seed gives a nested family: the first 100 of a 500-sample
    are exactly the 100-sample. Re-running with a bigger --n-scenarios therefore keeps and
    extends earlier results instead of starting a different sample.
    """
    pool = list(items)
    random.Random(seed).shuffle(pool)
    return pool if n is None or n >= len(pool) else pool[:max(0, n)]


# --------------------------------------------------------------------------- #
# Results file                                                                #
# --------------------------------------------------------------------------- #

def row_key(row: dict) -> tuple:
    return (row["condition"], row["item_id"], (row.get("label") or {}).get("value"))


def load_rows(path: Path) -> list[dict]:
    rows = []
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass        # a half-written last line after a hard kill
    return rows


# --------------------------------------------------------------------------- #
# Summary: mean identity-bias gap per condition and round                      #
# --------------------------------------------------------------------------- #

def summarize(rows: Sequence[dict], values: Sequence[str]) -> list[dict]:
    """One line per condition: n items, mean gap after each round, final gap with standard error."""
    if len(values) != 2:
        return []
    a, b = values
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {}).setdefault(r["item_id"], {})[(r.get("label") or {}).get("value")] = r
    out = []
    for cond, items in by.items():
        pairs = [(v[a], v[b]) for v in items.values() if a in v and b in v]
        if not pairs:
            continue
        n_rounds = min(min(len(x["per_round_scores"]), len(y["per_round_scores"])) for x, y in pairs)
        per_round = [mean(x["per_round_scores"][i] - y["per_round_scores"][i] for x, y in pairs)
                     for i in range(n_rounds)]
        finals = [x["final_score"] - y["final_score"] for x, y in pairs]
        se = stdev(finals) / math.sqrt(len(finals)) if len(finals) > 1 else float("nan")
        out.append({"condition": cond, "test": pairs[0][0].get("test"), "n_items": len(pairs),
                    "mean_score_" + a: mean(x["final_score"] for x, _ in pairs),
                    "mean_score_" + b: mean(y["final_score"] for _, y in pairs),
                    "gap_per_round": per_round, "final_gap": mean(finals), "final_gap_se": se})
    out.sort(key=lambda d: d["condition"])
    return out
