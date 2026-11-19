"""The full set of experiments, the random scenario sampler, and the summary statistics.

Three tests, eight conditions. Every condition is run on the same scenarios. Each scenario is
shown with ITS OWN author profile from popquorn_biases.jsonl (gender, race, age, occupation,
education), so a scenario is judged once per condition. Bias is measured across scenarios:
e.g. how far scores sit from the gold rating for women vs. men, and how a biased judge moves that.

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
import re
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

# popquorn rows whose question contains unescaped double quotes are not valid JSON, but the
# row layout is fixed, so they can still be read exactly.
_POPQUORN_ROW = re.compile(r'^\{"item_id": "([^"]+)", "question": "(.*)", "response": (-?\d+)\}\s*$', re.DOTALL)


def _recover_row(line: str) -> Optional[dict]:
    m = _POPQUORN_ROW.match(line.strip())
    if not m:
        return None
    try:   # undo ordinary JSON escapes (\n, \u00e9, ...); keep the stray quotes as-is
        question = json.loads('"' + m.group(2).replace('\\"', '\u0022').replace('"', '\\"') + '"')
    except ValueError:
        question = m.group(2)
    return {"item_id": m.group(1), "question": question, "response": int(m.group(3))}


def load_items(path: Path) -> tuple[list[EvaluationItem], list[int]]:
    """All items in the file, plus the 1-based numbers of lines that could not be read.

    Each item's metadata["case_index"] is its 0-based row number in the file: case 0 is the
    first row, case 1 the second, and so on. Rows with unescaped quotes are recovered."""
    items, bad = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            d = _recover_row(line)
        try:
            if d is None:
                raise ValueError("unreadable")
            it = EvaluationItem.from_dict(d)
        except (ValueError, KeyError, TypeError):
            bad.append(n)
            continue
        items.append(replace(it, metadata={**it.metadata, "case_index": n - 1}))
    return items, bad


def select_cases(items: Sequence[EvaluationItem], start: int, end: int) -> list[EvaluationItem]:
    """Items whose case_index is in [start, end] (inclusive), in file order."""
    return [it for it in items if start <= it.metadata.get("case_index", -1) <= end]


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
    """One saved run per (condition, scenario): each scenario has exactly one author profile."""
    return (row["condition"], row["item_id"])


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

def row_profile(row: dict) -> dict:
    """The author profile a saved run was judged with (gender from the label if no profile)."""
    label = row.get("label") or {}
    prof = dict(label.get("profile") or {})
    if not prof and label.get("attribute"):
        prof[label["attribute"]] = label.get("value")
    return prof


def _mean_se(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    return mean(xs), (stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def summarize(rows: Sequence[dict], groups: Sequence[str] = ("woman", "man")) -> list[dict]:
    """One line per condition: n scenarios, mean score after each round, mean final score,
    mean (final score - gold), that error by gender, and the woman - man error gap.

    Different scenarios carry different genders, so the gap compares score - gold (how much
    more/less offensive than the reference rating a text is judged) across groups."""
    a, b = groups
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["item_id"]] = r      # last write wins
    out = []
    for cond, items in by.items():
        rs = [r for r in items.values() if r.get("final_score") == r.get("final_score")]
        if not rs:
            continue
        n_rounds = min(len(r["per_round_scores"]) for r in rs)
        err = {}
        for r in rs:
            g = r.get("gold")
            if isinstance(g, (int, float)):
                err.setdefault(str(row_profile(r).get("gender", "")).lower(), []).append(r["final_score"] - g)
        all_err = [x for xs in err.values() for x in xs]
        gap = gap_se = float("nan")
        if err.get(a) and err.get(b):
            ma, sa = _mean_se(err[a]); mb, sb = _mean_se(err[b])
            gap = ma - mb
            gap_se = math.sqrt(sa ** 2 + sb ** 2) if sa == sa and sb == sb else float("nan")
        d = {"condition": cond, "test": rs[0].get("test"), "n_items": len(rs),
             "mean_score_per_round": [mean(r["per_round_scores"][i] for r in rs) for i in range(n_rounds)],
             "mean_final_score": mean(r["final_score"] for r in rs),
             "mean_score_minus_gold": _mean_se(all_err)[0]}
        for g in sorted(err):
            d[f"n_{g}"] = len(err[g])
            d[f"score_minus_gold_{g}"] = mean(err[g])
        d[f"gap_{a}_minus_{b}"] = gap
        d[f"gap_{a}_minus_{b}_se"] = gap_se
        out.append(d)
    out.sort(key=lambda d: d["condition"])
    return out
