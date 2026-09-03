#!/usr/bin/env python3
"""Run the identity-bias judging pipeline over a file of items.

Examples
--------
Offline smoke test (no API key, no cost):

    python scripts/run_experiment.py --items examples/items.jsonl \
        --attribute gender --values woman,man --judges 3 --rounds 2 --dry-run

Real run, single judge:

    python scripts/run_experiment.py --items examples/items.jsonl \
        --attribute gender --values woman,man --judges 1 --out results.jsonl

Each input line is a JSON object: {"item_id", "question", "response", "rubric"?}.
Output is one PanelResult JSON per (item, label) written to --out; when exactly
two --values are given, per-item and mean identity bias gaps are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_panel import DEFAULT_MODEL, EvaluationItem, RunConfig, ScoreScale, run
from identity_panel.analysis import identity_bias_gap, make_label_variants


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True, type=Path,
                   help="JSONL file: {item_id, question, response, rubric?} per line")
    p.add_argument("--attribute", default="gender", help="demographic attribute to vary")
    p.add_argument("--values", default="woman,man",
                   help="comma-separated identity values to compare")
    p.add_argument("--include-control", action="store_true",
                   help="also run an unlabeled variant per item")
    p.add_argument("--judges", type=int, default=1, help="panel size (1, 3, 5, ...)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model for every judge seat")
    p.add_argument("--models", default=None,
                   help="comma-separated model per seat; overrides --judges/--model count")
    p.add_argument("--discussion", action="store_true", help="enable discussion rounds")
    p.add_argument("--rounds", type=int, default=0,
                   help="number of discussion rounds (>0 implies --discussion)")
    p.add_argument("--aggregate", default="mean",
                   choices=["mean", "median", "majority"])
    p.add_argument("--effort", default=None,
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="thinking effort per judge call (default: model default)")
    p.add_argument("--scale", default="1,10", help="min,max integer score scale")
    p.add_argument("--no-peer-reasoning", action="store_true",
                   help="in discussion, show only peer scores, not their reasoning")
    p.add_argument("--out", type=Path, default=Path("results.jsonl"))
    p.add_argument("--dry-run", action="store_true",
                   help="use the offline MockAnthropic client (no API calls)")
    p.add_argument("--mock-seed", type=int, default=0)
    return p.parse_args(argv)


def build_config(a: argparse.Namespace) -> RunConfig:
    lo, hi = (int(x) for x in a.scale.split(","))
    scale = ScoreScale(minimum=lo, maximum=hi)
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    n = len(models) if models else a.judges
    discussion = a.discussion or a.rounds > 0
    rounds = a.rounds or (1 if discussion else 0)

    common = dict(scale=scale, aggregate=a.aggregate, mock_seed=a.mock_seed,
                  show_peer_reasoning=not a.no_peer_reasoning)

    if discussion:
        return RunConfig.deliberating_panel(n, rounds, model=a.model, models=models,
                                            effort=a.effort, **common)
    if n == 1:
        return RunConfig.single(model=models[0] if models else a.model,
                                effort=a.effort, **common)
    return RunConfig.independent_panel(n, model=a.model, models=models,
                                       effort=a.effort, **common)


def load_items(path: Path) -> list[EvaluationItem]:
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            items.append(EvaluationItem.from_dict(json.loads(line)))
    return items


def main(argv=None) -> None:
    a = parse_args(argv)
    config = build_config(a)
    values = [v.strip() for v in a.values.split(",") if v.strip()]
    base_items = load_items(a.items)

    print(f"config: panel_size={config.panel_size} discussion={config.discussion} "
          f"rounds={config.discussion_rounds} aggregate={config.aggregate} "
          f"dry_run={a.dry_run}\n")

    per_item: list[tuple[str, dict[str, "PanelResult"]]] = []
    with a.out.open("w") as fh:
        for base in base_items:
            variants = make_label_variants(base, a.attribute, values,
                                           include_control=a.include_control)
            results: dict = {}
            for v_item in variants:
                res = run(v_item, config, dry_run=a.dry_run)
                fh.write(json.dumps(res.to_dict()) + "\n")
                key = v_item.label.value if v_item.label else "__control__"
                results[key] = res
                rounds = [round(s, 2) for s in res.per_round_scores]
                print(f"[{base.item_id}] {key:>12}: rounds={rounds} "
                      f"final={res.final_score:.2f}")
            per_item.append((base.item_id, results))

    if len(values) == 2:
        print(f"\n=== identity bias gap ({a.attribute}: {values[0]} - {values[1]}) ===")
        gaps = []
        for item_id, results in per_item:
            if values[0] in results and values[1] in results:
                gap = identity_bias_gap(results[values[0]], results[values[1]])
                gaps.append(gap)
                per_round = [round(x, 2) for x in gap.per_round_gap]
                print(f"[{item_id}] per-round={per_round} final={gap.final_gap:+.2f}")
        if gaps:
            mean_final = sum(g.final_gap for g in gaps) / len(gaps)
            print(f"\nmean final gap over {len(gaps)} items: {mean_final:+.3f}")

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
