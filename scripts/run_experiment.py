#!/usr/bin/env python3
"""Run the identity-bias judging pipeline over a file of items.

Needs OPENROUTER_API_KEY for real runs (models are OpenRouter slugs, e.g.
anthropic/claude-sonnet-4.5). --dry-run uses an offline mock: no key, no cost.

Examples
--------
Plain run, 3 judges, 2 discussion rounds, no biased judge (legacy behaviour):

    python scripts/run_experiment.py --items examples/popquorn.jsonl \
        --attribute gender --values woman,man --judges 3 --rounds 2 \
        --scale 1,5 --scale-description "1 = least offensive, 5 = most offensive"

Step 1 - check the biased judge actually differs from a normal one (single judge, both
labels, favored/disfavored default to the first/second --values):

    python scripts/run_experiment.py --items examples/popquorn.jsonl --verify-bias \
        --attribute gender --values woman,man --scale 1,5 --verify-n 10

Step 2 - the three settings with the SAME biased seat/persona (no_bias -> bias_present
-> bias_first); only presence, then position, changes:

    python scripts/run_experiment.py --items examples/popquorn.jsonl --bias-setting all \
        --attribute gender --values woman,man --judges 3 --rounds 2 --scale 1,5 \
        --scale-description "1 = least offensive, 5 = most offensive"

Each input line is a JSON object: {"item_id", "question", "response"?, "rubric"?}. A
non-string "response" (popquorn's gold rating) is kept as metadata["gold"] and the text
quoted in "question" is what gets judged. Output is one PanelResult JSON per
(setting, item, label) in --out; with exactly two --values the identity bias gaps are
printed per setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_panel import (BIAS_SETTINGS, DEFAULT_MODEL, BiasSpec, EvaluationItem, RunConfig,
                            ScoreScale, check_judge_bias, get_client, make_panel_configs, run)
from identity_panel.analysis import make_label_variants


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", required=True, type=Path,
                   help="JSONL file: {item_id, question, response?, rubric?} per line")
    p.add_argument("--attribute", default="gender", help="demographic attribute to vary")
    p.add_argument("--values", default="woman,man",
                   help="comma-separated identity values to compare")
    p.add_argument("--include-control", action="store_true",
                   help="also run an unlabeled variant per item")
    p.add_argument("--limit", type=int, default=None, help="only use the first N items")
    p.add_argument("--skip-bad-lines", action="store_true",
                   help="skip lines of --items that are not valid JSON instead of stopping")
    p.add_argument("--judges", type=int, default=1, help="panel size (1, 3, 5, ...)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="OpenRouter model slug for every judge seat")
    p.add_argument("--models", default=None,
                   help="comma-separated model per seat; overrides --judges/--model count")
    p.add_argument("--discussion", action="store_true", help="enable discussion rounds")
    p.add_argument("--rounds", type=int, default=0,
                   help="number of discussion rounds (>0 implies --discussion)")
    p.add_argument("--aggregate", default="mean", choices=["mean", "median", "majority"])
    p.add_argument("--effort", default=None,
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="reasoning effort per judge call (default: model default)")
    p.add_argument("--scale", default="1,10", help="min,max integer score scale")
    p.add_argument("--scale-description", default=None,
                   help='what the scale means, e.g. "1 = least offensive, 5 = most offensive"')
    p.add_argument("--no-peer-reasoning", action="store_true",
                   help="in discussion, show only peer scores, not their reasoning")
    p.add_argument("--simultaneous", action="store_true",
                   help="all judges respond at once each round (turn order then has no effect)")

    g = p.add_argument_group("biased judge")
    g.add_argument("--bias-setting", choices=[*BIAS_SETTINGS, "all"], default=None,
                   help="no_bias | bias_present | bias_first | all (run the three in turn). "
                        "Omit for a plain run with no biased-judge machinery.")
    g.add_argument("--bias-favored", default=None,
                   help="value scored HIGHER by the biased judge (default: first --values)")
    g.add_argument("--bias-disfavored", default=None,
                   help="value scored LOWER by the biased judge (default: second --values)")
    g.add_argument("--bias-strength", choices=["mild", "strong"], default="strong")
    g.add_argument("--biased-judge", default=None,
                   help="judge_id of the biased seat (default: the last seat)")
    g.add_argument("--verify-bias", action="store_true",
                   help="only check that the biased judge differs from a normal judge, then exit")
    g.add_argument("--verify-n", type=int, default=10, help="items used by --verify-bias")
    g.add_argument("--min-gap", type=float, default=1.0,
                   help="--verify-bias passes if biased gap and separation >= this")

    p.add_argument("--out", type=Path, default=Path("results.jsonl"),
                   help="raw JSONL results (one line per setting/item/label), written as it goes")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="long-format CSV, one row per judge evaluation (default: <out>.csv)")
    p.add_argument("--gaps-out", type=Path, default=None,
                   help="CSV of identity bias gaps per setting/item/round (default: <out>_gaps.csv)")
    p.add_argument("--resume", action="store_true",
                   help="keep existing --out and skip (setting,item,label) combinations already done")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    p.add_argument("--min-interval", type=float, default=3.5,
                   help="min seconds between API calls (free OpenRouter models: ~20 req/min)")
    p.add_argument("--max-failures", type=int, default=5,
                   help="stop after this many consecutive failed runs (rate/daily limit hit); "
                        "rerun with --resume later")
    p.add_argument("--dry-run", action="store_true",
                   help="use the offline mock client (no API calls, no key)")
    p.add_argument("--mock-seed", type=int, default=0)
    return p.parse_args(argv)


def build_scale(a: argparse.Namespace) -> ScoreScale:
    lo, hi = (int(x) for x in a.scale.split(","))
    if a.scale_description:
        return ScoreScale(minimum=lo, maximum=hi, description=a.scale_description)
    return ScoreScale(minimum=lo, maximum=hi)


def build_config(a: argparse.Namespace) -> RunConfig:
    models = [m.strip() for m in a.models.split(",")] if a.models else None
    n = len(models) if models else a.judges
    discussion = a.discussion or a.rounds > 0
    rounds = a.rounds or (1 if discussion else 0)

    common = dict(scale=build_scale(a), aggregate=a.aggregate, mock_seed=a.mock_seed,
                  show_peer_reasoning=not a.no_peer_reasoning,
                  sequential_turns=not a.simultaneous)

    if discussion:
        return RunConfig.deliberating_panel(n, rounds, model=a.model, models=models,
                                            effort=a.effort, **common)
    if n == 1:
        return RunConfig.single(model=models[0] if models else a.model,
                                effort=a.effort, **common)
    return RunConfig.independent_panel(n, model=a.model, models=models,
                                       effort=a.effort, **common)


def build_bias(a: argparse.Namespace, values: list[str]) -> BiasSpec:
    fav = a.bias_favored or (values[0] if len(values) > 0 else None)
    dis = a.bias_disfavored or (values[1] if len(values) > 1 else None)
    if not fav or not dis:
        sys.exit("error: need --bias-favored and --bias-disfavored (or two --values)")
    return BiasSpec(a.attribute, fav, dis, a.bias_strength)


def load_items(path: Path, limit=None, skip_bad=False) -> list[EvaluationItem]:
    items, bad = [], []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(EvaluationItem.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            bad.append(lineno)
            where = f"col {exc.colno}: ..." + line[max(0, exc.colno - 40): exc.colno + 40] + "..." \
                if isinstance(exc, json.JSONDecodeError) else repr(exc)
            print(f"warning: {path.name} line {lineno} is invalid ({type(exc).__name__}: "
                  f"{getattr(exc, 'msg', exc)}) {where}", file=sys.stderr)
    if bad and not skip_bad:
        sys.exit(f"\nerror: {len(bad)} invalid line(s) in {path.name}: {bad[:10]}"
                 f"{' ...' if len(bad) > 10 else ''}\nFix them (usually an unescaped \" inside the "
                 f"text), or rerun with --skip-bad-lines to ignore them.")
    if bad:
        print(f"skipped {len(bad)} invalid line(s): {bad[:10]}", file=sys.stderr)
    return items[:limit] if limit else items


def do_verify(a, bias: BiasSpec, items, client) -> int:
    chk = check_judge_bias(items[: a.verify_n], bias, client=client, dry_run=a.dry_run, model=a.model,
                           effort=a.effort, scale=build_scale(a), min_gap=a.min_gap)
    print(f"bias check: {bias.attribute}  favored={bias.favored_value!r} "
          f"disfavored={bias.disfavored_value!r}  strength={bias.strength}  n={chk.n_items}")
    print(f"  normal judge mean gap (fav - dis): {chk.normal_mean_gap:+.2f}")
    print(f"  biased judge mean gap (fav - dis): {chk.biased_mean_gap:+.2f}")
    print(f"  separation: {chk.separation:+.2f}  (need >= {chk.min_gap})")
    print(f"  biased reasoning that mentions identity: {chk.leaks}")
    for e in chk.errors:
        print(f"  ERROR seen: {e}")
    for ex in chk.example_reasoning:
        print(f"  sample biased reasoning: {ex}")
    print("PASS" if chk.passed else "FAIL")
    return 0 if chk.passed else 1


CSV_FIELDS = ["setting", "item_id", "attribute", "label_value", "round_index", "phase",
              "speaking_position", "judge_id", "is_biased", "model", "score", "reasoning",
              "error", "round_aggregate", "final_score", "gold", "question", "request_id"]


def _key(row):
    return (row.get("setting"), row["item_id"], (row.get("label") or {}).get("value"))


def load_rows(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[_key(r)] = r
    return rows


def write_raw_csv(rows: dict, path: Path) -> int:
    n = 0
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows.values():
            cfg = r["config"]
            order = cfg.get("judge_order", [])
            biased = set(cfg.get("biased_judge_ids", []))
            label = r.get("label") or {}
            for rd in r["rounds"]:
                for e in rd["evaluations"]:
                    w.writerow({
                        "setting": r.get("setting"), "item_id": r["item_id"],
                        "attribute": label.get("attribute"), "label_value": label.get("value"),
                        "round_index": rd["round_index"],
                        "phase": "independent" if rd["round_index"] == 0 else "discussion",
                        "speaking_position": order.index(e["judge_id"]) if e["judge_id"] in order else "",
                        "judge_id": e["judge_id"], "is_biased": e["judge_id"] in biased,
                        "model": e["model"], "score": e["score"], "reasoning": e["reasoning"],
                        "error": e["error"], "round_aggregate": rd["aggregate_score"],
                        "final_score": r["final_score"], "gold": r.get("gold"),
                        "question": r.get("question"), "request_id": e["request_id"],
                    })
                    n += 1
    return n


def compute_gaps(rows: dict, values: list[str]) -> list[dict]:
    """Per (setting, item, round): score[values[0]] - score[values[1]]."""
    if len(values) != 2:
        return []
    out = []
    for (setting, item_id, val), ra in rows.items():
        if val != values[0] or (setting, item_id, values[1]) not in rows:
            continue
        rb = rows[(setting, item_id, values[1])]
        n = min(len(ra["per_round_scores"]), len(rb["per_round_scores"]))
        for i in range(n):
            a_, b_ = ra["per_round_scores"][i], rb["per_round_scores"][i]
            out.append({"setting": setting, "item_id": item_id, "round_index": i,
                        "is_final": i == n - 1, f"score_{values[0]}": a_,
                        f"score_{values[1]}": b_, "gap": a_ - b_})
    return out


def write_gaps_csv(gaps: list[dict], values: list[str], path: Path) -> None:
    fields = ["setting", "item_id", "round_index", "is_final",
              f"score_{values[0]}", f"score_{values[1]}", "gap"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(gaps)


def summarize(gaps: list[dict], a, values) -> None:
    if not gaps:
        print("\n(no complete item pairs to summarise yet)")
        return
    for setting in dict.fromkeys(g["setting"] for g in gaps):
        gs = [g for g in gaps if g["setting"] == setting and not math.isnan(g["gap"])]
        items = {g["item_id"] for g in gs}
        n_rounds = max(g["round_index"] for g in gs) + 1
        per_round = []
        for r in range(n_rounds):
            xs = [g["gap"] for g in gs if g["round_index"] == r]
            per_round.append(round(sum(xs) / len(xs), 3) if xs else None)
        fin = [g["gap"] for g in gs if g["is_final"]]
        print(f"\n=== [{setting}] {a.attribute}: {values[0]} - {values[1]}  ({len(items)} items) ===")
        print(f"mean gap per round: {per_round}")
        print(f"mean final gap:     {sum(fin) / len(fin):+.3f}" if fin else "mean final gap: n/a")


def main(argv=None) -> int:
    a = parse_args(argv)
    values = [v.strip() for v in a.values.split(",") if v.strip()]
    items = load_items(a.items, a.limit, a.skip_bad_lines)
    client = get_client(dry_run=a.dry_run, mock_seed=a.mock_seed,
                        min_interval=0.0 if a.dry_run else a.min_interval)

    if a.verify_bias:
        return do_verify(a, build_bias(a, values), items, client)

    base = build_config(a)
    if a.bias_setting is None:
        settings = [(None, base)]
    else:
        bias = build_bias(a, values)
        if bias.attribute != a.attribute:
            sys.exit("error: bias attribute must equal --attribute")
        cfgs = make_panel_configs(base, bias, biased_judge_id=a.biased_judge)
        wanted = BIAS_SETTINGS if a.bias_setting == "all" else (a.bias_setting,)
        settings = [(name, cfgs[name]) for name in wanted]
        if not base.discussion:
            print("note: no discussion rounds, so speaking order has no effect "
                  "(bias_present == bias_first in behaviour)\n")

    csv_out = a.csv_out or a.out.with_suffix(".csv")
    gaps_out = a.gaps_out or a.out.with_name(a.out.stem + "_gaps.csv")
    if a.out.exists() and not (a.resume or a.overwrite):
        sys.exit(f"error: {a.out} exists. Use --resume to continue it or --overwrite to replace it.")
    rows = load_rows(a.out) if a.resume else {}

    n_labels = len(values) + (1 if a.include_control else 0)
    calls_per_run = base.panel_size * (1 + base.discussion_rounds)
    total_runs = len(items) * len(settings) * n_labels
    todo_runs = sum(1 for it in items for st, _ in settings
                    for v in ([None] if a.include_control else []) + values
                    if (st, it.item_id, v) not in rows)
    print(f"{len(items)} items x {len(settings)} setting(s) x {n_labels} labels = {total_runs} panel runs "
          f"({len(rows)} already done, {todo_runs} to do)")
    print(f"~{todo_runs * calls_per_run} API calls, at >= {0 if a.dry_run else a.min_interval}s each "
          f"=> ~{todo_runs * calls_per_run * (0 if a.dry_run else a.min_interval) / 3600:.1f} h minimum")
    for st, cfg in settings:
        order = [f"{j.judge_id}{'*' if j.bias_tag else ''}" for j in cfg.judges]
        print(f"  [{st}] order={order} (*=biased) rounds={cfg.discussion_rounds} "
              f"aggregate={cfg.aggregate} model(s)={sorted({j.model for j in cfg.judges})}")
    print()

    consecutive_fail, stopped = 0, False
    mode = "a" if a.resume else "w"
    try:
        with a.out.open(mode, encoding="utf-8") as fh:
            # item-major: finish every setting for one item before moving on, so an
            # interrupted / rate-limited run still has balanced data across settings.
            for base_item in items:
                variants = make_label_variants(base_item, a.attribute, values,
                                               include_control=a.include_control)
                for setting, config in settings:
                    for v_item in variants:
                        val = v_item.label.value if v_item.label else None
                        if (setting, base_item.item_id, val) in rows:
                            continue
                        res = run(v_item, config, client=client)
                        errs = [e.error for rd in res.rounds for e in rd.evaluations if e.error]
                        tag = f"[{setting or '-'}] [{base_item.item_id}] {val or '__control__':>10}"
                        if errs or math.isnan(res.final_score):
                            consecutive_fail += 1
                            print(f"{tag}: FAILED ({len(errs)} errors, not saved; first: {errs[0] if errs else 'nan'})")
                            if consecutive_fail >= a.max_failures:
                                stopped = True
                                break
                            continue
                        consecutive_fail = 0
                        row = res.to_dict()
                        row.update(setting=setting, gold=base_item.metadata.get("gold"),
                                   question=base_item.question)
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        rows[_key(row)] = row
                        print(f"{tag}: rounds={[round(x, 2) for x in res.per_round_scores]} "
                              f"final={res.final_score:.2f}")
                    if stopped:
                        break
                if stopped:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted - saving what we have")
        stopped = True
    finally:
        n = write_raw_csv(rows, csv_out)
        gaps = compute_gaps(rows, values)
        if gaps:
            write_gaps_csv(gaps, values, gaps_out)

    summarize(gaps, a, values)
    print(f"\nwrote {a.out} ({len(rows)} panel runs)\n      {csv_out} ({n} evaluation rows)"
          + (f"\n      {gaps_out}" if gaps else ""))
    if stopped:
        print(f"\nSTOPPED EARLY. Rerun the same command with --resume to continue "
              f"(likely a rate/daily limit or bad key/model).")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
