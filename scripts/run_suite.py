#!/usr/bin/env python3
"""Run ALL experiments (3 tests / 8 conditions) on a local open-weights model.

    python scripts/run_suite.py --items examples/popquorn.jsonl --n-scenarios 300 --seed 0 \
        --model Qwen/Qwen2.5-7B-Instruct --out results/results.jsonl

--n-scenarios N   pick N scenarios at random from the file (omit for all of them). The pick is
                  reproducible (--seed) and nested: asking for 500 later keeps the first 300's results.
--out             JSONL, one line per (condition, scenario, identity label): every judge's score and
                  reasoning in every round, the per-round and final group score, and the gold rating.
Re-running the same command resumes: finished (condition, scenario, label) lines are skipped.
--max-runtime-minutes stops cleanly (finishing the runs in flight) so a Kaggle session can end tidily.

Try it without a GPU:  add --dry-run  (mock judges, instant, numbers are meaningless).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_panel import BiasSpec, MockAnthropic, check_judge_bias, run
from identity_panel.analysis import make_label_variants
from identity_panel.client import CachedClient
from identity_panel.suite import (DEFAULT_SCALE, build_conditions, load_items, load_rows, row_key,
                                  sample_items, select_conditions, summarize)

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=Path("examples/popquorn.jsonl"))
    p.add_argument("--n-scenarios", type=int, default=None, help="how many scenarios to pick at random (default: all)")
    p.add_argument("--seed", type=int, default=0, help="seed for the random scenario pick")
    p.add_argument("--attribute", default="gender")
    p.add_argument("--values", default="woman,man", help="the two identity labels compared")
    p.add_argument("--rounds", type=int, default=2, help="discussion rounds in the discussion conditions")
    p.add_argument("--scale", default="1,5")
    p.add_argument("--scale-description", default=DEFAULT_SCALE.description)
    p.add_argument("--aggregate", default="mean", choices=["mean", "median", "majority"])
    p.add_argument("--simultaneous", action="store_true",
                   help="in discussion all judges answer at once (speaking order then has no effect)")

    g = p.add_argument_group("which experiments")
    g.add_argument("--tests", default=None, help="only these tests, e.g. 1,2 (default: 1,2,3)")
    g.add_argument("--conditions", default="all", help="comma list of condition names (default: all)")
    g.add_argument("--bias-favored", default=None, help="scored HIGHER by the biased judge (default: first value)")
    g.add_argument("--bias-disfavored", default=None, help="scored LOWER by the biased judge (default: second value)")
    g.add_argument("--bias-strength", choices=["mild", "strong"], default="strong")
    g.add_argument("--verify-bias", action="store_true",
                   help="only check the biased judge differs from a normal one, then exit")
    g.add_argument("--verify-n", type=int, default=10)
    g.add_argument("--min-gap", type=float, default=1.0)

    m = p.add_argument_group("model / speed")
    m.add_argument("--model", default=DEFAULT_LOCAL_MODEL, help="Hugging Face id or a folder with the weights")
    m.add_argument("--temperature", type=float, default=0.7,
                   help="sampling temperature. Must be > 0, or all judges with the same prompt answer identically")
    m.add_argument("--workers", type=int, default=32, help="panel runs in flight at once (feeds the GPU batches)")
    m.add_argument("--max-batch", type=int, default=16, help="prompts per GPU batch (lower it if you run out of memory)")
    m.add_argument("--max-new-tokens", type=int, default=200)
    m.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    m.add_argument("--load-in-4bit", action="store_true", help="4-bit weights (needs bitsandbytes; less memory, often slower)")
    m.add_argument("--dry-run", action="store_true", help="mock judges: no model, no GPU")
    m.add_argument("--mock-seed", type=int, default=0)

    o = p.add_argument_group("output / control")
    o.add_argument("--out", type=Path, default=Path("results.jsonl"))
    o.add_argument("--cache", type=Path, default=None, help="answer cache (default: <out>.cache.jsonl)")
    o.add_argument("--no-cache", action="store_true")
    o.add_argument("--overwrite", action="store_true", help="start over instead of resuming")
    o.add_argument("--summary-only", action="store_true", help="just summarise an existing --out")
    o.add_argument("--max-runtime-minutes", type=float, default=None)
    o.add_argument("--max-failures", type=int, default=20, help="stop after this many failed runs in a row")
    return p.parse_args(argv)


def make_client(a):
    if a.dry_run:
        return MockAnthropic(seed=a.mock_seed)
    from identity_panel.local_model import LocalHFClient

    print(f"loading {a.model} ...", flush=True)
    return LocalHFClient(a.model, max_batch=a.max_batch, max_new_tokens=a.max_new_tokens,
                         dtype=a.dtype, load_in_4bit=a.load_in_4bit)


def write_summary(rows, values, out: Path):
    summ = summarize(rows, values)
    if not summ:
        print("(nothing to summarise yet)")
        return
    a, b = values
    print(f"\n=== identity bias gap = score({a}) - score({b}) ===")
    print(f"{'condition':<26}{'n':>6}  {'gap after each round':<26}{'final gap':>10}  {'+/- s.e.':>8}")
    for s in summ:
        rnds = "[" + ", ".join(f"{x:+.2f}" for x in s["gap_per_round"]) + "]"
        print(f"{s['condition']:<26}{s['n_items']:>6}  {rnds:<26}{s['final_gap']:>+10.3f}  {s['final_gap_se']:>8.3f}")
    path = out.with_name(out.stem + "_summary.csv")
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "test", "n_items", f"mean_score_{a}", f"mean_score_{b}", "gap_per_round", "final_gap", "final_gap_se"])
        for s in summ:
            w.writerow([s["condition"], s["test"], s["n_items"], round(s[f"mean_score_{a}"], 4),
                        round(s[f"mean_score_{b}"], 4), json.dumps([round(x, 4) for x in s["gap_per_round"]]),
                        round(s["final_gap"], 4), round(s["final_gap_se"], 4)])
    print(f"\nsummary table: {path}")


def main(argv=None) -> int:
    a = parse_args(argv)
    values = [v.strip() for v in a.values.split(",") if v.strip()]
    if len(values) != 2:
        sys.exit("error: --values needs exactly two labels, e.g. woman,man")
    if a.temperature <= 0 and not a.verify_bias:
        sys.exit("error: --temperature must be > 0 (otherwise judges with the same prompt give identical answers)")
    lo, hi = (int(x) for x in a.scale.split(","))
    from identity_panel import ScoreScale

    scale = ScoreScale(lo, hi, a.scale_description)
    bias = BiasSpec(a.attribute, a.bias_favored or values[0], a.bias_disfavored or values[1], a.bias_strength)

    if a.summary_only:
        write_summary(load_rows(a.out), values, a.out)
        return 0

    all_items, bad = load_items(a.items)
    if bad:
        print(f"note: skipped {len(bad)} unreadable line(s) in {a.items.name}: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    items = sample_items(all_items, a.n_scenarios, a.seed)
    print(f"scenarios: {len(items)} of {len(all_items)} (seed {a.seed})")

    conditions = select_conditions(
        build_conditions(bias, rounds=a.rounds, model=a.model, scale=scale, temperature=a.temperature,
                         aggregate=a.aggregate, sequential_turns=not a.simultaneous, mock_seed=a.mock_seed),
        a.conditions, a.tests)

    if a.verify_bias:
        client = make_client(a)
        chk = check_judge_bias(items[:a.verify_n], bias, client=client, dry_run=a.dry_run, model=a.model,
                               scale=scale, min_gap=a.min_gap)
        print(f"normal judge gap {chk.normal_mean_gap:+.2f} | biased judge gap {chk.biased_mean_gap:+.2f} | "
              f"separation {chk.separation:+.2f} (need >= {a.min_gap}) | identity mentioned in reasoning: {chk.leaks}")
        for e in chk.errors:
            print("  ERROR:", e)
        for ex in chk.example_reasoning:
            print("  sample biased reasoning:", ex)
        print("PASS" if chk.passed else ("INCONCLUSIVE" if chk.separation != chk.separation else "FAIL"))
        return 0 if chk.passed else 1

    if a.overwrite and a.out.exists():
        a.out.unlink()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.out)
    done = {row_key(r) for r in rows}

    tasks = []                                     # item-major: a partial run has every condition for each item
    for item in items:
        for v_item in make_label_variants(item, a.attribute, values):
            for cond in conditions:
                if (cond.name, item.item_id, v_item.label.value) not in done:
                    tasks.append((cond, item, v_item))
    per_item = sum(c.calls_per_run for c in conditions) * len(values)
    print(f"{len(conditions)} conditions x {len(items)} scenarios x {len(values)} labels = "
          f"{len(conditions) * len(items) * len(values)} panel runs ({len(done)} done, {len(tasks)} to do)")
    print(f"<= {per_item} model calls per scenario ({per_item * len(items)} total; fewer with cache sharing)")
    for c in conditions:
        order = [f"{j.judge_id}{'*' if j.bias_tag else ''}" for j in c.config.judges]
        print(f"  {c.name:<26}{c.description}   order={order}")
    if not tasks:
        write_summary(rows, values, a.out)
        return 0

    client = make_client(a)
    if not a.no_cache:
        client = CachedClient(client, a.cache or a.out.with_suffix(".cache.jsonl"))
    if a.dry_run:
        a.workers = min(a.workers, 8)

    lock = threading.Lock()
    started = time.monotonic()
    deadline = started + a.max_runtime_minutes * 60 if a.max_runtime_minutes else None
    state = {"ok": 0, "failed": 0, "consec": 0, "why": None, "last_print": started}

    def work(task):
        cond, item, v_item = task
        return run(v_item, cond.config, client=client)

    def handle(task, fut, fh):
        cond, item, v_item = task
        label = v_item.label.value
        try:
            res = fut.result()
            errs = [e.error for rd in res.rounds for e in rd.evaluations if e.error]
            bad = bool(errs) or res.final_score != res.final_score
        except Exception as exc:  # noqa: BLE001
            errs, bad = [f"{type(exc).__name__}: {exc}"], True
        if bad:
            state["failed"] += 1
            state["consec"] += 1
            print(f"FAILED {cond.name} {item.item_id} {label}: {errs[0] if errs else 'no score'} (not saved; rerun to retry)")
            if state["consec"] >= a.max_failures:
                state["why"] = f"{a.max_failures} failed runs in a row"
            return
        state["consec"] = 0
        row = res.to_dict()
        row.update(condition=cond.name, test=cond.test, gold=item.metadata.get("gold"), seed=a.seed)
        with lock:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
        state["ok"] += 1

    def progress():
        now = time.monotonic()
        if now - state["last_print"] < 60:
            return
        state["last_print"] = now
        rate = state["ok"] / ((now - started) / 60)
        left = len(tasks) - state["ok"] - state["failed"]
        eta = f"{left / rate / 60:.1f} h" if rate > 0 else "?"
        stats = getattr(getattr(client, "inner", client), "stats", None)
        print(f"[{(now - started) / 60:.0f} min] {state['ok']}/{len(tasks)} runs saved, {rate:.1f} runs/min, "
              f"ETA {eta}" + (f", gpu {stats}" if stats else "") + (f", cache hits {client.hits}" if hasattr(client, "hits") else ""),
              flush=True)

    try:
        with a.out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
            pending, it = {}, iter(tasks)
            exhausted = False
            while True:
                while not exhausted and not state["why"] and len(pending) < a.workers:
                    if deadline and time.monotonic() > deadline:
                        state["why"] = "time limit (--max-runtime-minutes)"
                        break
                    try:
                        t = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[ex.submit(work, t)] = t
                if not pending:
                    break
                finished, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for f in finished:
                    handle(pending.pop(f), f, fh)
                progress()
    except KeyboardInterrupt:
        state["why"] = "interrupted"
        print("\ninterrupted - progress so far is saved")
    finally:
        if hasattr(getattr(client, "inner", client), "close"):
            getattr(client, "inner", client).close()

    print(f"\nsaved {state['ok']} runs this session ({state['failed']} failed) -> {a.out}")
    write_summary(rows, values, a.out)
    if state["why"] and not state["why"].startswith("time limit"):
        print(f"\nSTOPPED EARLY: {state['why']}. Rerun the same command to continue.")
        return 2
    if state["why"]:
        print(f"\nstopped cleanly: {state['why']}. Rerun the same command to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
