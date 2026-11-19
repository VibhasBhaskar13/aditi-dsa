#!/usr/bin/env python3
"""Run ALL experiments (3 tests / 8 conditions) on a local open-weights model.

    python scripts/run_suite.py --model Qwen/Qwen2.5-7B-Instruct --cases 0-2000

--cases START-END  which test cases to run: 0-based row numbers of popquorn.jsonl, both ends
                   included (case 0 = first row). Omit for every case. Lets several people run
                   at once: e.g. one person 0-2000 and another 2001-4000 for the same model.
--biases FILE      popquorn_biases.jsonl: the author profile (gender, race, age, occupation,
                   education) shown to the judges for each case. Every case is shown with ALL
                   of its labels.
Output             one set of files per model and case range, so runs never overwrite each other:
                     results/results_7b_cases_0-2000.jsonl   every judge's score + reasoning, every round
                     results/results_7b_cases_0-2000.xlsx    the same as an Excel workbook
                     results/results_7b_cases_0-2000_summary.csv
                     results/results_7b_cases_0-2000.cache.jsonl   (needed to resume - keep it)
Re-running the SAME command resumes where it stopped (finished cases are skipped).
--max-runtime-minutes stops cleanly (finishing the runs in flight) so a Kaggle session can end tidily.

Try it without a GPU: add --dry-run (mock judges, instant, numbers are meaningless).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from identity_panel import BiasSpec, MockAnthropic, check_judge_bias, run
from identity_panel.client import CachedClient
from identity_panel.demographics import (DEMOGRAPHIC_FIELDS, MAIN_ATTRIBUTE, default_out_name, load_profiles,
                                         model_size_tag, parse_cases, profile_label)
from identity_panel.suite import (DEFAULT_SCALE, build_conditions, load_items, load_rows, row_key, row_profile,
                                  select_cases, select_conditions, summarize)

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=ROOT / "examples" / "popquorn.jsonl")
    p.add_argument("--biases", type=Path, default=None,
                   help="author profile per case (default: popquorn_biases.jsonl next to --items)")
    p.add_argument("--cases", default=None,
                   help="case range START-END, 0-based rows of --items, inclusive (e.g. 0-2000). Default: all")
    p.add_argument("--seed", type=int, default=0, help="recorded with each result (cases are no longer sampled)")
    p.add_argument("--rounds", type=int, default=2, help="discussion rounds in the discussion conditions")
    p.add_argument("--scale", default="1,5")
    p.add_argument("--scale-description", default=DEFAULT_SCALE.description)
    p.add_argument("--aggregate", default="mean", choices=["mean", "median", "majority"])
    p.add_argument("--simultaneous", action="store_true",
                   help="in discussion all judges answer at once (speaking order then has no effect)")

    g = p.add_argument_group("which experiments")
    g.add_argument("--tests", default=None, help="only these tests, e.g. 1,2 (default: 1,2,3)")
    g.add_argument("--conditions", default="all", help="comma list of condition names (default: all)")
    g.add_argument("--bias-favored", default="woman", help="gender scored HIGHER by the biased judge")
    g.add_argument("--bias-disfavored", default="man", help="gender scored LOWER by the biased judge")
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
    o.add_argument("--out-dir", type=Path, default=Path("results"),
                   help="folder for the results (file names are chosen from --model and --cases)")
    o.add_argument("--out", type=Path, default=None, help="explicit results file (overrides --out-dir naming)")
    o.add_argument("--cache", type=Path, default=None, help="answer cache (default: <out>.cache.jsonl)")
    o.add_argument("--no-cache", action="store_true")
    o.add_argument("--no-excel", action="store_true", help="skip writing the .xlsx workbook")
    o.add_argument("--overwrite", action="store_true", help="start over instead of resuming")
    o.add_argument("--summary-only", action="store_true", help="just summarise an existing --out (and rewrite the .xlsx)")
    o.add_argument("--max-runtime-minutes", type=float, default=None)
    o.add_argument("--max-failures", type=int, default=20, help="stop after this many failed runs in a row")
    return p.parse_args(argv)


def resolve_paths(a, n_cases: int):
    """(start, end, out): the case range and the results file for this model + range."""
    start, end = parse_cases(a.cases, n_cases)
    out = a.out or (a.out_dir / default_out_name(a.model, start, end))
    return start, end, out


def output_files(out: Path) -> dict:
    return {"jsonl": out, "cache": out.with_suffix(".cache.jsonl"),
            "summary": out.with_name(out.stem + "_summary.csv"), "xlsx": out.with_suffix(".xlsx")}


def make_client(a):
    if a.dry_run:
        return MockAnthropic(seed=a.mock_seed)
    from identity_panel.local_model import LocalHFClient

    print(f"loading {a.model} ...", flush=True)
    return LocalHFClient(a.model, max_batch=a.max_batch, max_new_tokens=a.max_new_tokens,
                         dtype=a.dtype, load_in_4bit=a.load_in_4bit)


# --------------------------------------------------------------------------- #
# Output tables: summary CSV and Excel workbook                                #
# --------------------------------------------------------------------------- #

def _fmt(x, nd=4):
    return round(x, nd) if isinstance(x, float) and x == x else ("" if isinstance(x, float) else x)


def write_summary(rows, out: Path, groups=("woman", "man")):
    summ = summarize(rows, groups)
    if not summ:
        print("(nothing to summarise yet)")
        return []
    a, b = groups
    gap = f"gap_{a}_minus_{b}"
    print(f"\n=== mean score, and gender gap in (score - gold) = {a} - {b} ===")
    print(f"{'condition':<26}{'n':>6} {'mean score per round':<26}{'final':>7}{'gap':>9}{'+/- s.e.':>9}")
    for s in summ:
        rnds = "[" + ", ".join(f"{x:.2f}" for x in s["mean_score_per_round"]) + "]"
        print(f"{s['condition']:<26}{s['n_items']:>6} {rnds:<26}{s['mean_final_score']:>7.2f}"
              f"{s[gap]:>+9.3f}{s[gap + '_se']:>9.3f}")
    cols = ["condition", "test", "n_items", "mean_score_per_round", "mean_final_score", "mean_score_minus_gold"]
    cols += sorted({k for s in summ for k in s if k.startswith(("n_", "score_minus_gold_")) and k != "n_items"})
    cols += [gap, gap + "_se"]
    path = output_files(out)["summary"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in summ:
            w.writerow([json.dumps([round(x, 4) for x in s[c]]) if c == "mean_score_per_round" else _fmt(s.get(c, ""))
                        for c in cols])
    print(f"\nsummary table: {path}")
    return summ


def _run_record(r):
    prof = row_profile(r)
    model = (r.get("config") or {}).get("models") or [r.get("model_name")]
    model = model[0] if model else None
    rec = {"case_index": r.get("case_index"), "item_id": r["item_id"], "condition": r.get("condition"),
           "test": r.get("test"), "model": model, "model_size": r.get("model_size") or model_size_tag(model or "")}
    rec.update({f: prof.get(f) for f in DEMOGRAPHIC_FIELDS})
    return rec


def build_tables(rows):
    """(runs, judges): one row per panel run, and one row per judge evaluation (with reasoning)."""
    runs, judges = [], []
    for r in sorted(rows, key=lambda r: (r.get("case_index") or 0, r.get("condition") or "")):
        base = _run_record(r)
        cfg = r.get("config") or {}
        order = cfg.get("judge_order") or []
        biased = set(cfg.get("biased_judge_ids") or [])
        n_fail = 0
        for rd in r.get("rounds") or []:
            for e in rd.get("evaluations") or []:
                n_fail += int(bool(e.get("error")) or e.get("score") is None)
                judges.append({**base, "round_index": rd["round_index"],
                               "phase": "independent" if rd["round_index"] == 0 else "discussion",
                               "speaking_position": order.index(e["judge_id"]) if e["judge_id"] in order else None,
                               "judge_id": e["judge_id"], "is_biased": e["judge_id"] in biased,
                               "judge_model": e.get("model"), "score": e.get("score"),
                               "reasoning": e.get("reasoning"), "error": e.get("error"),
                               "round_aggregate": rd.get("aggregate_score"), "final_score": r.get("final_score"),
                               "gold": r.get("gold")})
        rec = {**base, "panel_size": cfg.get("panel_size"), "discussion": cfg.get("discussion"),
               "gold": r.get("gold"), "final_score": r.get("final_score")}
        for i, s in enumerate(r.get("per_round_scores") or []):
            rec[f"score_round_{i}"] = s
        rec.update(n_failed_evaluations=n_fail, author_label=(r.get("label") or {}).get("rendered"),
                   question=r.get("question"))
        runs.append(rec)
    return runs, judges


EXCEL_MAX_ROWS = 1_000_000          # Excel's hard limit is 1,048,576 rows per sheet


def write_excel(rows, out: Path, summ, info: dict) -> Path:
    import pandas as pd
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    def clean(df):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v)[:32000] if isinstance(v, str) else v)
        return df

    runs, judges = build_tables(rows)
    path = output_files(out)["xlsx"]
    tmp = path.with_name(path.stem + ".tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        pd.DataFrame([{"field": k, "value": v} for k, v in info.items()]).to_excel(xw, sheet_name="info", index=False)
        s = pd.DataFrame(summ)
        if len(s):
            s["mean_score_per_round"] = s["mean_score_per_round"].map(lambda xs: json.dumps([round(x, 4) for x in xs]))
        s.to_excel(xw, sheet_name="summary", index=False)
        clean(pd.DataFrame(runs)).to_excel(xw, sheet_name="runs", index=False)
        jd = clean(pd.DataFrame(judges))
        for k in range(0, max(len(jd), 1), EXCEL_MAX_ROWS):
            name = "judge_evaluations" + (f"_{k // EXCEL_MAX_ROWS + 1}" if len(jd) > EXCEL_MAX_ROWS else "")
            jd.iloc[k:k + EXCEL_MAX_ROWS].to_excel(xw, sheet_name=name, index=False)
    tmp.replace(path)
    print(f"excel workbook: {path} ({len(runs)} runs, {len(judges)} judge evaluations)")
    return path


def write_outputs(rows, out: Path, a, start, end, excel=True):
    summ = write_summary(rows, out, (a.bias_favored, a.bias_disfavored))
    if excel and not a.no_excel:
        try:
            write_excel(rows, out, summ, {
                "model": a.model, "model_size": model_size_tag(a.model), "cases": f"{start}-{end}",
                "items_file": a.items.name, "biases_file": a.biases.name if a.biases else "",
                "panel_runs_saved": len(rows), "written": datetime.now().isoformat(timespec="seconds")})
        except Exception as exc:  # noqa: BLE001 - never lose a session over the spreadsheet
            print(f"WARNING: could not write the Excel file ({type(exc).__name__}: {exc}). "
                  f"The .jsonl has everything; rerun with --summary-only to retry.")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    a = parse_args(argv)
    if a.temperature <= 0 and not a.verify_bias:
        sys.exit("error: --temperature must be > 0 (otherwise judges with the same prompt give identical answers)")
    a.biases = a.biases or a.items.with_name("popquorn_biases.jsonl")
    if not a.biases.exists():
        sys.exit(f"error: author-profile file {a.biases} not found (set --biases)")
    lo, hi = (int(x) for x in a.scale.split(","))
    from identity_panel import ScoreScale

    scale = ScoreScale(lo, hi, a.scale_description)
    bias = BiasSpec(MAIN_ATTRIBUTE, a.bias_favored.lower(), a.bias_disfavored.lower(), a.bias_strength)

    all_items, bad = load_items(a.items)
    if bad:
        print(f"note: {len(bad)} unreadable line(s) in {a.items.name} skipped: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    n_rows = max(it.metadata["case_index"] for it in all_items) + 1
    try:
        start, end, out = resolve_paths(a, n_rows)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    profiles = load_profiles(a.biases)
    items = []
    for it in select_cases(all_items, start, end):
        if it.item_id not in profiles:
            sys.exit(f"error: case {it.metadata['case_index']} ({it.item_id}) has no line in {a.biases.name}")
        items.append(it.with_label(profile_label(profiles[it.item_id])))
    size = model_size_tag(a.model)
    print(f"model: {a.model} ({size}) | cases {start}-{end}: {len(items)} scenarios | results: {out}")

    if a.summary_only:
        write_outputs(load_rows(out), out, a, start, end)
        return 0

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

    files = output_files(out)
    if a.overwrite:
        for f in (files["jsonl"], files["cache"]):
            if f.exists():
                f.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out)
    done = {row_key(r) for r in rows}

    tasks = []                       # case order: a partial run has every condition for each finished case
    for item in items:
        for cond in conditions:
            if (cond.name, item.item_id) not in done:
                tasks.append((cond, item))
    complete = sum(1 for it in items if all((c.name, it.item_id) in done for c in conditions))
    per_item = sum(c.calls_per_run for c in conditions)
    print(f"{len(conditions)} conditions x {len(items)} scenarios = {len(conditions) * len(items)} panel runs "
          f"({len(done)} done, {len(tasks)} to do)")
    if tasks and complete:
        print(f"resuming: {complete} case(s) already complete, picking up at case {tasks[0][1].metadata['case_index']}")
    print(f"<= {per_item} model calls per scenario ({per_item * len(items)} total; fewer with cache sharing)")
    for c in conditions:
        order = [f"{j.judge_id}{'*' if j.bias_tag else ''}" for j in c.config.judges]
        print(f"  {c.name:<26}{c.description}  order={order}")
    if not tasks:
        print("all requested cases are already done.")
        write_outputs(rows, out, a, start, end)
        return 0

    client = make_client(a)
    if not a.no_cache:
        client = CachedClient(client, a.cache or files["cache"])
    if a.dry_run:
        a.workers = min(a.workers, 8)

    lock = threading.Lock()
    started = time.monotonic()
    deadline = started + a.max_runtime_minutes * 60 if a.max_runtime_minutes else None
    state = {"ok": 0, "failed": 0, "consec": 0, "why": None, "last_print": started}

    def work(task):
        cond, item = task
        return run(item, cond.config, client=client)

    def handle(task, fut, fh):
        cond, item = task
        try:
            res = fut.result()
            errs = [e.error for rd in res.rounds for e in rd.evaluations if e.error]
            bad_ = bool(errs) or res.final_score != res.final_score
        except Exception as exc:  # noqa: BLE001
            errs, bad_ = [f"{type(exc).__name__}: {exc}"], True
        case = item.metadata["case_index"]
        if bad_:
            state["failed"] += 1
            state["consec"] += 1
            print(f"FAILED {cond.name} case {case} ({item.item_id}): {errs[0] if errs else 'no score'} "
                  f"(not saved; rerun to retry)")
            if state["consec"] >= a.max_failures:
                state["why"] = f"{a.max_failures} failed runs in a row"
            return
        state["consec"] = 0
        row = res.to_dict()
        row.update(condition=cond.name, test=cond.test, gold=item.metadata.get("gold"), seed=a.seed,
                   case_index=case, model_name=a.model, model_size=size, question=item.question)
        if row.get("label"):   # record the exact author sentence the judges were shown
            row["label"]["rendered"] = item.label.render(inline=not item.response.strip())
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
        with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
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

    print(f"\nsaved {state['ok']} runs this session ({state['failed']} failed) -> {out}")
    done = {row_key(r) for r in rows}
    left = [it for it in items if not all((c.name, it.item_id) in done for c in conditions)]
    print(f"cases complete: {len(items) - len(left)}/{len(items)}"
          + (f"; next run starts at case {left[0].metadata['case_index']}" if left else " - range finished"))
    write_outputs(rows, out, a, start, end)
    if state["why"] and not state["why"].startswith("time limit"):
        print(f"\nSTOPPED EARLY: {state['why']}. Rerun the same command to continue.")
        return 2
    if state["why"]:
        print(f"\nstopped cleanly: {state['why']}. Rerun the same command to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
