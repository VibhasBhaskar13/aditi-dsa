#!/usr/bin/env python3
"""
Analysis of LLM-judge panel experiments (single judge / independent panel /
deliberating panel / panel containing a biased judge), for multiple model sizes.

Usage
-----
Put every results file into the folder that holds this script, WITHOUT renaming them
(e.g. results_7b_cases_0-2000.jsonl, results_7b_cases_2001-4000.jsonl,
results_3b_cases_0-2000.jsonl, ...), then run:

    python analysis.py                      # reads results_*_cases_*.jsonl next to this script
    python analysis.py --results-dir some/folder --outdir out

* One file = one model + one range of test cases. Files of the same model are combined; the
  model of every row comes from the file's config (the `model` column), not from file order.
* The run prints which cases each model has, flags gaps/overlaps between people's ranges,
  and checks that every model saw exactly the same author labels for each case.
* Every table is written to <outdir>/tables/*.csv and every figure to <outdir>/figures/*.png.

Data design (from the real rows)
--------------------------------
* Scores are 1-5 OFFENSIVENESS ratings (higher = judged more offensive); `gold` is an
  ordinal 1-5 reference rating per item (same for both labels of an item).
* Every item appears ONCE per condition, with its own author profile from
  popquorn_biases.jsonl (gender, race, age, occupation, education). Different items carry
  different profiles, so group differences are measured on score MINUS GOLD (how much more
  or less offensive than the reference rating a text is judged), which removes most of the
  difference in content between groups. The biased judge is told to adjust for gender
  (woman scored higher, man lower), so its effect shows up as a woman-vs-man difference
  in the paired bias effect.
* `test` is the experiment group id (1 = no biased judge, 2 = biased judge last, ...),
  not a train/test flag.

Key design choices
------------------
* The unit of analysis is the ITEM. Seeds are repeated measures of the same item, so
  seeds are averaged within item first and all confidence intervals come from a
  bootstrap over items (not over rows). Treating every (item, seed) row as independent
  would give CIs that are too narrow.
* Bias effect is a PAIRED difference: the score of a panel containing a biased judge
  minus the score of the matching panel without one (same model, item, seed,
  panel_size, discussion). This is far more accurate than comparing group means.
* "Movement toward the biased judge" uses the biased judge's *initial* score as a
  fixed reference (the old version used its final score, which itself moves).
* Parse failures (score None / error) are reported so they cannot silently bias means.
"""
import argparse
import glob
import json
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

warnings.filterwarnings("ignore", category=RuntimeWarning)
RNG = np.random.default_rng(0)
DPI = 150
DEMOGRAPHICS = ["gender", "race", "age", "occupation", "education"]
RESULT_PATTERN = "results_*_cases_*.jsonl"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ----------------------------------------------------------------------------
# Loading / flattening
# ----------------------------------------------------------------------------
def load_jsonl(path):
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  WARNING: {bad} malformed lines skipped in {path}")
    return rows


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def model_tag(cfg):
    """'Qwen/Qwen2.5-3B-Instruct' -> '3B'. Mixed-model panels -> 'mixed(3B+7B)'."""
    tags = set()
    for m in cfg.get("models") or []:
        found = re.findall(r"(\d+(?:\.\d+)?)\s*[Bb]", str(m))
        tags.add(f"{found[-1]}B" if found else str(m))
    if not tags:
        return "unknown"
    return tags.pop() if len(tags) == 1 else "mixed(" + "+".join(sorted(tags)) + ")"


def model_sort_key(tag):
    m = re.match(r"[\d.]+", tag)
    return (float(m.group()) if m else 1e9, tag)


def derive_condition(cfg):
    n = cfg.get("panel_size", 1)
    if n == 1:
        return "single"
    parts = [str(n), "discussion" if cfg.get("discussion") else "independent"]
    biased = cfg.get("biased_judge_ids") or []
    order = cfg.get("judge_order") or []
    if biased:
        parts.append("biased_first" if order and order[0] in biased else "biased")
    return "_".join(parts)


def parse_rows(rows):
    """Returns (runs, judges, rounds) DataFrames. run_id links them."""
    runs, judges, rounds_ = [], [], []
    for run_id, row in enumerate(rows):
        cfg = row.get("config") or {}
        label = row.get("label") or {}
        biased = list(cfg.get("biased_judge_ids") or [])
        order = list(cfg.get("judge_order") or [])
        bias_present = cfg.get("bias_present")
        bias_present = bool(bias_present) if bias_present is not None else len(biased) > 0

        n_eval = n_fail = 0
        for rd in row.get("rounds") or []:
            for ev in rd.get("evaluations") or []:
                score = to_float(ev.get("score"))
                fail = np.isnan(score) or bool(ev.get("error"))
                n_eval += 1
                n_fail += int(fail)
                jid = ev.get("judge_id")
                judges.append(dict(
                    run_id=run_id, round=rd.get("round_index", 0), judge_id=jid,
                    position=order.index(jid) if jid in order else np.nan,
                    is_biased=jid in biased, score=score, failed=fail))

        prs = row.get("per_round_scores")
        if not prs:
            prs = [rd.get("aggregate_score") for rd in row.get("rounds") or []]
        for i, s in enumerate(prs):
            rounds_.append(dict(run_id=run_id, round=i, score=to_float(s)))

        attr, val = label.get("attribute"), label.get("value")
        prof = dict(label.get("profile") or ({attr: val} if attr else {}))
        runs.append(dict(
            run_id=run_id, model=row.get("model_size") or model_tag(cfg), item_id=row.get("item_id"),
            case_index=row.get("case_index"), source_file=row.get("_file"),
            **{f: prof.get(f) for f in DEMOGRAPHICS},
            seed=row.get("seed", 0),
            condition=row.get("condition") or derive_condition(cfg),
            panel_size=cfg.get("panel_size", 1), discussion=bool(cfg.get("discussion")),
            n_rounds=len(row.get("rounds") or []), bias_present=bias_present,
            n_biased=len(biased), bias_first=bool(order and order[0] in biased),
            label=val if attr is not None else None,
            gold=to_float(row.get("gold")), test=row.get("test"),
            final_score=to_float(row.get("final_score")),
            n_eval=n_eval, n_fail=n_fail))
    runs = pd.DataFrame(runs)
    runs["error_vs_gold"] = runs["final_score"] - runs["gold"]
    return runs, pd.DataFrame(judges), pd.DataFrame(rounds_)


def find_result_files(folder):
    """Every results_<model>_cases_<a>-<b>.jsonl in `folder` (answer caches excluded)."""
    files = sorted(p for p in glob.glob(os.path.join(folder, RESULT_PATTERN))
                   if not p.endswith(".cache.jsonl"))
    return files


def _ranges(idx):
    """[0,1,2,5,6] -> '0-2, 5-6'."""
    idx = sorted(set(int(i) for i in idx))
    out, start = [], None
    for a, b in zip(idx, idx[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            out.append(f"{start}-{a}" if a != start else str(a))
            start = None
    return ", ".join(out) if out else "(none)"


def load_all(files):
    """Read every file, tag rows with their file, and drop exact repeats of a
    (model, condition, case) - e.g. if two people's ranges overlapped."""
    rows = []
    print("Result files:")
    for path in files:
        rs = load_jsonl(path)
        tags = sorted({r.get("model_size") or model_tag(r.get("config") or {}) for r in rs})
        cases = [r.get("case_index") for r in rs if r.get("case_index") is not None]
        print(f"  {os.path.basename(path)}: {len(rs)} runs, model(s) {tags}, "
              f"cases {min(cases) if cases else '?'}-{max(cases) if cases else '?'}")
        if len(tags) > 1:
            print(f"    WARNING: this file mixes models {tags}")
        for r in rs:
            r["_file"] = os.path.basename(path)
        rows.extend(rs)
    seen, out, dup = {}, [], 0
    for r in rows:
        k = (r.get("model_size") or model_tag(r.get("config") or {}), r.get("condition"), r.get("item_id"))
        if k in seen:
            dup += 1
            out[seen[k]] = r
        else:
            seen[k] = len(out)
            out.append(r)
    if dup:
        print(f"  note: {dup} duplicate (model, condition, case) runs found in overlapping files; kept the last copy")
    return out


def coverage_report(runs):
    """Which cases each model has finished in every condition; flags missing ranges."""
    n_cond = runs.groupby("model")["condition"].nunique()
    rows = []
    for m, g in runs.groupby("model"):
        per_case = g.groupby("case_index")["condition"].nunique()
        complete = per_case[per_case == n_cond[m]].index
        partial = per_case[per_case < n_cond[m]].index
        lo, hi = int(per_case.index.min()), int(per_case.index.max())
        missing = sorted(set(range(lo, hi + 1)) - set(int(i) for i in per_case.index))
        rows.append(dict(model=m, n_conditions=int(n_cond[m]), n_cases_complete=len(complete),
                         complete_cases=_ranges(complete), partial_cases=_ranges(partial),
                         missing_inside_range=_ranges(missing)))
    return pd.DataFrame(rows)


def label_consistency(runs):
    """Sanity check: every model and condition must have shown the SAME author labels for a
    case. Returns the cases where they differ (empty = all consistent)."""
    cols = [c for c in DEMOGRAPHICS if c in runs]
    sig = runs[["item_id", "model", "condition"] + cols].copy()
    sig["profile"] = sig[cols].astype(str).agg(" | ".join, axis=1)
    n = sig.groupby("item_id")["profile"].nunique()
    bad = n[n > 1].index
    return sig[sig.item_id.isin(bad)].drop_duplicates(["item_id", "model", "profile"])


# ----------------------------------------------------------------------------
# Statistics helpers (all bootstrap over ITEMS)
# ----------------------------------------------------------------------------
def boot_means(x, B):
    n = len(x)
    out = []
    for i in range(0, B, 200):
        idx = RNG.integers(0, n, (min(200, B - i), n))
        out.append(x[idx].mean(axis=1))
    return np.concatenate(out)


def boot_mean_ci(x, B=1000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) < 2:
        return x.mean(), np.nan, np.nan
    b = boot_means(x, B)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def summarize(df, by, col, B=1000):
    """Mean of `col` per group, seeds averaged within item first, bootstrap CI over items."""
    d = df.dropna(subset=[col])
    il = d.groupby(by + ["item_id"])[col].mean().reset_index()
    rows = []
    for key, g in il.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        m, lo, hi = boot_mean_ci(g[col].values, B)
        rows.append({**dict(zip(by, key)), "n_items": len(g), "mean": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def parse_failures(runs):
    g = runs.groupby(["model", "condition"])[["n_eval", "n_fail"]].sum()
    g["fail_rate"] = g.n_fail / g.n_eval.clip(lower=1)
    return g.reset_index()


def baseline_scores(runs):
    return summarize(runs, ["model", "condition"], "final_score")


def paired_bias_effect(runs, pair_on, group_cols, B=1000):
    """final_score(panel with biased judge) - final_score(matching panel without)."""
    key = list(pair_on)
    ctrl = (runs[~runs.bias_present].groupby(key)["final_score"].mean()
            .rename("ctrl").reset_index())
    trt = runs[runs.bias_present]
    m = trt.merge(ctrl, on=key, how="inner")
    if m.empty and "seed" in key:
        print("  No pairs when matching on seed; retrying without seed (seed-averaged).")
        return paired_bias_effect(runs, [k for k in key if k != "seed"], group_cols, B)
    m["diff"] = m["final_score"] - m["ctrl"]
    rows = []
    il = m.dropna(subset=["diff"]).groupby(group_cols + ["item_id"])["diff"].mean().reset_index()
    for k, g in il.groupby(group_cols):
        k = k if isinstance(k, tuple) else (k,)
        d = g["diff"].values
        mean, lo, hi = boot_mean_ci(d, B)
        try:
            p = st.wilcoxon(d).pvalue
        except ValueError:
            p = np.nan
        dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
        rows.append({**dict(zip(group_cols, k)), "n_items": len(d), "mean_diff": mean,
                     "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p, "cohens_dz": dz})
    return pd.DataFrame(rows)


def run_dynamics(judges):
    """Per-run: how much judges move between first and last round, consensus, and the
    movement of neutral vs. biased judges (positive shift = higher offensiveness score)."""
    out = []
    j = judges.dropna(subset=["score"])
    for rid, g in j.groupby("run_id"):
        r0, r1 = g["round"].min(), g["round"].max()
        if r0 == r1:
            continue
        first = g[g["round"] == r0].set_index("judge_id")["score"]
        last = g[g["round"] == r1].set_index("judge_id")["score"]
        common = first.index.intersection(last.index)
        if len(common) == 0:
            continue
        f0, f1 = first[common], last[common]
        d = f1 - f0
        rec = dict(run_id=rid, abs_shift=d.abs().mean(), signed_shift=d.mean(),
                   frac_changed=(d != 0).mean())
        if len(common) > 1:
            rec["disagreement_start"] = f0.std(ddof=0)
            rec["disagreement_end"] = f1.std(ddof=0)
            rec["consensus_gain"] = rec["disagreement_start"] - rec["disagreement_end"]
        flags = g.groupby("judge_id")["is_biased"].first()
        b = [x for x in common if flags[x]]
        n = [x for x in common if not flags[x]]
        if b and n:
            # The biased judge also moves during discussion, so track BOTH groups.
            rec["neutral_shift"] = f1[n].mean() - f0[n].mean()
            rec["biased_shift"] = f1[b].mean() - f0[b].mean()
            rec["gap_start"] = f0[b].mean() - f0[n].mean()      # biased minus neutral
            rec["gap_end"] = f1[b].mean() - f1[n].mean()
        out.append(rec)
    return pd.DataFrame(out)


def accuracy_table(runs, B=300):
    """Agreement of mean final score with gold label (per model x condition).
    AUC = P(score of gold-max item > score of gold-min item); only if gold is binary."""
    rows = []
    for (model, cond), g in runs.groupby(["model", "condition"]):
        il = g.groupby("item_id").agg(score=("final_score", "mean"), gold=("gold", "first")).dropna()
        if len(il) < 10 or il.gold.nunique() < 2:
            continue
        binary = il.gold.nunique() == 2

        def stats(d):
            sp = st.spearmanr(d.score, d.gold)[0]
            auc = np.nan
            if binary:
                hi, lo = d.gold.max(), d.gold.min()
                p, n = d.score[d.gold == hi], d.score[d.gold == lo]
                if len(p) and len(n):
                    auc = st.mannwhitneyu(p, n).statistic / (len(p) * len(n))
            return sp, auc

        sp, auc = stats(il)
        bs = np.array([stats(il.iloc[RNG.integers(0, len(il), len(il))]) for _ in range(B)])
        rows.append(dict(model=model, condition=cond, n_items=len(il), spearman=sp,
                         mae=(il.score - il.gold).abs().mean(), mean_error=(il.score - il.gold).mean(),
                         spearman_lo=np.nanpercentile(bs[:, 0], 2.5),
                         spearman_hi=np.nanpercentile(bs[:, 0], 97.5),
                         auc=auc, auc_lo=np.nanpercentile(bs[:, 1], 2.5) if binary else np.nan,
                         auc_hi=np.nanpercentile(bs[:, 1], 97.5) if binary else np.nan))
    return pd.DataFrame(rows)


def seed_variability(runs):
    # replicates = different seeds for the SAME item and label (labels are NOT replicates)
    g = runs.groupby(["model", "condition", "item_id", "label"], dropna=False)["final_score"].agg(
        ["mean", "std", "count"])
    g = g[g["count"] >= 2].reset_index().rename(columns={"std": "sd"})
    if g.empty:
        print("  Seed variability: no item/label has >=2 seeds, so it cannot be estimated.")
        return g, pd.DataFrame(columns=["model", "condition", "n_items", "mean_within_item_sd", "icc"])
    rows = []
    for (model, cond), d in g.groupby(["model", "condition"]):
        within = (d["sd"] ** 2).mean()
        k = d["count"].mean()
        between = max(d["mean"].var(ddof=1) - within / k, 0)
        rows.append(dict(model=model, condition=cond, n_items=len(d), mean_within_item_sd=d["sd"].mean(),
                         icc=between / (between + within) if between + within > 0 else np.nan))
    return g, pd.DataFrame(rows)


def _boot_diff_ci(x, y, B=1000):
    """Mean(x) - mean(y) for two independent groups of items, bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return (x.mean() - y.mean()) if len(x) and len(y) else np.nan, np.nan, np.nan
    b = boot_means(x, B) - boot_means(y, B)
    return x.mean() - y.mean(), *np.percentile(b, [2.5, 97.5])


def demographic_table(runs, attr, value="error_vs_gold", B=1000):
    """Per model x condition x level of `attr`: n items, mean final score, mean score - gold
    (95% CI over items), and a Kruskal-Wallis test across levels."""
    rows = []
    d = runs.dropna(subset=[attr, value])
    for (m, c), g in d.groupby(["model", "condition"]):
        groups = {lvl: gg[value].values for lvl, gg in g.groupby(attr)}
        try:
            kw = st.kruskal(*[v for v in groups.values() if len(v) > 0]).pvalue if len(groups) > 1 else np.nan
        except ValueError:
            kw = np.nan
        for lvl, gg in g.groupby(attr):
            mean, lo, hi = boot_mean_ci(gg[value].values, B)
            rows.append(dict(model=m, condition=c, attribute=attr, level=lvl, n_items=len(gg),
                             mean_final_score=gg["final_score"].mean(), mean_score_minus_gold=mean,
                             ci_lo=lo, ci_hi=hi, kruskal_p_across_levels=kw))
    return pd.DataFrame(rows)


def gender_gap(runs, a="woman", b="man", B=1000):
    """Per model x condition: mean(score - gold | a) - mean(score - gold | b), bootstrap CI over
    items, Mann-Whitney p. Positive = texts by `a` judged relatively MORE offensive."""
    rows = []
    for (m, c), g in runs.groupby(["model", "condition"]):
        x = g.loc[g.gender == a, "error_vs_gold"].values
        y = g.loc[g.gender == b, "error_vs_gold"].values
        mean, lo, hi = _boot_diff_ci(x, y, B)
        try:
            p = st.mannwhitneyu(x[~np.isnan(x)], y[~np.isnan(y)]).pvalue
        except ValueError:
            p = np.nan
        rows.append(dict(model=m, condition=c, gap_definition=f"{a} - {b} (score - gold)",
                         n_items_a=len(x), n_items_b=len(y), mean_gap=mean, ci_lo=lo, ci_hi=hi,
                         mannwhitney_p=p))
    return pd.DataFrame(rows)


def gender_gap_shift(runs, pair_on, a="woman", b="man", B=1000):
    """How much the biased judge widens the gender gap: per item, paired bias effect
    (panel with biased judge - matched panel without), then mean for `a` minus mean for `b`."""
    key = list(pair_on)
    ctrl = runs[~runs.bias_present].groupby(key)["final_score"].mean().rename("ctrl").reset_index()
    m = runs[runs.bias_present].merge(ctrl, on=key, how="inner")
    m["diff"] = m["final_score"] - m["ctrl"]
    rows = []
    for (mod, c), g in m.groupby(["model", "condition"]):
        x, y = g.loc[g.gender == a, "diff"].values, g.loc[g.gender == b, "diff"].values
        mean, lo, hi = _boot_diff_ci(x, y, B)
        rows.append(dict(model=mod, condition=c, definition=f"bias effect on {a} - bias effect on {b}",
                         n_items_a=len(x), n_items_b=len(y), mean_bias_effect_a=np.nanmean(x) if len(x) else np.nan,
                         mean_bias_effect_b=np.nanmean(y) if len(y) else np.nan, mean_diff=mean, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def model_comparison(runs, ref, other):
    """Per-item paired difference (other - ref) in mean final score, per condition."""
    il = runs.groupby(["model", "condition", "item_id"])["final_score"].mean().reset_index()
    a = il[il.model == ref].drop(columns="model").rename(columns={"final_score": "ref"})
    b = il[il.model == other].drop(columns="model").rename(columns={"final_score": "other"})
    m = a.merge(b, on=["condition", "item_id"])
    m["diff"] = m["other"] - m["ref"]
    rows = []
    for c, g in m.groupby("condition"):
        mean, lo, hi = boot_mean_ci(g["diff"].values)
        r = st.pearsonr(g.ref, g.other)[0] if len(g) > 2 and g.ref.std() > 0 and g.other.std() > 0 else np.nan
        rows.append(dict(condition=c, n_items=len(g), mean_diff=mean, ci_lo=lo, ci_hi=hi, pearson_r=r))
    return m, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def colors_for(models):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def grouped_bars(ax, tab, val, lo, hi, models, cmap, conds):
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        t = tab[tab.model == m].set_index("condition").reindex(conds)
        x = np.arange(len(conds)) + (i - (len(models) - 1) / 2) * w
        y = t[val].values
        err = None
        if lo in t and hi in t:
            err = np.vstack([np.nan_to_num(y - t[lo].values), np.nan_to_num(t[hi].values - y)])
            err = np.clip(err, 0, None)
        ax.bar(x, np.nan_to_num(y), w * 0.95, yerr=err, capsize=3, color=cmap[m], label=m)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")


def safe(fn):
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # keep going if one figure lacks the data it needs
            print(f"  skipped {fn.__name__}: {type(e).__name__}: {e}")
    return wrapper


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
@safe
def fig_baseline(tab, models, cmap, out):
    conds = sorted(tab.condition.unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(conds) + 3, 4.5))
    grouped_bars(ax, tab, "mean", "ci_lo", "ci_hi", models, cmap, conds)
    ax.set_ylabel("Mean final score (95% CI over items)")
    ax.set_title("Mean score by condition and model")
    ax.legend(title="Model", frameon=False)
    savefig(fig, f"{out}/01_mean_score_by_condition.png")


@safe
def fig_bias_effect(eff, models, cmap, out):
    conds = sorted(eff.condition.unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(conds) + 3, 4.5))
    grouped_bars(ax, eff.rename(columns={"mean_diff": "v"}), "v", "ci_lo", "ci_hi", models, cmap, conds)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Score change vs. matched\nunbiased panel")
    ax.set_title("Paired bias effect (95% CI over items)")
    ax.legend(title="Model", frameon=False)
    savefig(fig, f"{out}/02_paired_bias_effect.png")


@safe
def fig_trajectory(tab, models, cmap, out):
    conds = sorted(tab.condition.unique())
    ccol = {c: plt.cm.tab10(i % 10) for i, c in enumerate(conds)}
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for c in conds:
            t = tab[(tab.model == m) & (tab.condition == c)].sort_values("round")
            if t.empty:
                continue
            ls = "--" if "biased" in c else "-"
            ax.plot(t["round"], t["mean"], ls, marker="o", color=ccol[c], label=c)
            ax.fill_between(t["round"], t.ci_lo, t.ci_hi, color=ccol[c], alpha=0.15)
        ax.set_title(f"Qwen2.5-{m}" if m[0].isdigit() else m)
        ax.set_xlabel("Discussion round")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Mean aggregate score")
    axes[0][-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig, f"{out}/03_per_round_trajectory.png")


@safe
def fig_position(tab, models, out):
    conds = sorted(tab.condition.unique())
    ccol = {c: plt.cm.tab10(i % 10) for i, c in enumerate(conds)}
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for c in conds:
            t = tab[(tab.model == m) & (tab.condition == c)].sort_values("position")
            if t.empty:
                continue
            ls = "--" if "biased" in c else "-"
            ax.errorbar(t.position + 1, t["mean"], yerr=[t["mean"] - t.ci_lo, t.ci_hi - t["mean"]],
                        fmt=ls, marker="o", color=ccol[c], label=c, capsize=2)
        ax.set_title(f"{m}: round-0 score by speaking position")
        ax.set_xlabel("Judge position in turn order")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Mean round-0 score")
    axes[0][-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig, f"{out}/04_round0_score_by_position.png")


@safe
def fig_dynamics(dyn, models, cmap, out):
    metrics = [("abs_shift", "Mean |score change| per judge"),
               ("frac_changed", "Share of judges who changed score"),
               ("consensus_gain", "Disagreement reduction (SD start - end)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        tab = summarize(dyn, ["model", "condition"], col)
        conds = sorted(tab.condition.unique())
        grouped_bars(ax, tab, "mean", "ci_lo", "ci_hi", models, cmap, conds)
        ax.set_title(title, fontsize=10)
    axes[0].legend(title="Model", frameon=False)
    savefig(fig, f"{out}/05_deliberation_dynamics.png")


@safe
def fig_conformity(dyn, models, cmap, out):
    d = dyn.dropna(subset=["neutral_shift"])
    cols = [("neutral_shift", "Neutral judges: score change\n(first -> last round)"),
            ("biased_shift", "Biased judge: score change\n(first -> last round)"),
            ("gap_end", "Biased minus neutral score\nat last round")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, title) in zip(axes, cols):
        tab = summarize(d, ["model", "condition"], col)
        conds = sorted(tab.condition.unique())
        grouped_bars(ax, tab, "mean", "ci_lo", "ci_hi", models, cmap, conds)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
    axes[0].legend(title="Model", frameon=False)
    savefig(fig, f"{out}/06_neutral_vs_biased_movement.png")


@safe
def fig_gender_gap(gap_tab, shift_tab, models, cmap, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    conds = sorted(gap_tab.condition.unique())
    grouped_bars(axes[0], gap_tab.rename(columns={"mean_gap": "v"}), "v", "ci_lo", "ci_hi", models, cmap, conds)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Gender gap in (final score - gold)\nwoman - man, across items; 95% CI")
    if len(shift_tab):
        conds2 = sorted(shift_tab.condition.unique())
        grouped_bars(axes[1], shift_tab.rename(columns={"mean_diff": "v"}), "v", "ci_lo", "ci_hi", models, cmap, conds2)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("How much the biased judge widens that gap\n(paired bias effect: woman - man)")
    axes[0].legend(title="Model", frameon=False)
    savefig(fig, f"{out}/11_gender_gap.png")


@safe
def fig_demographic(tab, attr, models, out):
    """Heatmap: mean (score - gold) for each level of `attr` (rows) x condition (cols), per model."""
    conds = sorted(tab.condition.unique())
    levels = sorted(tab.level.unique(), key=str)
    fig, axes = plt.subplots(1, len(models), figsize=(1.2 * len(conds) * len(models) + 4, 0.45 * len(levels) + 2.5),
                             squeeze=False)
    vmax = np.nanmax(np.abs(tab.mean_score_minus_gold)) or 1
    im = None
    for ax, m in zip(axes[0], models):
        t = tab[tab.model == m].pivot(index="level", columns="condition", values="mean_score_minus_gold")
        t = t.reindex(index=levels, columns=conds)
        n = tab[tab.model == m].groupby("level")["n_items"].max().reindex(levels)
        im = ax.imshow(t.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([f"{l} (n={int(k) if k == k else 0})" for l, k in zip(levels, n)]
                           if ax is axes[0][0] else [], fontsize=8)
        for (i, j), v in np.ndenumerate(t.values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else "black")
        ax.set_title(f"{m}: score - gold by {attr}")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Mean score - gold")
    fig.savefig(f"{out}/12_by_{attr}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}/12_by_{attr}.png")


@safe
def fig_label_heatmap(eff, models, out):
    eff = eff.dropna(subset=["label"])
    top = eff.groupby("label")["n_items"].sum().nlargest(25).index
    eff = eff[eff.label.isin(top)]
    conds = sorted(eff.condition.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(1.3 * len(conds) * len(models) + 4, 0.35 * len(top) + 2),
                             squeeze=False)
    vmax = np.nanmax(np.abs(eff.mean_diff)) or 1
    for ax, m in zip(axes[0], models):
        p = eff[eff.model == m].pivot(index="label", columns="condition", values="mean_diff").reindex(
            index=sorted(top), columns=conds)
        im = ax.imshow(p.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels(p.index if ax is axes[0][0] else [], fontsize=8)
        for (i, j), v in np.ndenumerate(p.values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else "black")
        ax.set_title(f"{m}: bias effect by author gender")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Score change")
    fig.savefig(f"{out}/07_bias_effect_by_label.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}/07_bias_effect_by_label.png")


@safe
def fig_seeds(g, summ, models, cmap, out):
    if g.empty:
        print("  skipped fig_seeds: no repeated seeds")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tab = summarize(g.rename(columns={"sd": "v"}), ["model", "condition"], "v")
    conds = sorted(tab.condition.unique())
    grouped_bars(axes[0], tab, "mean", "ci_lo", "ci_hi", models, cmap, conds)
    axes[0].set_title("Seed-to-seed SD of final score within an item")
    grouped_bars(axes[1], summ, "icc", "none", "none", models, cmap, conds)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Reliability across seeds (ICC; 1 = items fully determine score)")
    axes[0].legend(title="Model", frameon=False)
    savefig(fig, f"{out}/08_seed_variability.png")


@safe
def fig_accuracy(acc, models, cmap, out):
    cols = [("spearman", "spearman_lo", "spearman_hi", "Spearman correlation with gold")]
    if acc.auc.notna().any():
        cols.append(("auc", "auc_lo", "auc_hi", "AUC vs. binary gold (0.5 = chance)"))
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 4.5), squeeze=False)
    conds = sorted(acc.condition.unique())
    for ax, (v, lo, hi, title) in zip(axes[0], cols):
        grouped_bars(ax, acc, v, lo, hi, models, cmap, conds)
        ax.set_title(title)
        if v == "auc":
            ax.axhline(0.5, color="k", lw=0.8, ls="--")
    axes[0][0].legend(title="Model", frameon=False)
    savefig(fig, f"{out}/09_agreement_with_gold.png")


@safe
def fig_model_scatter(m, ref, other, out):
    conds = sorted(m.condition.unique())
    ncol = min(len(conds), 4)
    nrow = int(np.ceil(len(conds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), conds):
        g = m[m.condition == c]
        jit = lambda n: RNG.normal(0, 0.03, n)
        ax.scatter(g.ref + jit(len(g)), g.other + jit(len(g)), s=8, alpha=0.3)
        ax.plot([1, 5], [1, 5], "k--", lw=0.8)
        r = st.pearsonr(g.ref, g.other)[0] if len(g) > 2 and g.ref.std() > 0 and g.other.std() > 0 else np.nan
        ax.set_title(f"{c} (r={r:.2f})", fontsize=9)
        ax.set_xlabel(f"{ref} item mean")
        ax.set_ylabel(f"{other} item mean")
    for ax in axes.ravel()[len(conds):]:
        ax.axis("off")
    savefig(fig, f"{out}/10_model_scatter.png")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def show(title, df):
    print(f"\n--- {title} ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(no data)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Analyse results_<model>_cases_<a>-<b>.jsonl files.")
    ap.add_argument("--results-dir", default=here,
                    help="folder holding the results files (default: the folder of this script)")
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit result files instead of searching --results-dir")
    ap.add_argument("--outdir", default=os.path.join(here, "analysis_out"))
    ap.add_argument("--pair-on", default="model,item_id,label,seed,panel_size,discussion",
                    help="columns that define a matched control for the paired bias effect")
    ap.add_argument("--gap", default="woman,man", help="the two genders compared in the gender gap")
    args = ap.parse_args()

    files = args.files or find_result_files(args.results_dir)
    if not files:
        raise SystemExit(f"No {RESULT_PATTERN} files found in {args.results_dir}. Put the results files "
                         "there (without renaming them) or pass --files.")
    tdir, fdir = f"{args.outdir}/tables", f"{args.outdir}/figures"
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    rows = load_all(files)
    runs, judges, rounds_ = parse_rows(rows)
    models = sorted(runs.model.unique(), key=model_sort_key)
    cmap = colors_for(models)
    ga, gb = [x.strip().lower() for x in args.gap.split(",")]

    def save(name, df):
        df.to_csv(f"{tdir}/{name}.csv", index=False)
        return df

    # 0. which cases each model has, and the label sanity check
    show("0a. Coverage (cases finished in every condition, per model)", save("coverage", coverage_report(runs)))
    bad = label_consistency(runs)
    save("label_mismatches", bad)
    print("\n--- 0b. Label check: same author labels for every model/condition of a case ---")
    print("  OK - every model saw identical labels for every case." if bad.empty else
          f"  WARNING: {bad.item_id.nunique()} case(s) were shown different labels - see tables/label_mismatches.csv")
    if len(models) > 1:
        sets = {m: set(runs.loc[runs.model == m, "item_id"]) for m in models}
        common = set.intersection(*sets.values())
        print(f"  cases shared by all models: {len(common)}; " +
              ", ".join(f"only {m}: {len(v - common)}" for m, v in sets.items()))

    print("\nRows per model/condition:")
    print(runs.groupby(["model", "condition"]).size().unstack(0).fillna(0).astype(int))
    show("0c. Parse-failure rate", save("parse_failures", parse_failures(runs)))

    # 1. baseline scores
    base = save("baseline_scores", baseline_scores(runs))
    show("1. Mean final score (95% bootstrap CI over items)", base)

    # 2. paired bias effect
    pair_on = [c.strip() for c in args.pair_on.split(",")]
    eff = save("bias_effect", paired_bias_effect(runs, pair_on, ["model", "condition"]))
    show("2. Paired bias effect (panel with biased judge - matched panel without)", eff)
    eff_lab = save("bias_effect_by_gender", paired_bias_effect(runs, pair_on, ["model", "condition", "label"]))
    show("2b. Paired bias effect by author gender", eff_lab)

    # 3. trajectory
    r = rounds_.merge(runs[["run_id", "model", "condition", "item_id"]], on="run_id")
    traj = save("per_round_trajectory", summarize(r, ["model", "condition", "round"], "score"))
    show("3. Per-round trajectory (multi-round conditions)", traj[traj.groupby(["model", "condition"])["round"].transform("max") > 0])

    # 4. judge-level order effects (round-0 score by speaking position)
    j = judges.merge(runs[["run_id", "model", "condition", "item_id", "panel_size"]], on="run_id")
    j0 = j[(j["round"] == 0) & (j.panel_size > 1)]
    pos = save("round0_score_by_position", summarize(j0, ["model", "condition", "position"], "score"))

    # 5-6. deliberation dynamics + conformity
    dyn = run_dynamics(judges).merge(runs[["run_id", "model", "condition", "item_id"]], on="run_id")
    for col, title in [("abs_shift", "5a. Mean |score change| per judge (first -> last round)"),
                       ("consensus_gain", "5b. Disagreement reduction (SD of judges, start - end)"),
                       ("neutral_shift", "6a. Neutral judges' mean score change, first -> last round"),
                       ("biased_shift", "6b. Biased judge's score change, first -> last round"),
                       ("gap_start", "6c. Biased minus neutral score, first round"),
                       ("gap_end", "6d. Biased minus neutral score, last round")]:
        if col in dyn and dyn[col].notna().any():
            show(title, save(f"dynamics_{col}", summarize(dyn, ["model", "condition"], col)))

    # 6e-f. gender gap across items, and how the biased judge changes it
    gg = save("gender_gap", gender_gap(runs, ga, gb))
    show(f"6e. Gender gap in (final score - gold): {ga} - {gb}", gg)
    gshift = save("gender_gap_shift", gender_gap_shift(runs, pair_on, ga, gb))
    show(f"6f. How much the biased judge widens that gap ({ga} - {gb})", gshift)

    # 6g. every demographic attribute
    demo = {}
    for attr in DEMOGRAPHICS:
        if attr in runs and runs[attr].notna().any():
            demo[attr] = save(f"by_{attr}", demographic_table(runs, attr))
            kw = demo[attr].drop_duplicates(["model", "condition"])[["model", "condition", "kruskal_p_across_levels"]]
            show(f"6g. Does (score - gold) differ across {attr}? (Kruskal-Wallis p)", kw)

    # 7. seeds
    sg, ss = seed_variability(runs)
    show("7. Seed variability (within-item SD, ICC)", save("seed_variability", ss))

    # 8. agreement with gold
    acc = save("agreement_with_gold", accuracy_table(runs))
    show("8. Agreement with gold label", acc)

    # 9. model comparison
    mscatter = None
    if len(models) >= 2:
        ref, other = models[0], models[1]
        mscatter, mc = model_comparison(runs, ref, other)
        show(f"9. {other} minus {ref} (paired by item)", save("model_comparison", mc))

    print("\nFigures:")
    fig_baseline(base, models, cmap, fdir)
    fig_bias_effect(eff, models, cmap, fdir)
    fig_trajectory(traj, models, cmap, fdir)
    fig_position(pos, models, fdir)
    if len(dyn):
        fig_dynamics(dyn, models, cmap, fdir)
        fig_conformity(dyn, models, cmap, fdir)
    fig_label_heatmap(eff_lab, models, fdir)
    fig_gender_gap(gg, gshift, models, cmap, fdir)
    for attr, tab in demo.items():
        fig_demographic(tab, attr, models, fdir)
    fig_seeds(sg, ss, models, cmap, fdir)
    fig_accuracy(acc, models, cmap, fdir)
    if mscatter is not None:
        fig_model_scatter(mscatter, models[0], models[1], fdir)
    print(f"\nDone. Tables -> {tdir}, figures -> {fdir}")


if __name__ == "__main__":
    main()
