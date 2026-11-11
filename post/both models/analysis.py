#!/usr/bin/env python3
"""
Analysis of LLM-judge panel experiments (single judge / independent panel /
deliberating panel / panel containing a biased judge), for multiple model sizes.

Usage
-----
python analyze_judge_panels.py --file1 "results (1).jsonl" --file2 "results (2).jsonl" --outdir out

* file2 may contain several models (e.g. 3B rows followed by 7B rows). Rows are
  assigned to a model from config["models"], NOT from line position. If --file1 is
  given it is only used to verify that file2 starts with an identical copy of it.
* Every table is written to <outdir>/tables/*.csv and every figure to <outdir>/figures/*.png.

Data design (from the real rows)
--------------------------------
* Scores are 1-5 OFFENSIVENESS ratings (higher = judged more offensive); `gold` is an
  ordinal 1-5 reference rating per item (same for both labels of an item).
* Every item appears once per identity label (e.g. "woman" / "man"), usually with seed 0.
  So the paired replicate for an item is its LABEL, not its seed. The biased judge is
  instructed to adjust for the author's gender, so the bias shows up as a LABEL GAP
  (score for "woman" minus score for "man"), not only as a change in the overall level.
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
        runs.append(dict(
            run_id=run_id, model=model_tag(cfg), item_id=row.get("item_id"),
            seed=row.get("seed", 0),
            condition=row.get("condition") or derive_condition(cfg),
            panel_size=cfg.get("panel_size", 1), discussion=bool(cfg.get("discussion")),
            n_rounds=len(row.get("rounds") or []), bias_present=bias_present,
            n_biased=len(biased), bias_first=bool(order and order[0] in biased),
            label=f"{attr}:{val}" if attr is not None else None,
            gold=to_float(row.get("gold")), test=row.get("test"),
            final_score=to_float(row.get("final_score")),
            n_eval=n_eval, n_fail=n_fail))
    return pd.DataFrame(runs), pd.DataFrame(judges), pd.DataFrame(rounds_)


def load_all(file1, file2):
    rows2 = load_jsonl(file2)
    if file1:
        rows1 = load_jsonl(file1)
        n = len(rows1)
        same = rows2[:n] == rows1
        print(f"file1: {n} rows; file2: {len(rows2)} rows; "
              f"file2 starts with an identical copy of file1: {same}")
        tags = {model_tag(r.get("config") or {}) for r in rows2}
        if len(tags) == 1 and same and len(rows2) > n:
            print("  Model tags are identical in file2 -> falling back to the line split "
                  "(rows after file1 = second model).")
            for r in rows2[n:]:
                r.setdefault("config", {})["models"] = ["second_model"]
    return rows2


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


def label_gap(runs, B=1000):
    """Within-item gap between the two identity labels (second label minus first, alphabetical,
    e.g. woman - man), per model x condition, plus the paired change in that gap caused by a
    biased judge. Returns (gap_table, gap_shift_table)."""
    labs = sorted(runs.label.dropna().unique())
    if len(labs) != 2:
        print(f"  label_gap needs exactly 2 labels, found {labs}")
        return pd.DataFrame(), pd.DataFrame()
    first, second = labs
    keys = ["model", "condition", "item_id", "seed"]
    p = runs.pivot_table(index=keys, columns="label", values="final_score", aggfunc="mean").dropna()
    p["final_score"] = p[second] - p[first]
    p = p.reset_index()[keys + ["final_score"]]
    meta = runs[keys + ["panel_size", "discussion", "bias_present"]].drop_duplicates(subset=keys)
    gdf = p.merge(meta, on=keys)
    rows = []
    il = gdf.groupby(["model", "condition", "item_id"])["final_score"].mean().reset_index()
    for (m, c), g in il.groupby(["model", "condition"]):
        mean, lo, hi = boot_mean_ci(g["final_score"].values, B)
        try:
            pv = st.wilcoxon(g["final_score"].values).pvalue
        except ValueError:
            pv = np.nan
        rows.append(dict(model=m, condition=c, gap_definition=f"{second} - {first}", n_items=len(g),
                         mean_gap=mean, ci_lo=lo, ci_hi=hi, wilcoxon_p=pv))
    gdf = gdf.assign(label=None)
    shift = paired_bias_effect(gdf, ["model", "item_id", "seed", "panel_size", "discussion"],
                               ["model", "condition"], B)
    return pd.DataFrame(rows), shift


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
def fig_label_gap(gap_tab, shift_tab, models, cmap, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    conds = sorted(gap_tab.condition.unique())
    grouped_bars(axes[0], gap_tab.rename(columns={"mean_gap": "v"}), "v", "ci_lo", "ci_hi", models, cmap, conds)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Label gap in final score\n(alphabetically 2nd label minus 1st, e.g. woman - man; same item; 95% CI)")
    conds2 = sorted(shift_tab.condition.unique())
    grouped_bars(axes[1], shift_tab.rename(columns={"mean_diff": "v"}), "v", "ci_lo", "ci_hi", models, cmap, conds2)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Change in label gap caused by the biased judge\n(vs. matched panel without one)")
    axes[0].legend(title="Model", frameon=False)
    savefig(fig, f"{out}/11_label_gap.png")


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
        ax.set_title(f"{m}: bias effect by identity label")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--file1", default=None, help="3B-only file (optional; used only for verification)")
    ap.add_argument("--file2", required=True, help="file with all models")
    ap.add_argument("--outdir", default="analysis_out")
    ap.add_argument("--pair-on", default="model,item_id,label,seed,panel_size,discussion",
                    help="columns that define a matched control for the paired bias effect")
    args = ap.parse_args()

    tdir, fdir = f"{args.outdir}/tables", f"{args.outdir}/figures"
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    rows = load_all(args.file1, args.file2)
    runs, judges, rounds_ = parse_rows(rows)
    models = sorted(runs.model.unique(), key=model_sort_key)
    cmap = colors_for(models)
    print(f"\nRows per model/condition:")
    print(runs.groupby(["model", "condition"]).size().unstack(0).fillna(0).astype(int))

    def save(name, df):
        df.to_csv(f"{tdir}/{name}.csv", index=False)
        return df

    # 0. data quality
    show("0. Parse-failure rate", save("parse_failures", parse_failures(runs)))

    # 1. baseline scores
    base = save("baseline_scores", baseline_scores(runs))
    show("1. Mean final score (seeds averaged within item; 95% bootstrap CI over items)", base)

    # 2. paired bias effect
    pair_on = [c.strip() for c in args.pair_on.split(",")]
    eff = save("bias_effect", paired_bias_effect(runs, pair_on, ["model", "condition"]))
    show("2. Paired bias effect (panel with biased judge - matched panel without)", eff)
    eff_lab = save("bias_effect_by_label", paired_bias_effect(runs, pair_on, ["model", "condition", "label"]))

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

    # 6e. label gap (e.g. woman - man) and how the biased judge changes it
    lg, lg_shift = label_gap(runs)
    if len(lg):
        show("6e. Label gap in final score (within item)", save("label_gap", lg))
        show("6f. Change in label gap caused by biased judge (paired)", save("label_gap_shift", lg_shift))

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
    if len(lg):
        fig_label_gap(lg, lg_shift, models, cmap, fdir)
    fig_seeds(sg, ss, models, cmap, fdir)
    fig_accuracy(acc, models, cmap, fdir)
    if mscatter is not None:
        fig_model_scatter(mscatter, models[0], models[1], fdir)
    print(f"\nDone. Tables -> {tdir}, figures -> {fdir}")


if __name__ == "__main__":
    main()