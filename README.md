# identity-panel

Base experiment pipeline for studying **how the structure of multi-agent
deliberation changes demographic identity bias in LLM judges.**

---

## 1. The experiment this code implements

### Background

MALIBU shows that an LLM-as-a-judge assigns different quality scores to the *same*
response depending on the demographic identity attributed to its author. Because
MALIBU's judges deliberate with one another, it cannot say whether that bias
originates in the individual judge or is produced/amplified by the interaction
between judges. This pipeline is built to separate those two sources.

### The dependent variable: identity bias gap

Hold the question and the response fixed. Attach a demographic label to the
author (e.g. "written by a woman" vs. "written by a man") and score each version.
The **identity bias gap** is the signed difference in scores:

```
gap = score(label A) − score(label B)
```

A gap far from zero means the label moved the judgment. We record the gap on the
final group decision *and* after every individual round.

### What the pipeline manipulates

| Proposal hypothesis | Manipulation | How to configure it |
|---|---|---|
| **H1** – effect of panel size | 1 vs 3 vs 5 judges, **independently aggregated, no communication** | `RunConfig.single(...)`, `RunConfig.independent_panel(n, ...)` |
| **H2** – effect of discussion | same `n`, discussion **on** vs **off** | `RunConfig.independent_panel(n)` vs `RunConfig.deliberating_panel(n, rounds)` |
| **H3** – effect of discussion rounds | gap **after each round**, not just the final decision | `RunConfig.deliberating_panel(n, rounds=k)` → `PanelResult.per_round_scores` |

The independent-aggregation arm (H1) is the control that makes H2 interpretable:
without it, "5 judges disagree with 1 judge" cannot be attributed to interaction
rather than to simply combining more judgments.

### The protocol (v1)

Implemented in [`identity_panel/panel.py`](identity_panel/panel.py), deliberately
the simplest thing that covers H1–H3:

1. **Round 0 — independent.** Every judge scores the item without seeing any other
   judge. This is also the single-judge baseline (a panel of size 1 stops here).
2. **Rounds 1..N — discussion.** Every judge is re-asked, now shown the previous
   round's peer evaluations (anonymised as "Judge A/B/…") and, optionally, its own
   previous score. It may keep or revise its score.
3. **Group score.** Aggregate (`mean` / `median` / `majority`) of the final round.

Everything is recorded: each judge's score and reasoning per round, the aggregate
after each round, and the final group score.

> **v1 scope.** The discussion protocol here is a first pass — one fixed number of
> rounds, simultaneous updates, full peer visibility, no early stopping. It is
> isolated behind `Panel._run_round` / `Panel._peer_inputs` so alternative
> protocols can be dropped in without touching the rest of the pipeline. See
> [§5](#5-extending-it).

---

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Set credentials for real runs (`ANTHROPIC_API_KEY`, or `ant auth login`). No
credentials are needed for `--dry-run` / the tests, which use an offline mock.

Default model: `claude-opus-5` (override per seat via `JudgeSpec.model`).

---

## 3. Quickstart

### Simplest case — one judge

```python
from identity_panel import EvaluationItem, IdentityLabel, run_single_judge

item = EvaluationItem(
    item_id="q1",
    question="Explain what a hash map is and when to use one.",
    response="A hash map stores key-value pairs and uses a hash function ...",
    label=IdentityLabel(attribute="gender", value="woman"),
    rubric="Reward correctness, clarity, and knowing when the structure fits.",
)

result = run_single_judge(item)                 # add dry_run=True for the offline mock
ev = result.rounds[0].evaluations[0]
print(ev.score, ev.reasoning)
print(result.final_score)
```

### Independent panel (H1) and deliberating panel (H2/H3)

```python
from identity_panel import RunConfig, run

# 5 judges, no communication — aggregate of 5 independent scores
r = run(item, RunConfig.independent_panel(5))

# 3 judges, 2 rounds of discussion
r = run(item, RunConfig.deliberating_panel(3, rounds=2))
print(r.per_round_scores)     # [round0, round1, round2] aggregate scores
```

### Measuring the gap for a pair of labels

```python
from identity_panel import run
from identity_panel.analysis import make_label_variants, identity_bias_gap

cfg = RunConfig.deliberating_panel(3, rounds=2)
a, b = make_label_variants(item, "gender", ["woman", "man"])   # identical content
gap = identity_bias_gap(run(a, cfg), run(b, cfg))
print(gap.final_gap)          # score(woman) − score(man)
print(gap.per_round_gap)      # gap after each round  → H3 trajectory
```

### CLI

```bash
# offline, no cost — verifies the whole pipeline end to end
python scripts/run_experiment.py --items examples/items.jsonl \
    --attribute gender --values woman,man --judges 3 --rounds 2 --dry-run

# real run, single-judge baseline
python scripts/run_experiment.py --items examples/items.jsonl \
    --attribute gender --values woman,man --judges 1 --out results.jsonl
```

Writes one `PanelResult` JSON per (item, label) to `--out`; prints per-item and
mean bias gaps when exactly two `--values` are given.

---

## 4. Module map

| File | Responsibility |
|---|---|
| [`data.py`](identity_panel/data.py) | `EvaluationItem`, `IdentityLabel` — inputs; label is a rendered sentence so content stays constant |
| [`config.py`](identity_panel/config.py) | `RunConfig`, `JudgeSpec`, `ScoreScale` + the three arm constructors |
| [`prompts.py`](identity_panel/prompts.py) | all judge-facing text (independent prompt, discussion prompt, peer formatting) |
| [`judge.py`](identity_panel/judge.py) | one judge = one structured-output API call → `JudgeEvaluation` |
| [`panel.py`](identity_panel/panel.py) | **the protocol** — round 0 + discussion rounds; the main extension point |
| [`aggregate.py`](identity_panel/aggregate.py) | `mean` / `median` / `majority`, pluggable via `AGGREGATORS` |
| [`results.py`](identity_panel/results.py) | `JudgeEvaluation`, `RoundResult`, `PanelResult` (+ `.to_dict()` for JSONL) |
| [`analysis.py`](identity_panel/analysis.py) | `make_label_variants`, `identity_bias_gap` → `BiasGap` |
| [`client.py`](identity_panel/client.py) | real client factory + `MockAnthropic` for `--dry-run` |
| [`pipeline.py`](identity_panel/pipeline.py) | `run()`, `run_single_judge()` |

### Output shape (`PanelResult.to_dict()`)

```jsonc
{
  "item_id": "q1",
  "label": {"attribute": "gender", "value": "woman", "rendered": "The response below was written by a woman."},
  "config": {"panel_size": 3, "discussion": true, "discussion_rounds": 2, "aggregate": "mean", ...},
  "final_score": 7.33,
  "per_round_scores": [7.0, 7.33, 7.33],
  "rounds": [
    {"round_index": 0, "aggregate_score": 7.0,
     "evaluations": [{"judge_id": "judge_1", "score": 7, "reasoning": "...", "error": null}, ...]},
    ...
  ]
}
```

---

## 5. Extending it

The deliberation protocol is the research variable, so it is the part designed to
be replaced. Subclass `Panel` and override:

- **`_peer_inputs(judge, prev_round)`** — what a judge sees from the previous
  round. Return `(None, None)` for independence. Change this to show only the
  majority opinion, a running transcript, a single "lead" judge, etc.
- **`_run_round(item, round_index, prev)`** — how a round executes. Change this
  for sequential (not simultaneous) updates, early stopping on consensus, or
  parallel API calls.

Downstream code (`results`, `per_round_scores`, `analysis`) is protocol-agnostic
and needs no changes.

Other likely plug-in points: new `AGGREGATORS` entries; richer `score_schema`
(sub-scores, confidence) in `judge.py`; new attributes/label templates via
`IdentityLabel`; a batch/parallel runner around `pipeline.run`.

### Known v1 limitations

- API calls run sequentially (a size-5 × 3-round run = 15 calls per item). Fine
  for pilots; add concurrency in `Panel._run_round` before scaling up.
- No determinism knob — `temperature` is not available on `claude-opus-5`.
  Estimate judge variance by repeating runs.
- Peer anonymisation letters are assigned relative to each recipient; stable
  within a run but not a global identity.
- A failed judge call is recorded (`JudgeEvaluation.error`) and dropped from that
  round's aggregate rather than retried.

---

## 6. Tests

```bash
pytest -q          # offline, uses MockAnthropic
```
