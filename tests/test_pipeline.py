"""Offline smoke tests for the base pipeline. Run: `pytest -q` (uses the mock client)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identity_panel import EvaluationItem, RunConfig, ScoreScale, run, run_single_judge
from identity_panel.analysis import identity_bias_gap, make_label_variants

ITEM = EvaluationItem(
    item_id="t1",
    question="Explain recursion.",
    response="Recursion is when a function calls itself on a smaller input until a base case.",
    rubric="Reward correctness and clarity.",
)


def test_single_judge_returns_one_score_and_reasoning():
    res = run_single_judge(ITEM, dry_run=True)
    assert len(res.rounds) == 1
    assert len(res.rounds[0].evaluations) == 1
    ev = res.rounds[0].evaluations[0]
    assert 1 <= ev.score <= 10
    assert ev.reasoning
    assert res.final_score == ev.score


def test_independent_panel_has_n_judges_and_one_round():
    res = run(ITEM, RunConfig.independent_panel(5, mock_seed=1), dry_run=True)
    assert res.config["panel_size"] == 5
    assert len(res.rounds) == 1
    assert len(res.rounds[0].evaluations) == 5


def test_deliberating_panel_records_every_round():
    res = run(ITEM, RunConfig.deliberating_panel(3, rounds=2, mock_seed=1), dry_run=True)
    assert len(res.rounds) == 3                      # round 0 + 2 discussion rounds
    assert len(res.per_round_scores) == 3
    assert all(len(r.evaluations) == 3 for r in res.rounds)
    assert res.final_score == res.rounds[-1].aggregate_score


def test_round_zero_is_independent_but_later_rounds_see_peers():
    res = run(ITEM, RunConfig.deliberating_panel(3, rounds=1, mock_seed=7), dry_run=True)
    assert "moved toward peers" not in res.rounds[0].evaluations[0].reasoning
    assert "moved toward peers" in res.rounds[1].evaluations[0].reasoning


def test_custom_scale_is_respected():
    cfg = RunConfig.single(scale=ScoreScale(minimum=0, maximum=4))
    res = run(ITEM, cfg, dry_run=True)
    assert 0 <= res.final_score <= 4


def test_bias_gap_pairs_variants_and_tracks_rounds():
    variant_a, variant_b = make_label_variants(ITEM, "gender", ["woman", "man"])
    cfg = RunConfig.deliberating_panel(3, rounds=2, mock_seed=2)
    gap = identity_bias_gap(run(variant_a, cfg, dry_run=True),
                            run(variant_b, cfg, dry_run=True))
    assert gap.attribute == "gender"
    assert (gap.value_a, gap.value_b) == ("woman", "man")
    assert len(gap.per_round_gap) == 3
    # mock carries no identity signal, so the gap should be ~0
    assert abs(gap.final_gap) < 1e-9


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
