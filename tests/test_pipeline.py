"""Offline smoke tests for the base pipeline. Run: `pytest -q` (uses the mock client)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

import pytest

from identity_panel import (BiasSpec, EvaluationItem, MockAnthropic, OpenRouterClient,
                            RunConfig, ScoreScale, check_judge_bias, make_panel_configs,
                            run, run_single_judge)
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


# ---------------------------------------------------------------------------
# Biased judge + the three settings
# ---------------------------------------------------------------------------

BIAS = BiasSpec("gender", favored_value="woman", disfavored_value="man")
BASE = RunConfig.deliberating_panel(3, rounds=2, mock_seed=3)


def test_persona_is_only_on_the_biased_seat():
    cfgs = make_panel_configs(BASE, BIAS)
    assert all(j.persona is None and j.bias_tag is None for j in cfgs["no_bias"].judges)
    biased = [j for j in cfgs["bias_present"].judges if j.bias_tag]
    assert [j.judge_id for j in biased] == ["judge_3"]
    assert biased[0].persona and "woman" in biased[0].persona
    assert all(j.persona is None for j in cfgs["bias_present"].judges if not j.bias_tag)


def test_settings_change_only_presence_then_position():
    cfgs = make_panel_configs(BASE, BIAS)
    no, present, first = cfgs["no_bias"], cfgs["bias_present"], cfgs["bias_first"]
    # no_bias -> bias_present: same order, same ids/models, only the persona differs
    assert [j.judge_id for j in no.judges] == [j.judge_id for j in present.judges]
    assert [j.model for j in no.judges] == [j.model for j in present.judges]
    # bias_present -> bias_first: identical seats (same persona), only the order moves
    assert sorted(map(repr, present.judges)) == sorted(map(repr, first.judges))
    assert [j.judge_id for j in present.judges] == ["judge_1", "judge_2", "judge_3"]
    assert [j.judge_id for j in first.judges] == ["judge_3", "judge_1", "judge_2"]
    assert first.judges[0].bias_tag
    # everything else about the run is untouched
    for c in (present, first):
        assert (c.scale, c.aggregate, c.discussion_rounds) == (no.scale, no.aggregate, no.discussion_rounds)


def test_biased_seat_cannot_be_first_or_unknown():
    with pytest.raises(ValueError):
        make_panel_configs(BASE, BIAS, biased_judge_id="judge_1")
    with pytest.raises(ValueError):
        make_panel_configs(BASE, BIAS, biased_judge_id="nope")


def test_trace_metadata_records_bias_and_order():
    cfgs = make_panel_configs(BASE, BIAS)
    ITEM_L = ITEM.with_label(None)
    assert run(ITEM_L, cfgs["no_bias"], dry_run=True).config["bias_present"] is False
    c = run(ITEM_L, cfgs["bias_first"], dry_run=True).config
    assert c["bias_present"] and c["biased_judge_ids"] == ["judge_3"]
    assert c["judge_order"][0] == "judge_3" and c["biased_positions"] == [0]


def test_mock_biased_judge_differs_but_normal_judge_does_not():
    chk = check_judge_bias([ITEM], BIAS, dry_run=True)
    assert chk.normal_mean_gap == 0
    assert chk.biased_mean_gap >= 1.0 and chk.passed


def test_bias_shows_up_in_panel_gap_only_when_present():
    va, vb = make_label_variants(ITEM, "gender", ["woman", "man"])
    cfgs = make_panel_configs(BASE, BIAS)
    gaps = {k: identity_bias_gap(run(va, c, dry_run=True), run(vb, c, dry_run=True))
            for k, c in cfgs.items()}
    assert gaps["no_bias"].final_gap == 0 and gaps["no_bias"].per_round_gap == [0, 0, 0]
    assert gaps["bias_present"].per_round_gap[0] > 0
    # round 0 is independent, so position cannot matter there
    assert gaps["bias_present"].per_round_gap[0] == gaps["bias_first"].per_round_gap[0]


class _Recorder:
    """Client that records every user prompt as (system, user) and returns score 5."""
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, *, model, max_tokens, system, messages, **_):
        self.calls.append((system, messages[-1]["content"]))
        return MockAnthropic(0).messages.create(model=model, max_tokens=max_tokens, system=system,
                                                messages=messages,
                                                output_config={"format": {"schema": {"properties": {"score": {"minimum": 1, "maximum": 10}}}}})


def test_first_speaker_is_visible_to_later_speakers_within_a_round():
    cfg = make_panel_configs(RunConfig.deliberating_panel(3, rounds=1, mock_seed=1), BIAS)["bias_first"]
    rec = _Recorder()
    run(ITEM, cfg, client=rec)
    round1 = [u for _, u in rec.calls[3:]]          # after the 3 independent calls
    assert "Other judges'" in round1[0]
    # 2nd speaker's peer block reflects the 1st speaker's *round-1* reply (mock text says "moved toward peers")
    assert "moved toward peers" in round1[1] and "moved toward peers" not in round1[0]


def test_simultaneous_mode_ignores_order():
    cfg = RunConfig.deliberating_panel(3, rounds=1, mock_seed=1, sequential_turns=False)
    rec = _Recorder()
    run(ITEM, cfg, client=rec)
    assert all("moved toward peers" not in u for _, u in rec.calls[3:])


# ---------------------------------------------------------------------------
# popquorn-style items (gold rating in "response", text inside "question")
# ---------------------------------------------------------------------------

def test_popquorn_row_is_judgeable_and_gold_is_kept():
    it = EvaluationItem.from_dict({"item_id": "q1", "response": 1,
        "question": "Evaluate ... offensiveness ...: some text"})
    assert it.metadata["gold"] == 1 and it.response == ""
    labeled = it.with_label(make_label_variants(it, "gender", ["woman"])[0].label)
    res = run(labeled, RunConfig.single(scale=ScoreScale(1, 5)), dry_run=True)
    assert 1 <= res.final_score <= 5


# ---------------------------------------------------------------------------
# OpenRouter adapter (no network: fake SDK)
# ---------------------------------------------------------------------------

class _FakeSDK:
    def __init__(self, content, finish="stop", fail_first=False):
        self.calls, self._content, self._finish, self._fail = [], content, finish, fail_first
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self._fail and "response_format" in kw:
            BadRequestError = type("BadRequestError", (Exception,), {})
            raise BadRequestError("response_format unsupported")
        msg = SimpleNamespace(content=self._content, refusal=None)
        return SimpleNamespace(id="gen-1", choices=[SimpleNamespace(message=msg, finish_reason=self._finish)])


def test_openrouter_adapter_end_to_end_and_fallback():
    sdk = _FakeSDK('```json\n{"score": 4, "reasoning": "fine"}\n```', fail_first=True)
    res = run_single_judge(ITEM, client=OpenRouterClient(sdk_client=sdk))
    assert res.final_score == 4
    assert "response_format" in sdk.calls[0] and "response_format" not in sdk.calls[1]  # retried without
    assert sdk.calls[0]["messages"][0]["role"] == "system"
    assert sdk.calls[0]["temperature"] == 0.0


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient()


def test_openrouter_memoises_unsupported_structured_output():
    sdk = _FakeSDK('{"score": 4, "reasoning": "ok"}', fail_first=True)
    sdk_err = type("NotFoundError", (Exception,), {})
    orig = sdk._create
    def create(**kw):
        if "response_format" in kw:
            sdk.calls.append(kw); raise sdk_err("no endpoints")
        return orig(**kw)
    sdk.chat.completions.create = create
    c = OpenRouterClient(sdk_client=sdk)
    run_single_judge(ITEM, client=c); run_single_judge(ITEM, client=c)
    assert sum("response_format" in k for k in sdk.calls) == 1   # 2nd call skipped it


def test_experiment_csv_resume_and_early_stop(tmp_path, monkeypatch):
    import importlib.util, csv as _csv
    spec = importlib.util.spec_from_file_location("rx", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py")
    rx = importlib.util.module_from_spec(spec); spec.loader.exec_module(rx)
    items = tmp_path / "i.jsonl"
    items.write_text("\n".join('{"item_id": "q%d", "question": "rate: text %d", "response": 1}' % (i, i) for i in range(4)))
    out = tmp_path / "r.jsonl"
    args = ["--items", str(items), "--dry-run", "--bias-setting", "all", "--judges", "3",
            "--rounds", "1", "--scale", "1,5", "--out", str(out), "--max-failures", "2"]

    class Dead:                                   # every call raises -> all runs fail
        def __init__(self): self.messages = self
        def create(self, **kw): raise RuntimeError("429 rate limit")
    real = rx.get_client
    monkeypatch.setattr(rx, "get_client", lambda **kw: Dead())
    assert rx.main(args + ["--overwrite"]) == 2                  # stopped early, nothing saved
    assert out.read_text() == ""
    monkeypatch.setattr(rx, "get_client", real)
    assert rx.main(args + ["--resume"]) == 0                     # resumes and completes
    n = sum(1 for _ in out.open())
    assert n == 4 * 3 * 2                                        # items x settings x labels
    rows = list(_csv.DictReader(open(tmp_path / "r.csv", encoding="utf-8-sig")))
    assert len(rows) == n * 3 * 2 and {"setting", "judge_id", "score", "reasoning", "is_biased"} <= set(rows[0])
    assert rx.main(args + ["--resume"]) == 0 and sum(1 for _ in out.open()) == n   # idempotent


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
