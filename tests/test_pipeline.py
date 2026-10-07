"""Offline smoke tests for the base pipeline. Run: `pytest -q` (uses the mock client)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
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


def test_openrouter_waits_and_retries_on_429(monkeypatch):
    import identity_panel.client as cl
    slept = []
    monkeypatch.setattr(cl.time, "sleep", lambda w: slept.append(w))
    sdk = _FakeSDK('{"score": 3, "reasoning": "ok"}')
    real, n = sdk._create, {"k": 0}
    RateLimitError = type("RateLimitError", (Exception,), {})
    def flaky(**kw):
        n["k"] += 1
        if n["k"] <= 2: raise RateLimitError("429")
        return real(**kw)
    sdk.chat.completions.create = flaky
    res = run_single_judge(ITEM, client=OpenRouterClient(sdk_client=sdk, rate_limit_waits=(7, 9)))
    assert res.final_score == 3 and slept == [7, 9]
    # once retries are exhausted the error is surfaced (judge records it, run is not silently wrong)
    n["k"] = -99
    res = run_single_judge(ITEM, client=OpenRouterClient(sdk_client=sdk, rate_limit_waits=(1,)))
    assert res.rounds[0].evaluations[0].error.startswith("RateLimitError")


# ---------------------------------------------------------------------------
# Gemini provider, quota handling, cache
# ---------------------------------------------------------------------------

def _gem(sdk, **kw):
    from identity_panel import GeminiClient
    return GeminiClient(sdk_client=sdk, **kw)


def test_gemini_request_shape():
    sdk = _FakeSDK('{"score": 3, "reasoning": "ok"}')
    cfg = RunConfig.single(model="gemini-3.1-flash-lite", effort="minimal")
    run(ITEM, cfg, client=_gem(sdk))
    kw = sdk.calls[0]
    assert kw["model"] == "gemini-3.1-flash-lite"
    assert kw["extra_body"] == {"reasoning_effort": "low"}   # "minimal" -> "low": safe on all Gemini models
    assert "temperature" not in kw
    assert "additionalProperties" not in json.dumps(kw["response_format"])
    assert "additionalProperties" not in kw["messages"][0]["content"]


def test_gemini3_skips_temperature_but_gemini25_sends_it():
    sdk = _FakeSDK('{"score": 3, "reasoning": "ok"}')
    run_single_judge(ITEM, model="gemini-3.1-flash-lite", client=_gem(sdk))
    run_single_judge(ITEM, model="gemini-2.5-flash", client=_gem(sdk))
    assert "temperature" not in sdk.calls[0] and sdk.calls[1]["temperature"] == 0.0


def test_gemini_requires_key(monkeypatch):
    from identity_panel import GeminiClient
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()


def _rl_sdk(msg, times, content='{"score": 3, "reasoning": "ok"}'):
    sdk = _FakeSDK(content)
    real, n = sdk._create, {"k": 0}
    RateLimitError = type("RateLimitError", (Exception,), {})
    def flaky(**kw):
        n["k"] += 1
        if n["k"] <= times:
            raise RateLimitError(msg)
        return real(**kw)
    sdk.chat.completions.create = flaky
    return sdk


def test_429_uses_server_retry_hint(monkeypatch):
    import identity_panel.client as cl
    slept = []
    monkeypatch.setattr(cl.time, "sleep", lambda w: slept.append(w))
    sdk = _rl_sdk("Quota exceeded for requests per minute. Please retry in 12.5s", 1)
    res = run_single_judge(ITEM, client=_gem(sdk))
    assert res.final_score == 3 and slept == [13.5]          # hint + 1s, not the 15s default


def test_daily_quota_is_not_retried(monkeypatch):
    import identity_panel.client as cl
    slept = []
    monkeypatch.setattr(cl.time, "sleep", lambda w: slept.append(w))
    sdk = _rl_sdk("GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded", 99)
    res = run_single_judge(ITEM, client=_gem(sdk))
    assert res.rounds[0].evaluations[0].error.startswith("DailyQuotaExhausted")
    assert slept == []                                        # no wait loop


def test_daily_budget_counts_and_persists(tmp_path):
    from identity_panel import DailyBudgetReached
    st = tmp_path / "q.json"
    sdk = _FakeSDK('{"score": 3, "reasoning": "ok"}')
    c = _gem(sdk, daily_budget=2, state_path=st)
    c.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "a"}])
    c.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "b"}])
    with pytest.raises(DailyBudgetReached):
        c.messages.create(model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "c"}])
    c2 = _gem(sdk, daily_budget=2, state_path=st)            # a restart remembers today's count
    assert c2.calls_today == 2


def test_cache_skips_repeat_calls(tmp_path):
    from identity_panel import CachedClient
    sdk = _FakeSDK('{"score": 4, "reasoning": "ok"}')
    c = CachedClient(_gem(sdk), tmp_path / "c.jsonl")
    run_single_judge(ITEM, client=c); run_single_judge(ITEM, client=c)
    assert len(sdk.calls) == 1 and c.hits == 1
    c2 = CachedClient(_gem(_FakeSDK("x")), tmp_path / "c.jsonl")   # survives restart
    assert run_single_judge(ITEM, client=c2).final_score == 4 and c2.hits == 1


def test_cache_shares_normal_judges_across_settings(tmp_path):
    from identity_panel import CachedClient
    sdk = _FakeSDK('{"score": 4, "reasoning": "ok"}')
    c = CachedClient(_gem(sdk), tmp_path / "c.jsonl")
    cfgs = make_panel_configs(RunConfig.independent_panel(3, model="gemini-2.5-flash"), BIAS)
    item = ITEM.with_label(make_label_variants(ITEM, "gender", ["woman"])[0].label)
    run(item, cfgs["no_bias"], client=c)           # 3 calls: identical prompts, but each seat samples separately
    assert len(sdk.calls) == 3
    run(item, cfgs["bias_present"], client=c)      # judges 1,2 identical -> cached; only judge 3 new
    assert len(sdk.calls) == 4 and c.hits == 2


def test_script_stops_at_daily_budget_then_resumes(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rx2", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py")
    rx = importlib.util.module_from_spec(spec); spec.loader.exec_module(rx)
    items = tmp_path / "i.jsonl"
    items.write_text("\n".join('{"item_id": "q%d", "question": "rate: t%d", "response": 1}' % (i, i) for i in range(3)))
    out = tmp_path / "r.jsonl"
    sdk = _FakeSDK('{"score": 3, "reasoning": "ok"}')
    monkeypatch.setattr(rx, "get_client", lambda **kw: _gem(sdk, daily_budget=kw["daily_budget"]))
    base_args = ["--items", str(items), "--judges", "3", "--scale", "1,5", "--out", str(out),
                 "--bias-setting", "no_bias", "--no-cache"]
    assert rx.main(base_args + ["--daily-budget", "12", "--overwrite"]) == 2                  # 12 calls = 2 panel runs x... stops
    done = sum(1 for _ in out.open())
    assert 0 < done < 6 and len(sdk.calls) <= 12
    sdk2 = _FakeSDK('{"score": 3, "reasoning": "ok"}')
    monkeypatch.setattr(rx, "get_client", lambda **kw: _gem(sdk2))  # "next day", no cap
    assert rx.main(base_args + ["--resume"]) == 0
    assert sum(1 for _ in out.open()) == 6


# ---------------------------------------------------------------------------
# Kaggle runner: state survives sessions via a (fake, local) Hugging Face dataset
# ---------------------------------------------------------------------------

class _FakeHub:
    """Stands in for HfApi: the 'remote repo' is a local folder."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True); self.commits = 0
    def create_repo(self, *a, **k): pass
    def list_repo_files(self, repo, repo_type=None): return [p.name for p in self.root.iterdir()]
    def hf_hub_download(self, repo, name, repo_type=None, local_dir=None):
        import shutil; dst = Path(local_dir) / name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.root / name, dst); return str(dst)
    def upload_folder(self, folder_path, repo_id, repo_type, allow_patterns, commit_message):
        import shutil; self.commits += 1
        for n in allow_patterns: shutil.copy(Path(folder_path) / n, self.root / n)


def _load_kaggle_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("kr", Path(__file__).resolve().parents[1] / "scripts" / "kaggle_runner.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_kaggle_runner_resumes_across_sessions(tmp_path, monkeypatch):
    kr = _load_kaggle_runner()
    hub = _FakeHub(tmp_path / "hub")
    monkeypatch.setattr(kr, "get_secret", lambda n: "x")
    items = tmp_path / "i.jsonl"
    items.write_text("\n".join('{"item_id": "q%d", "question": "rate: text %d", "response": 1}' % (i, i) for i in range(4)))
    base = ["--hf-repo", "u/r", "--dry-run", "--push-every", "60", "--entry", "run_experiment", "--", "--items", str(items),
            "--bias-setting", "all", "--judges", "3", "--rounds", "1", "--scale", "1,5"]

    def session(extra, tag):
        wd = tmp_path / tag                       # a brand-new empty disk each session
        return kr.main(["--workdir", str(wd), *base, *extra], api=hub), wd

    code, wd1 = session(["--limit", "2"], "s1")
    assert code == 0 and (hub.root / "results.jsonl").exists() and hub.commits == 1
    n1 = sum(1 for _ in (hub.root / "results.jsonl").open())
    assert n1 == 2 * 3 * 2                        # items x settings x labels
    code, wd2 = session([], "s2")                 # new disk: must pull, then do ONLY the remaining items
    assert code == 0
    assert sum(1 for _ in (hub.root / "results.jsonl").open()) == 4 * 3 * 2
    assert (wd2 / "results.jsonl").read_text().startswith((wd1 / "results.jsonl").read_text()[:200])  # kept old rows
    c = hub.commits
    code, _ = session([], "s3")                   # nothing left: no new commit, no duplicate rows
    assert code == 0 and hub.commits == c and sum(1 for _ in (hub.root / "results.jsonl").open()) == 24


def test_kaggle_notebook_is_valid():
    import nbformat
    nb = nbformat.read(str(Path(__file__).resolve().parents[1] / "scripts" / "kaggle_notebook.ipynb"), as_version=4)
    nbformat.validate(nb)
    assert any("kaggle_runner.py" in "".join(c.source) for c in nb.cells if c.cell_type == "code")


# ---------------------------------------------------------------------------
# The full suite: 3 tests / 8 conditions, random --n-scenarios, local model
# ---------------------------------------------------------------------------

def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _write_items(path, n):
    path.write_text("\n".join(json.dumps({"item_id": f"q{i}", "response": 1,
        "question": f"Evaluate and rank the following text for offensiveness on a scale of 1 to 5: text number {i}"})
        for i in range(n)), encoding="utf-8")


def _conds():
    from identity_panel.suite import build_conditions
    return {c.name: c for c in build_conditions(BiasSpec("gender", "woman", "man"), rounds=2)}


def test_suite_has_exactly_the_requested_eight_conditions():
    c = _conds()
    assert list(c) == ["1_single", "1_indep_3", "1_delib_3", "1_indep_5", "1_delib_5",
                       "2_biased_indep_5", "2_biased_delib_5", "3_biased_first_delib_5"]
    shape = {n: (x.config.panel_size, x.config.discussion) for n, x in c.items()}
    assert shape == {"1_single": (1, False), "1_indep_3": (3, False), "1_delib_3": (3, True),
                     "1_indep_5": (5, False), "1_delib_5": (5, True), "2_biased_indep_5": (5, False),
                     "2_biased_delib_5": (5, True), "3_biased_first_delib_5": (5, True)}
    assert {n: x.test for n, x in c.items()} == {"1_single": 1, "1_indep_3": 1, "1_delib_3": 1, "1_indep_5": 1,
        "1_delib_5": 1, "2_biased_indep_5": 2, "2_biased_delib_5": 2, "3_biased_first_delib_5": 3}
    for n in ("1_single", "1_indep_3", "1_delib_3", "1_indep_5", "1_delib_5"):      # test 1: nobody biased
        assert not any(j.bias_tag for j in c[n].config.judges)
    for n in ("2_biased_indep_5", "2_biased_delib_5", "3_biased_first_delib_5"):    # tests 2-3: exactly one biased
        assert sum(1 for j in c[n].config.judges if j.bias_tag) == 1
    assert c["2_biased_delib_5"].config.judges[-1].bias_tag            # biased judge speaks last ...
    assert c["3_biased_first_delib_5"].config.judges[0].bias_tag       # ... or first
    personas = {j.persona for n in ("2_biased_indep_5", "2_biased_delib_5", "3_biased_first_delib_5")
                for j in c[n].config.judges if j.bias_tag}
    assert len(personas) == 1                                          # the SAME biased judge everywhere
    assert c["1_delib_5"].config.judges[4].persona is None             # only difference 1_delib_5 -> 2_biased_delib_5


def test_sampling_is_random_reproducible_and_nested():
    from identity_panel.suite import sample_items
    items = [EvaluationItem(f"q{i}", "q", "r") for i in range(200)]
    a = [x.item_id for x in sample_items(items, 20, seed=1)]
    assert a == [x.item_id for x in sample_items(items, 20, seed=1)]          # reproducible
    assert a != [x.item_id for x in sample_items(items, 20, seed=2)]          # seed matters
    assert a != [f"q{i}" for i in range(20)]                                  # not just the first 20
    assert len(set(a)) == 20
    assert [x.item_id for x in sample_items(items, 50, seed=1)][:20] == a     # nested: growing N keeps old picks
    assert len(sample_items(items, None, 1)) == 200 and len(sample_items(items, 999, 1)) == 200


def test_run_suite_writes_jsonl_for_every_condition_and_resumes(tmp_path):
    rs = _load_script("run_suite")
    items, out = tmp_path / "i.jsonl", tmp_path / "res" / "results.jsonl"
    _write_items(items, 6)
    base = ["--items", str(items), "--dry-run", "--seed", "3", "--out", str(out)]
    assert rs.main([*base, "--n-scenarios", "2"]) == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * 2 * 8                                            # scenarios x labels x conditions
    assert {r["condition"] for r in rows} == set(_conds())
    assert {r["label"]["value"] for r in rows} == {"woman", "man"}
    assert all(r["gold"] == 1 and "rounds" in r and r["final_score"] == r["final_score"] for r in rows)
    first_ids = {r["item_id"] for r in rows}
    assert len(first_ids) == 2
    assert rs.main([*base, "--n-scenarios", "2"]) == 0                       # resume: nothing new
    assert sum(1 for _ in out.open(encoding="utf-8")) == len(rows)
    assert rs.main([*base, "--n-scenarios", "3"]) == 0                       # bigger N: only the extra scenario
    rows2 = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows2) == 3 * 2 * 8 and first_ids <= {r["item_id"] for r in rows2}
    assert (tmp_path / "res" / "results_summary.csv").exists()
    assert rs.main([*base, "--n-scenarios", "1", "--conditions", "1_single", "--overwrite"]) == 0
    assert sum(1 for _ in out.open(encoding="utf-8")) == 2                   # --conditions / --overwrite


def test_local_client_batches_concurrent_calls_and_parses_small_model_output():
    import threading
    from identity_panel.local_model import LocalHFClient, repair_json
    from identity_panel.client import CachedClient
    seen = []
    def fake_generate(specs):                       # stands in for model.generate on a GPU
        seen.append(len(specs))
        return [' 4, "reasoning": "Mild language, fine."}' for _ in specs]
    cli = LocalHFClient("fake", generate_fn=fake_generate, max_batch=8, max_wait=0.3)
    out = []
    def go(i):
        it = EvaluationItem(f"b{i}", f"rate text {i}", "")
        out.append(run(it, RunConfig.single(scale=ScoreScale(1, 5)), client=cli).final_score)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    cli.close()
    assert out == [4.0] * 8 and max(seen) > 1                       # 8 concurrent calls -> shared batch(es)
    assert repair_json('{"score": 3, "reasoning": "cut off mid') and json.loads(repair_json('{"score": 3, "reasoning": "cut off mid'))["score"] == 3
    assert json.loads(repair_json('{"score": 2'))["score"] == 2
    assert repair_json("no json here") == "no json here"


def test_kaggle_runner_runs_the_suite_across_sessions(tmp_path):
    kr = _load_script("kaggle_runner")
    hub = _FakeHub(tmp_path / "hub")
    kr.get_secret = lambda n: "x"
    items = tmp_path / "i.jsonl"; _write_items(items, 5)
    base = ["--hf-repo", "u/r", "--dry-run", "--push-every", "60", "--", "--items", str(items), "--seed", "1"]
    def session(n, tag):
        return kr.main(["--workdir", str(tmp_path / tag), *base, "--n-scenarios", str(n)], api=hub)
    assert session(1, "s1") == 0 and sum(1 for _ in (hub.root / "results.jsonl").open(encoding="utf-8")) == 1 * 2 * 8
    assert session(2, "s2") == 0 and sum(1 for _ in (hub.root / "results.jsonl").open(encoding="utf-8")) == 2 * 2 * 8


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
