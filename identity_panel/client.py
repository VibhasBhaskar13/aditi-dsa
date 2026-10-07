"""Model clients: Gemini (default) or OpenRouter for real runs, an offline mock for tests.

Both real providers speak the OpenAI chat-completions protocol, so one adapter serves both
(`ChatCompletionsClient`); it exposes the `client.messages.create(...)` surface `Judge`
uses, so nothing else in the pipeline knows which provider is behind it.

Free-tier survival kit (all optional, all in the client so every code path benefits):
  * min_interval   - spacing between calls (requests-per-minute limit)
  * daily_budget   - stop issuing calls once N were made today (Pacific time); persisted
                     in `state_path` so restarts remember
  * 429 handling   - honours the server's "retry in Ns" hint, tells per-minute from
                     per-day exhaustion (the latter is NOT retried)
  * CachedClient   - identical prompts are answered from disk, never re-sent
`get_client(dry_run=True)` returns a MockAnthropic that never touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .config import DEFAULT_PROVIDER

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

PROVIDERS = {
    "gemini": dict(base_url=GEMINI_BASE_URL, key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                   signup="https://aistudio.google.com/apikey"),
    "openrouter": dict(base_url=OPENROUTER_BASE_URL, key_envs=("OPENROUTER_API_KEY",),
                       signup="https://openrouter.ai/keys"),
}

# reasoning_effort values per provider. Google's OpenAI-compat docs list low|medium|high|none
# for Gemini, so "minimal" is sent as "low" (a value every Gemini model accepts) rather than
# risking a 400. OpenRouter's unified reasoning.effort takes low|medium|high.
_EFFORT_GEMINI = {"minimal": "low", "low": "low", "medium": "medium", "high": "high",
                  "xhigh": "high", "max": "high"}
_EFFORT_OPENROUTER = {"minimal": "low", "low": "low", "medium": "medium", "high": "high",
                      "xhigh": "high", "max": "high"}


class DailyQuotaExhausted(RuntimeError):
    """The provider says the DAILY quota is used up. Retrying today is pointless."""


class DailyBudgetReached(RuntimeError):
    """Our own --daily-budget cap was reached."""


# --------------------------------------------------------------------------- #
# Pacific-time helpers (Gemini's daily quota resets at midnight Pacific)       #
# --------------------------------------------------------------------------- #

def _pt_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:  # noqa: BLE001 - no tz database (Windows without `pip install tzdata`)
        # UTC-8 is never earlier than the real Pacific midnight, so waits err on the safe side.
        return datetime.now(timezone(timedelta(hours=-8)))


def pt_date() -> str:
    return _pt_now().date().isoformat()


def seconds_until_pt_reset() -> float:
    now = _pt_now()
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1.0, (nxt - now).total_seconds())


def get_client(*, dry_run: bool = False, mock_seed: Optional[int] = None,
               min_interval: float = 0.0, provider: Optional[str] = None,
               daily_budget: Optional[int] = None, state_path=None, cache_path=None):
    """Return a client with `.messages.create(...)`.

    Real clients read GEMINI_API_KEY (or GOOGLE_API_KEY) / OPENROUTER_API_KEY from the
    environment. Create ONE client and reuse it, otherwise throttling/budget can't work.
    `dry_run=True` returns the offline mock.
    """
    if dry_run:
        return MockAnthropic(seed=mock_seed)
    provider = provider or DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; options: {sorted(PROVIDERS)}")
    client = ChatCompletionsClient(provider=provider, min_interval=min_interval,
                                   daily_budget=daily_budget, state_path=state_path)
    return CachedClient(client, cache_path) if cache_path else client


# --------------------------------------------------------------------------- #
# Response shim shared by the real adapter, the cache and the mock.            #
# --------------------------------------------------------------------------- #

class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str, request_id: Optional[str]) -> None:
        self.content = [_Block(text)] if text else []
        self.stop_reason = "end_turn"
        self.stop_details = None
        self._request_id = request_id


# --------------------------------------------------------------------------- #
# Real provider adapter (OpenAI-compatible chat completions)                   #
# --------------------------------------------------------------------------- #

class ChatCompletionsClient:
    """Adapter: Anthropic-style `messages.create` -> OpenAI-style chat completions.

    Structured output is requested with `response_format=json_schema`. Not every
    model/provider supports that, so on a 400/404/422 the call is retried once without it
    (the system prompt also spells out the JSON shape) and the model is remembered, and the
    reply is tolerantly parsed (code fences / stray prose stripped).
    """

    def __init__(self, api_key: Optional[str] = None, *, provider: str = "openrouter",
                 base_url: Optional[str] = None, timeout: float = 120.0, max_retries: int = 0,
                 sdk_client=None, min_interval: float = 0.0,
                 rate_limit_waits=(15, 30, 60, 90, 120), transient_waits=(5, 15, 45),
                 daily_budget: Optional[int] = None, state_path=None) -> None:
        info = PROVIDERS[provider]
        self.provider = provider
        if sdk_client is None:
            key = api_key or next((os.environ[e] for e in info["key_envs"] if os.environ.get(e)), None)
            if not key:
                raise RuntimeError(
                    f"{info['key_envs'][0]} is not set. Get a key at {info['signup']}, then set it "
                    f"(macOS/Linux: export {info['key_envs'][0]}=...; PowerShell: "
                    f"$env:{info['key_envs'][0]}=\"...\") or use --dry-run for the offline mock."
                )
            from openai import OpenAI  # imported lazily so --dry-run needs no SDK

            # max_retries=0: WE retry (below), so every attempt is throttled and counted
            # against the daily budget instead of the SDK silently burning quota.
            sdk_client = OpenAI(api_key=key, base_url=base_url or info["base_url"],
                                timeout=timeout, max_retries=max_retries)
        self._sdk = sdk_client
        self._min_interval = min_interval
        self._last_call = 0.0
        self._rl_waits = tuple(rate_limit_waits)          # sleeps after successive 429s
        self._transient_waits = tuple(transient_waits)    # sleeps after network/5xx errors
        self._no_schema: set[str] = set()                 # models that rejected structured output
        self.daily_budget = daily_budget
        self._state_path = Path(state_path) if state_path else None
        self._day, self.calls_today = pt_date(), 0
        self._load_state()
        self.messages = _CCMessages(self)

    # -- daily call counter -------------------------------------------------
    def _load_state(self) -> None:
        if self._state_path and self._state_path.exists():
            try:
                st = json.loads(self._state_path.read_text(encoding="utf-8"))
                if st.get("date") == self._day:
                    self.calls_today = int(st.get("calls", 0))
            except (ValueError, OSError):
                pass

    def _save_state(self) -> None:
        if self._state_path:
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                self._state_path.write_text(
                    json.dumps({"date": self._day, "calls": self.calls_today}), encoding="utf-8")
            except OSError:
                pass

    def reset_day(self) -> None:
        self._day, self.calls_today = pt_date(), 0
        self._save_state()

    def _throttle(self) -> None:
        """Called before EVERY http request: roll the day, enforce budget, space calls."""
        if pt_date() != self._day:
            self.reset_day()
        if self.daily_budget is not None and self.calls_today >= self.daily_budget:
            raise DailyBudgetReached(f"{self.calls_today}/{self.daily_budget} calls used today (Pacific)")
        wait = self._last_call + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        self.calls_today += 1
        self._save_state()


class _CCMessages:
    def __init__(self, outer: ChatCompletionsClient) -> None:
        self._outer = outer

    def create(self, **kw):
        """Call once; on 429 / transient errors wait and retry (each attempt is counted)."""
        o = self._outer
        rl, tr = list(o._rl_waits), list(o._transient_waits)
        while True:
            try:
                return self._create_once(**kw)
            except (DailyQuotaExhausted, DailyBudgetReached):
                raise
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                if name == "RateLimitError":
                    if _is_daily_quota(exc):
                        raise DailyQuotaExhausted(_short(exc)) from exc
                    if not rl:
                        raise
                    hinted = _retry_after(exc)
                    if hinted is not None and hinted > 600:      # "retry in 3 hours" = daily wall
                        raise DailyQuotaExhausted(_short(exc)) from exc
                    slot = rl.pop(0)
                    wait = hinted + 1.0 if hinted is not None else slot
                    print(f"  [rate-limited (429); waiting {wait:.0f}s, {len(rl)} retries left]", flush=True)
                    time.sleep(wait)
                elif name in ("APIConnectionError", "APITimeoutError", "InternalServerError") and tr:
                    wait = tr.pop(0)
                    print(f"  [{name}; waiting {wait}s, {len(tr)} retries left]", flush=True)
                    time.sleep(wait)
                else:
                    raise

    def _create_once(self, *, model, max_tokens, system, messages, output_config=None,
                     temperature=None, **_):
        o = self._outer
        sdk = o._sdk
        output_config = output_config or {}
        schema = output_config.get("format", {}).get("schema")
        effort = output_config.get("effort")
        gemini = o.provider == "gemini"

        if schema:
            if gemini:
                schema = _strip_keys(schema, {"additionalProperties"})
            system = (f"{system}\n\nRespond with ONLY a single JSON object matching this "
                      f"JSON schema, with no other text:\n{json.dumps(schema)}")
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        if effort:
            if gemini:
                kwargs["extra_body"] = {"reasoning_effort": _EFFORT_GEMINI.get(effort, "high")}
            else:
                kwargs["extra_body"] = {"reasoning": {"effort": _EFFORT_OPENROUTER.get(effort, "high")}}
        if temperature is not None and not effort and not _skip_temperature(o.provider, model):
            kwargs["temperature"] = temperature

        if schema and model not in o._no_schema:
            rf = {"type": "json_schema",
                  "json_schema": {"name": "judge_score", "strict": True, "schema": schema}}
            try:
                o._throttle()
                raw = sdk.chat.completions.create(**kwargs, response_format=rf)
            except Exception as exc:  # noqa: BLE001
                # 400/404/422 = this model/provider can't do structured output. Remember it
                # (saves a wasted call every time on free tiers) and retry without it.
                if type(exc).__name__ not in ("BadRequestError", "NotFoundError",
                                              "UnprocessableEntityError"):
                    raise
                o._no_schema.add(model)
                o._throttle()
                raw = sdk.chat.completions.create(**kwargs)
        else:
            o._throttle()
            raw = sdk.chat.completions.create(**kwargs)

        if not getattr(raw, "choices", None):
            raise RuntimeError(f"{o.provider} returned no choices: {getattr(raw, 'error', None)}")
        choice = raw.choices[0]
        msg = choice.message
        resp = _Response(_extract_json(msg.content or ""), getattr(raw, "id", None))
        refusal = getattr(msg, "refusal", None)
        if refusal or choice.finish_reason == "content_filter":
            resp.stop_reason = "refusal"
            resp.stop_details = SimpleNamespace(category=refusal or "content_filter")
        elif choice.finish_reason == "length":
            resp.stop_reason = "max_tokens"
        return resp


class OpenRouterClient(ChatCompletionsClient):
    """Back-compat alias: the adapter pinned to OpenRouter."""

    def __init__(self, *a, **kw) -> None:
        kw.setdefault("provider", "openrouter")
        super().__init__(*a, **kw)


class GeminiClient(ChatCompletionsClient):
    """The adapter pinned to Google's Gemini API (GEMINI_API_KEY)."""

    def __init__(self, *a, **kw) -> None:
        kw.setdefault("provider", "gemini")
        super().__init__(*a, **kw)


def _skip_temperature(provider: str, model: str) -> bool:
    """Gemini 3 models are meant to run at the default temperature; a custom value can
    degrade output (unverified here - Google's Gemini 3 guidance, as I recall it).
    Set GEMINI_FORCE_TEMPERATURE=1 to send it anyway."""
    return (provider == "gemini" and model.startswith("gemini-3")
            and os.environ.get("GEMINI_FORCE_TEMPERATURE") != "1")


def _strip_keys(obj, keys):
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(v, keys) for v in obj]
    return obj


def _extract_json(text: str) -> str:
    """Strip code fences / surrounding prose; return the outermost {...} if present."""
    text = text.strip()
    lo, hi = text.find("{"), text.rfind("}")
    return text[lo:hi + 1] if lo != -1 and hi > lo else text


def _short(exc: Exception) -> str:
    return re.sub(r"\s+", " ", str(exc))[:300]


def _is_daily_quota(exc: Exception) -> bool:
    """Heuristic: does this 429 say the DAILY quota is gone (vs per-minute)?"""
    text = str(exc).lower()
    return any(k in text for k in ("perday", "per day", "per_day", "daily", "quota_exceeded",
                                   "requests per day"))


def _retry_after(exc: Exception) -> Optional[float]:
    """Seconds the server asked us to wait, if it said so."""
    resp = getattr(exc, "response", None)
    hdr = getattr(getattr(resp, "headers", None), "get", lambda *_: None)("retry-after")
    try:
        if hdr is not None:
            return float(hdr)
    except (TypeError, ValueError):
        pass
    m = (re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I)
         or re.search(r"retry[_ ]?delay\W+([\d.]+)\s*s", str(exc), re.I))
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Disk cache: identical prompt -> stored answer, no API call                   #
# --------------------------------------------------------------------------- #

class CachedClient:
    """Wraps a client; answers repeated identical requests from a JSONL file.

    The key includes the judge seat (`cache_key`), so judges that happen to send identical
    prompts still get independent samples. Saves daily quota in three ways: (1) the normal judges' independent round-0 answers are
    shared across the no_bias / bias_present / bias_first settings (same prompt, same answer,
    which also makes those settings differ ONLY in the biased judge); (2) calls that already
    succeeded inside a panel run that later failed are not repeated on resume; (3) re-running
    after a crash costs nothing. Delete the file (or use --no-cache) to force fresh samples.
    Only successful, non-empty answers are cached.
    """

    def __init__(self, inner, path) -> None:
        self.inner, self.path = inner, Path(path)
        self.hits = 0
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()                      # safe to share between worker threads
        self._inflight: dict[str, threading.Event] = {}    # key -> request currently being computed
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec
                    except (ValueError, KeyError):
                        pass
        self.messages = _CachedMessages(self)

    def __getattr__(self, name):          # calls_today, daily_budget, reset_day, ...
        return getattr(self.inner, name)

    @staticmethod
    def _key(kw) -> str:
        blob = json.dumps({k: kw.get(k) for k in ("model", "max_tokens", "system", "messages",
                                                  "output_config", "temperature", "cache_key")},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


class _CachedMessages:
    def __init__(self, outer: CachedClient) -> None:
        self._o = outer

    def create(self, **kw):
        o = self._o
        key = o._key(kw)
        while True:
            with o._lock:
                rec = o._mem.get(key)
                if rec:
                    o.hits += 1
                    return _Response(rec["text"], rec.get("request_id"))
                ev = o._inflight.get(key)
                if ev is None:                      # we are the one who computes it
                    o._inflight[key] = threading.Event()
                    break
            ev.wait()                               # someone else is computing the same request
        try:
            resp = o.inner.messages.create(**kw)
            if resp.stop_reason == "end_turn" and resp.content:
                rec = {"key": key, "text": resp.content[0].text, "request_id": resp._request_id,
                       "model": kw.get("model")}
                with o._lock:
                    o._mem[key] = rec
                    o.path.parent.mkdir(parents=True, exist_ok=True)
                    with o.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
            return resp
        finally:
            with o._lock:
                o._inflight.pop(key).set()


# --------------------------------------------------------------------------- #
# Offline mock. Deterministic, schema-valid, and *not* a model of real bias.  #
# --------------------------------------------------------------------------- #

class _Messages:
    def __init__(self, outer: "MockAnthropic") -> None:
        self._outer = outer

    def create(self, *, model, max_tokens, system, messages, output_config=None, **_):
        return self._outer._fake(model, system, messages, output_config)


class MockAnthropic:
    """Deterministic offline replacement for the real client.

    Produces `{"score": int, "reasoning": str}` JSON. The score is a stable hash of
    (seed, model, prompt) with the author label removed, nudged halfway toward the
    mean of any peer scores in the prompt so discussion rounds visibly converge.

    Plain judges carry no identity signal, so their bias gaps are ~0 by construction.
    To let the biased-judge plumbing be tested offline, a seat whose system prompt
    contains the `biased_persona` text shifts its score by +/-3 for the favored /
    disfavored author value. That tests wiring only; it says nothing about real models.
    """

    MOCK_BIAS_SHIFT = 3

    def __init__(self, seed: Optional[int] = None) -> None:
        self.messages = _Messages(self)
        self._seed = seed or 0

    def _fake(self, model, system, messages, output_config):
        schema = (output_config or {}).get("format", {}).get("schema", {})
        score_prop = schema.get("properties", {}).get("score", {})
        lo = int(score_prop.get("minimum", 1))
        hi = int(score_prop.get("maximum", 10))

        user_text = messages[-1]["content"]
        # Blind the hash to the identity label (peer/self blocks are kept).
        hashed = re.sub(r"## Author\n.*?\n\n", "", user_text, flags=re.DOTALL)
        digest = hashlib.sha256(f"{self._seed}|{model}|{hashed}".encode()).hexdigest()

        score = lo + int(digest[:8], 16) % (hi - lo + 1)
        shift = _mock_bias_shift(system, user_text, self.MOCK_BIAS_SHIFT)
        score += shift
        peers = _peer_scores(user_text)
        if peers:
            score = round(0.5 * score + 0.5 * (sum(peers) / len(peers)))
        score = max(lo, min(hi, int(score)))

        reasoning = (
            f"[mock:{model}] deterministic score {score} in [{lo},{hi}]"
            + (f"; moved toward peers {peers}" if peers else "")
        )
        return _Response(json.dumps({"score": score, "reasoning": reasoning}),
                         request_id=f"mock-{digest[:12]}")


def _mock_bias_shift(system: str, user_text: str, magnitude: int) -> int:
    hi = re.search(r'described as "([^"]*)", give a score[^\n]*? higher', system or "")
    lo = re.search(r'described as "([^"]*)", give a score[^\n]*? lower', system or "")
    author = re.search(r"## Author\n([^\n]*)", user_text)
    if not (hi and lo and author):
        return 0
    line = author.group(1)
    if re.search(rf"\b{re.escape(hi.group(1))}\b", line):
        return magnitude
    if re.search(rf"\b{re.escape(lo.group(1))}\b", line):
        return -magnitude
    return 0


def _peer_scores(user_text: str) -> list[int]:
    if "Other judges' evaluations" not in user_text:
        return []
    return [int(x) for x in re.findall(r"score (\d+)", user_text)]
