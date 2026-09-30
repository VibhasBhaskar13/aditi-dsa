"""Model client factory: OpenRouter for real runs, an offline mock for plumbing tests.

`get_client()` returns an `OpenRouterClient` that reads OPENROUTER_API_KEY. It exposes
the same `client.messages.create(...)` surface `Judge` already uses, so nothing else in
the pipeline knows which provider is behind it.

`get_client(dry_run=True)` returns a MockAnthropic that never touches the network, so
the whole pipeline can be exercised for free in CI and local dev.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from types import SimpleNamespace
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# OpenRouter's unified `reasoning.effort` takes low | medium | high; clamp the rest.
_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def get_client(*, dry_run: bool = False, mock_seed: Optional[int] = None,
               min_interval: float = 0.0):
    """Return a client with `.messages.create(...)`.

    Real client: OpenRouter, key from the OPENROUTER_API_KEY environment variable.
    Pass `dry_run=True` for the offline mock. `min_interval` = minimum seconds between
    API calls (use ~3.5 for OpenRouter free models, which allow ~20 requests/minute).
    Create ONE client and reuse it, otherwise the throttle cannot work.
    """
    if dry_run:
        return MockAnthropic(seed=mock_seed)
    return OpenRouterClient(min_interval=min_interval)


# --------------------------------------------------------------------------- #
# Response shim shared by the OpenRouter adapter and the mock.                #
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
# OpenRouter                                                                   #
# --------------------------------------------------------------------------- #

class OpenRouterClient:
    """Adapter: Anthropic-style `messages.create` -> OpenRouter chat completions.

    Structured output is requested with `response_format=json_schema`. Not every
    model/provider on OpenRouter supports that, so on a 400 the call is retried once
    without it (the system prompt also spells out the JSON shape), and the reply is
    tolerantly parsed (code fences / stray prose stripped).
    """

    def __init__(self, api_key: Optional[str] = None, *, base_url: str = OPENROUTER_BASE_URL,
                 timeout: float = 120.0, max_retries: int = 5, sdk_client=None,
                 min_interval: float = 0.0) -> None:
        if sdk_client is None:
            key = api_key or os.environ.get(API_KEY_ENV)
            if not key:
                raise RuntimeError(
                    f"{API_KEY_ENV} is not set. Get a key at https://openrouter.ai/keys, then "
                    f"`export {API_KEY_ENV}=sk-or-...` (or use --dry-run for the offline mock)."
                )
            from openai import OpenAI  # imported lazily so --dry-run needs no SDK

            sdk_client = OpenAI(api_key=key, base_url=base_url, timeout=timeout,
                                max_retries=max_retries)
        self._sdk = sdk_client
        self._min_interval = min_interval
        self._last_call = 0.0
        self._no_schema: set[str] = set()   # models that rejected structured output
        self.messages = _ORMessages(self)

    def _throttle(self) -> None:
        wait = self._last_call + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()


class _ORMessages:
    def __init__(self, outer: OpenRouterClient) -> None:
        self._outer = outer

    def create(self, *, model, max_tokens, system, messages, output_config=None,
               temperature=None, **_):
        sdk = self._outer._sdk
        output_config = output_config or {}
        schema = output_config.get("format", {}).get("schema")
        effort = output_config.get("effort")

        if schema:
            system = (f"{system}\n\nRespond with ONLY a single JSON object matching this "
                      f"JSON schema, with no other text:\n{json.dumps(schema)}")
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        if effort:
            kwargs["extra_body"] = {"reasoning": {"effort": _EFFORT_MAP.get(effort, "high")}}
        elif temperature is not None:      # reasoning modes often reject custom temperature
            kwargs["temperature"] = temperature

        outer = self._outer
        if schema and model not in outer._no_schema:
            rf = {"type": "json_schema",
                  "json_schema": {"name": "judge_score", "strict": True, "schema": schema}}
            try:
                outer._throttle()
                raw = sdk.chat.completions.create(**kwargs, response_format=rf)
            except Exception as exc:  # noqa: BLE001
                # 400/404/422 = this model/provider can't do structured output. Remember it
                # (saves a wasted call every time on free tiers) and retry without it.
                if type(exc).__name__ not in ("BadRequestError", "NotFoundError",
                                              "UnprocessableEntityError"):
                    raise
                outer._no_schema.add(model)
                outer._throttle()
                raw = sdk.chat.completions.create(**kwargs)
        else:
            outer._throttle()
            raw = sdk.chat.completions.create(**kwargs)

        if not getattr(raw, "choices", None):
            raise RuntimeError(f"OpenRouter returned no choices: {getattr(raw, 'error', None)}")
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


def _extract_json(text: str) -> str:
    """Strip code fences / surrounding prose; return the outermost {...} if present."""
    text = text.strip()
    lo, hi = text.find("{"), text.rfind("}")
    return text[lo:hi + 1] if lo != -1 and hi > lo else text


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
