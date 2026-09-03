"""Model client factory plus an offline stand-in for testing the plumbing.

`get_client(dry_run=True)` returns a MockAnthropic that never touches the network,
so the whole pipeline (panel size, discussion rounds, aggregation, bias-gap
analysis) can be exercised for free in CI and local dev.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional


def get_client(*, dry_run: bool = False, mock_seed: Optional[int] = None):
    """Return an Anthropic-compatible client.

    Real client resolves credentials from the environment (ANTHROPIC_API_KEY, or
    an `ant auth login` profile). Pass `dry_run=True` for the offline mock.
    """
    if dry_run:
        return MockAnthropic(seed=mock_seed)
    import anthropic

    return anthropic.Anthropic()


# --------------------------------------------------------------------------- #
# Offline mock. Deterministic, schema-valid, and *not* a model of real bias.  #
# --------------------------------------------------------------------------- #

class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str, request_id: str) -> None:
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"
        self.stop_details = None
        self._request_id = request_id


class _Messages:
    def __init__(self, outer: "MockAnthropic") -> None:
        self._outer = outer

    def create(self, *, model, max_tokens, system, messages, output_config=None, **_):
        return self._outer._fake(model, system, messages, output_config)


class MockAnthropic:
    """Deterministic offline replacement for `anthropic.Anthropic()`.

    Produces `{"score": int, "reasoning": str}` JSON. The score is a stable hash of
    (seed, model, prompt), nudged halfway toward the mean of any peer scores present
    in the prompt so discussion rounds visibly converge. It carries no identity
    signal, so bias gaps computed against mock output are ~0 by construction.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.messages = _Messages(self)
        self._seed = seed or 0

    def _fake(self, model, system, messages, output_config):
        schema = (output_config or {}).get("format", {}).get("schema", {})
        score_prop = schema.get("properties", {}).get("score", {})
        lo = int(score_prop.get("minimum", 1))
        hi = int(score_prop.get("maximum", 10))

        user_text = messages[-1]["content"]
        # Blind the mock to the identity label so bias gaps against mock output
        # are ~0 by construction (peer/self blocks are kept - discussion still moves).
        hashed = re.sub(r"## Author\n.*?\n\n", "", user_text, flags=re.DOTALL)
        digest = hashlib.sha256(
            f"{self._seed}|{model}|{hashed}".encode()
        ).hexdigest()

        score = lo + int(digest[:8], 16) % (hi - lo + 1)
        peers = _peer_scores(user_text)
        if peers:
            target = sum(peers) / len(peers)
            score = round(0.5 * score + 0.5 * target)
        score = max(lo, min(hi, int(score)))

        reasoning = (
            f"[mock:{model}] deterministic score {score} in [{lo},{hi}]"
            + (f"; moved toward peers {peers}" if peers else "")
        )
        payload = json.dumps({"score": score, "reasoning": reasoning})
        return _Response(payload, request_id=f"mock-{digest[:12]}")


def _peer_scores(user_text: str) -> list[int]:
    if "Other judges' evaluations" not in user_text:
        return []
    return [int(x) for x in re.findall(r"score (\d+)", user_text)]
