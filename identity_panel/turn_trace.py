"""Turn-level trace: who spoke, in what order, what they saw, and what moved.

`PanelResult` already records each judge's score and reasoning per round, the
aggregate after each round, and the final group score. What it does not record
is the turn: a judge's speaking position, the score it carried into the turn,
and which peer opinions were actually in front of it when it revised.

That last one matters most. `prompts.format_peer_block` anonymises peers as
"Judge A", "Judge B", and the letters are assigned relative to the recipient,
so Judge A is a different seat depending on who is reading. Without the mapping
you cannot say "judge_3 moved toward the biased judge", only "judge_3 moved".

TurnRecord stores the mapping alongside the move.

These records live beside PanelResult rather than inside it, so nothing in
results.py has to change and nobody gets a merge conflict. If the team likes the
shape, the clean version is a `turns: list[TurnRecord]` field on RoundResult.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Sequence


@dataclass
class PeerView:
    """One peer as it appeared to the judge taking this turn."""

    letter: str          # the "Judge A" label this recipient saw
    judge_id: str        # who that actually was
    score: Optional[int]
    is_biased: bool
    position: int        # that peer's speaking position this round


@dataclass
class TurnRecord:
    item_id: str
    round_index: int         # 0 = independent, 1..N = discussion
    position: int            # speaking position within the round, 0-based
    judge_id: str
    model: str
    is_biased: bool
    bias_tag: Optional[str]

    score_before: Optional[int]   # score carried into this turn
    score_after: Optional[int]    # score stated in this turn
    reasoning: str                # what the judge said this turn

    peers_seen: list[PeerView] = field(default_factory=list)
    peer_block: Optional[str] = None   # exact text shown, for audit
    saw_self_previous: bool = False
    speaking_order: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def delta(self) -> Optional[int]:
        if self.score_before is None or self.score_after is None:
            return None
        return self.score_after - self.score_before

    @property
    def biased_peer_score(self) -> Optional[int]:
        """Score of the biased peer this judge saw, if it saw one."""
        for p in self.peers_seen:
            if p.is_biased:
                return p.score
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["peers_seen"] = [asdict(p) for p in self.peers_seen]
        d["delta"] = self.delta
        return d


def save_turns_jsonl(turns: Sequence[TurnRecord], path: str) -> None:
    """Full fidelity, including the exact peer block each judge was shown."""
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def save_turns_csv(turns: Sequence[TurnRecord], path: str) -> None:
    """Flat, one row per turn. This is the table the per-round curve comes from."""
    cols = [
        "item_id", "round_index", "position", "judge_id", "model",
        "is_biased", "bias_tag", "score_before", "score_after", "delta",
        "n_peers_seen", "biased_peer_score", "peer_scores_seen",
        "saw_self_previous", "speaking_order", "error", "reasoning",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in turns:
            w.writerow({
                "item_id": t.item_id,
                "round_index": t.round_index,
                "position": t.position,
                "judge_id": t.judge_id,
                "model": t.model,
                "is_biased": t.is_biased,
                "bias_tag": t.bias_tag,
                "score_before": t.score_before,
                "score_after": t.score_after,
                "delta": t.delta,
                "n_peers_seen": len(t.peers_seen),
                "biased_peer_score": t.biased_peer_score,
                "peer_scores_seen": "|".join(
                    f"{p.judge_id}:{p.score}" for p in t.peers_seen
                ),
                "saw_self_previous": t.saw_self_previous,
                "speaking_order": "|".join(t.speaking_order),
                "error": t.error,
                "reasoning": (t.reasoning or "").replace("\n", " "),
            })


def print_turns(turns: Sequence[TurnRecord]) -> None:
    current_item = None
    for t in turns:
        if t.item_id != current_item:
            current_item = t.item_id
            print(f"\nitem {t.item_id}  order: {' -> '.join(t.speaking_order)}")
        flag = " [biased]" if t.is_biased else ""
        saw = ", ".join(f"{p.letter}={p.judge_id}:{p.score}" for p in t.peers_seen) or "nobody"
        print(
            f"  r{t.round_index} p{t.position} {t.judge_id}{flag}: "
            f"{t.score_before} -> {t.score_after}  saw[{saw}]"
        )