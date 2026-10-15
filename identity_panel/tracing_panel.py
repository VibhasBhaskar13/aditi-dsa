"""Panel subclass that makes speaking order a per-item variable and records turns.

Panel already implements sequential deliberation: round 0 is independent, rounds
1..N show each judge the latest opinion of every peer, and with
`sequential_turns=True` a judge who speaks earlier is seen, already updated, by
everyone after it. This subclass does not change that protocol. It adds the two
things the protocol does not currently expose:

    1. speaking order as a per-item setting rather than a property of RunConfig
    2. a turn-level trace: position, score before, what was said, score after,
       and which peers were actually visible at that moment

Usage:

    panel = TracingPanel(config, client, order="as_given", seed=0)
    results = [panel.run(item) for item in items]
    save_turns_csv(panel.turns, "turns.csv")
    save_turns_jsonl(panel.turns, "turns.jsonl")

`panel.turns` accumulates across every item the instance has run.
"""

from __future__ import annotations

import random
from string import ascii_uppercase
from typing import Optional, Sequence

from .aggregate import aggregate
from .config import RunConfig
from .data import EvaluationItem
from .judge import Judge
from .order import check_order_against_bias, resolve_order
from .results import JudgeEvaluation, PanelResult, RoundResult
from .panel import Panel
from .turn_trace import PeerView, TurnRecord


class PeerTraceDriftError(RuntimeError):
    """Raised when the reconstructed peer list stops matching the real peer block."""


class TracingPanel(Panel):
    def __init__(
        self,
        config: RunConfig,
        client,
        *,
        order: Sequence[str] | str = "as_given",
        seed: Optional[int] = 0,
        allow_reorder_with_bias: bool = False,
    ) -> None:
        super().__init__(config, client)
        check_order_against_bias(
            config, order, allow_reorder_with_bias=allow_reorder_with_bias
        )
        self.order = order
        self.seed = seed
        self._seat_order = list(self.judges)   # the config's own order, kept intact
        self._item_index = 0
        self._item_id: Optional[str] = None
        self._speaking_order: list[str] = []
        self.turns: list[TurnRecord] = []

    # -- order -------------------------------------------------------------

    def run(self, item: EvaluationItem) -> PanelResult:
        rng = random.Random(
            None if self.seed is None else self.seed + self._item_index
        )
        ids = [j.judge_id for j in self._seat_order]
        self._speaking_order = resolve_order(ids, self.order, rng, self._item_index)

        by_id = {j.judge_id: j for j in self._seat_order}
        # Panel reads turn order, and the peer-letter order, off self.judges, so
        # permuting it here is what actually changes who speaks when. Not
        # thread-safe; one TracingPanel instance per worker if you parallelise.
        self.judges = [by_id[jid] for jid in self._speaking_order]

        self._item_id = item.item_id
        self._item_index += 1
        return super().run(item)

    def _config_dict(self) -> dict:
        d = super()._config_dict()
        # super() reads judge_order off config.judges, which is the seat order,
        # not necessarily the order anyone spoke in.
        d["seat_order"] = [j.judge_id for j in self.config.judges]
        d["speaking_order"] = list(self._speaking_order)
        d["order_strategy"] = (
            self.order if isinstance(self.order, str) else list(self.order)
        )
        return d

    # -- tracing -----------------------------------------------------------

    def _run_round(
        self, item: EvaluationItem, round_index: int, prev: Optional[RoundResult]
    ) -> RoundResult:
        """Mirrors Panel._run_round, with a TurnRecord written per turn."""
        evaluations: list[JudgeEvaluation] = []
        prev_by_id = (
            {e.judge_id: e for e in prev.evaluations} if prev is not None else {}
        )

        for position, judge in enumerate(self.judges):
            peer_block, self_prev = self._peer_inputs(judge, prev, evaluations)
            peers = self._reconstruct_peers(judge, prev, evaluations)
            self._check_no_drift(peer_block, peers)

            ev = judge.evaluate(
                item,
                self.config.scale,
                round_index=round_index,
                peer_block=peer_block,
                self_prev=self_prev,
            )
            evaluations.append(ev)

            before = prev_by_id.get(judge.judge_id)
            self.turns.append(
                TurnRecord(
                    item_id=item.item_id,
                    round_index=round_index,
                    position=position,
                    judge_id=judge.judge_id,
                    model=judge.spec.model,
                    is_biased=judge.spec.bias_tag is not None,
                    bias_tag=judge.spec.bias_tag,
                    score_before=before.score if before is not None else None,
                    score_after=ev.score,
                    reasoning=ev.reasoning,
                    peers_seen=peers,
                    peer_block=peer_block,
                    saw_self_previous=self_prev is not None,
                    speaking_order=list(self._speaking_order),
                    error=ev.error,
                )
            )

        agg = aggregate(self.config.aggregate, [e.score for e in evaluations if e.ok])
        return RoundResult(round_index, evaluations, agg)

    def _reconstruct_peers(
        self,
        judge: Judge,
        prev: Optional[RoundResult],
        current: list[JudgeEvaluation],
    ) -> list[PeerView]:
        """Rebuild the letter -> judge_id mapping that format_peer_block produced.

        This duplicates the selection rule in Panel._peer_inputs, because that
        method returns rendered text and throws the peer list away. If Ava is
        willing to take a patch, the clean fix is for _peer_inputs to return the
        peers list as a third element and for this method to go away.
        _check_no_drift below is the canary until then.
        """
        if prev is None:
            return []

        latest = {e.judge_id: e for e in prev.evaluations}
        if self.config.sequential_turns and current:
            for e in current:
                if e.ok:
                    latest[e.judge_id] = e

        peers = [
            latest[j.judge_id]
            for j in self.judges
            if j.judge_id != judge.judge_id and j.judge_id in latest
        ]
        position_of = {jid: i for i, jid in enumerate(self._speaking_order)}
        bias_of = {j.judge_id: j.spec.bias_tag for j in self.judges}

        views: list[PeerView] = []
        # zip, then skip, exactly as format_peer_block does. A failed peer burns
        # its letter rather than shifting the ones after it, so the mapping only
        # matches if it is reproduced the same way.
        for letter, ev in zip(ascii_uppercase, peers):
            if not ev.ok:
                continue
            views.append(
                PeerView(
                    letter=letter,
                    judge_id=ev.judge_id,
                    score=ev.score,
                    is_biased=bias_of.get(ev.judge_id) is not None,
                    position=position_of.get(ev.judge_id, -1),
                )
            )
        return views

    @staticmethod
    def _check_no_drift(peer_block: Optional[str], peers: list[PeerView]) -> None:
        if peer_block is None:
            if peers:
                raise PeerTraceDriftError(
                    "no peer block was shown but peers were reconstructed"
                )
            return
        rendered = sum(1 for line in peer_block.splitlines() if line.startswith("- Judge "))
        if rendered != len(peers):
            raise PeerTraceDriftError(
                f"peer block shows {rendered} peers, trace reconstructed "
                f"{len(peers)}. Panel._peer_inputs or prompts.format_peer_block "
                "changed; update TracingPanel._reconstruct_peers to match."
            )