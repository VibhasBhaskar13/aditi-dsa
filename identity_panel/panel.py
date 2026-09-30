"""The panel orchestrator - the main extension point.

Protocol implemented here (deliberately the simplest thing that covers H1-H3):

    round 0        every judge evaluates independently (no judge sees another)
    round 1..N     every judge re-evaluates, now shown the previous round's peer
                   evaluations (anonymised) and optionally its own previous score
    group score    aggregate of the final round

Turn order: with `config.sequential_turns=True` (default) judges speak one at a time in
`config.judges` order within each discussion round, and each sees the latest opinion of
every peer, so a judge who speaks first is seen (already updated) by everyone after it.
With False all judges respond simultaneously and order does not matter.

To change the deliberation protocol (e.g. only show the majority opinion, let
judges see the running transcript, stop early on consensus), subclass Panel and
override `_run_round` or `_peer_inputs`. Everything downstream (results,
per-round scores, bias-gap analysis) is protocol-agnostic.
"""

from __future__ import annotations

from typing import Optional

from . import prompts
from .aggregate import aggregate
from .bias import bias_trace_metadata
from .config import RunConfig
from .data import EvaluationItem
from .judge import Judge
from .results import JudgeEvaluation, PanelResult, RoundResult


class Panel:
    def __init__(self, config: RunConfig, client) -> None:
        self.config = config
        self.judges = [Judge(spec, client) for spec in config.judges]

    def run(self, item: EvaluationItem) -> PanelResult:
        rounds: list[RoundResult] = [self._run_round(item, 0, prev=None)]
        if self.config.discussion:
            for r in range(1, self.config.discussion_rounds + 1):
                rounds.append(self._run_round(item, r, prev=rounds[-1]))
        return PanelResult(
            item_id=item.item_id,
            label=item.label.to_dict() if item.label is not None else None,
            config=self._config_dict(),
            rounds=rounds,
            final_score=rounds[-1].aggregate_score,
        )

    # -- override to change the protocol -----------------------------------

    def _run_round(self, item: EvaluationItem, round_index: int,
                   prev: Optional[RoundResult]) -> RoundResult:
        evaluations: list[JudgeEvaluation] = []
        for judge in self.judges:
            peer_block, self_prev = self._peer_inputs(judge, prev, evaluations)
            evaluations.append(judge.evaluate(
                item, self.config.scale, round_index=round_index,
                peer_block=peer_block, self_prev=self_prev,
            ))
        agg = aggregate(self.config.aggregate,
                        [e.score for e in evaluations if e.ok])
        return RoundResult(round_index, evaluations, agg)

    def _peer_inputs(self, judge: Judge, prev: Optional[RoundResult],
                     current: Optional[list[JudgeEvaluation]] = None):
        """What `judge` gets to see. `(None, None)` == independent.

        `current` holds this round's evaluations so far (judges who already spoke).
        Peers are listed in speaking order, each at their most recent valid opinion.
        """
        if prev is None:
            return None, None
        latest = {e.judge_id: e for e in prev.evaluations}
        if self.config.sequential_turns and current:
            for e in current:
                if e.ok:
                    latest[e.judge_id] = e
        peers = [latest[j.judge_id] for j in self.judges
                 if j.judge_id != judge.judge_id and j.judge_id in latest]
        block = prompts.format_peer_block(
            peers, show_reasoning=self.config.show_peer_reasoning
        )
        self_prev = None
        if self.config.reveal_self_previous:
            self_prev = next(
                (e for e in prev.evaluations if e.judge_id == judge.judge_id), None
            )
        return block, self_prev

    # -- bookkeeping ------------------------------------------------------

    def _config_dict(self) -> dict:
        c = self.config
        d = {
            "panel_size": c.panel_size,
            "discussion": c.discussion,
            "discussion_rounds": c.discussion_rounds,
            "aggregate": c.aggregate,
            "scale": [c.scale.minimum, c.scale.maximum],
            "models": [s.model for s in c.judges],
            "show_peer_reasoning": c.show_peer_reasoning,
            "reveal_self_previous": c.reveal_self_previous,
            "sequential_turns": c.sequential_turns,
            "judge_order": [s.judge_id for s in c.judges],
        }
        d.update(bias_trace_metadata(c))  # adds biased_judge_ids / bias_present; empty/False if none
        return d