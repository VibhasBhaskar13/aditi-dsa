"""Top-level entry points."""

from __future__ import annotations

from typing import Optional

from .client import get_client
from .config import DEFAULT_MODEL, RunConfig, ScoreScale
from .data import EvaluationItem
from .panel import Panel
from .results import PanelResult


def run(item: EvaluationItem, config: RunConfig, *, client=None,
        dry_run: bool = False) -> PanelResult:
    """Run any configuration (single judge, independent panel, or deliberating panel)."""
    client = client or get_client(dry_run=dry_run, mock_seed=config.mock_seed)
    return Panel(config, client).run(item)


def run_single_judge(item: EvaluationItem, *, model: str = DEFAULT_MODEL,
                     scale: Optional[ScoreScale] = None, client=None,
                     dry_run: bool = False) -> PanelResult:
    """The simplest case: one judge -> one score + reasoning.

    Returns a PanelResult with a single round holding a single evaluation, so
    downstream code is identical to the multi-judge path.
    """
    return run(item, RunConfig.single(model=model, scale=scale),
               client=client, dry_run=dry_run)
