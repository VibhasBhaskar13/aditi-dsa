"""Base pipeline for studying how multi-agent deliberation changes identity bias in LLM judges.

Public surface:
    EvaluationItem, IdentityLabel        - the thing being judged + the demographic label
    RunConfig, JudgeSpec, ScoreScale     - experiment configuration
    run(), run_single_judge()            - top-level entry points
    Panel, Judge                         - the machinery (subclass Panel to change the protocol)
    JudgeEvaluation, RoundResult, PanelResult   - structured output
    identity_bias_gap(), make_label_variants()  - the dependent variable
    BiasSpec, make_panel_configs()       - the biased judge + no_bias/bias_present/bias_first settings
    check_judge_bias()                   - verify the biased judge differs from a normal one
"""

from .config import RunConfig, JudgeSpec, ScoreScale, DEFAULT_MODEL, DEFAULT_PROVIDER
from .data import EvaluationItem, IdentityLabel
from .results import JudgeEvaluation, RoundResult, PanelResult
from .judge import Judge
from .panel import Panel
from .pipeline import run, run_single_judge
from .aggregate import AGGREGATORS, aggregate
from .analysis import identity_bias_gap, BiasGap, make_label_variants
from .client import (get_client, MockAnthropic, OpenRouterClient, GeminiClient,
                     ChatCompletionsClient, CachedClient, DailyQuotaExhausted, DailyBudgetReached)
from .bias import BiasSpec, biased_persona, make_panel_configs, BIAS_SETTINGS
from .bias_check import check_judge_bias, BiasCheck

__all__ = [
    "RunConfig", "JudgeSpec", "ScoreScale", "DEFAULT_MODEL", "DEFAULT_PROVIDER",
    "EvaluationItem", "IdentityLabel",
    "JudgeEvaluation", "RoundResult", "PanelResult",
    "Judge", "Panel", "run", "run_single_judge",
    "AGGREGATORS", "aggregate",
    "identity_bias_gap", "BiasGap", "make_label_variants",
    "get_client", "MockAnthropic", "OpenRouterClient", "GeminiClient",
    "ChatCompletionsClient", "CachedClient", "DailyQuotaExhausted", "DailyBudgetReached",
    "BiasSpec", "biased_persona", "make_panel_configs", "BIAS_SETTINGS",
    "check_judge_bias", "BiasCheck",
]
