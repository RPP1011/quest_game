"""12-dimension heuristic / critic scorer (Day 2) + Day 6 LLM-judge dims.

Measurement substrate for rerank, prompt-optimizer, and stress-test
degradation tracking. The Day 2 core is heuristic / critic only; Day 6
adds three LLM-judge dims (``tension_execution``,
``emotional_trajectory``, ``choice_hook_quality``) via
:meth:`Scorer.score_with_llm_judges`, which return an
:class:`ExtendedScorecard` that the pipeline persists into the same
``dimension_scores`` table under a single scorecard header.
"""
from .scorer import (
    DIMENSION_NAMES,
    DIMENSION_SOURCES,
    ExtendedScorecard,
    LLM_JUDGE_DIMS,
    Scorecard,
    Scorer,
)

__all__ = [
    "DIMENSION_NAMES",
    "DIMENSION_SOURCES",
    "ExtendedScorecard",
    "LLM_JUDGE_DIMS",
    "Scorecard",
    "Scorer",
]
