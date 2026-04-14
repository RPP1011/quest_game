"""12-dimension heuristic / critic scorer (Day 2 of roadmap-3mo).

Measurement substrate for rerank, prompt-optimizer, and stress-test
degradation tracking. No LLM calls in v1 — LLM-judge dims land on Day 6
and slot into the same ``Scorecard`` shape.
"""
from .scorer import (
    DIMENSION_NAMES,
    DIMENSION_SOURCES,
    Scorecard,
    Scorer,
)

__all__ = ["DIMENSION_NAMES", "DIMENSION_SOURCES", "Scorecard", "Scorer"]
