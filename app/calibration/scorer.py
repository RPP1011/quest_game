"""Critic-issue -> scalar score conversion.

Critics (``app.planning.critics``) return ``list[ValidationIssue]``. We map
that to a scalar score in [0, 1] for the calibration harness so critic-based
dimensions can live alongside heuristic- and LLM-judged dimensions:

    score = max(0.0, 1.0 - (num_errors * ERROR_WEIGHT + num_warnings * WARNING_WEIGHT))

Defaults: ERROR_WEIGHT = 0.25, WARNING_WEIGHT = 0.10. One error drops you to
0.75; four errors zeroes you out. Ten warnings also zero you out. Chosen
because calibration passages are ~200-600 words: at that size two or three
voice-violations is a meaningful failure, not a blip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


ERROR_WEIGHT = 0.25
WARNING_WEIGHT = 0.10


class _HasSeverity(Protocol):
    severity: str  # "error" | "warning" | other


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def critic_score(
    issues: Iterable[_HasSeverity],
    *,
    error_weight: float = ERROR_WEIGHT,
    warning_weight: float = WARNING_WEIGHT,
) -> float:
    errors = 0
    warnings = 0
    for issue in issues:
        sev = getattr(issue, "severity", "warning").lower()
        if sev == "error":
            errors += 1
        elif sev == "warning":
            warnings += 1
    return max(0.0, 1.0 - (errors * error_weight + warnings * warning_weight))


@dataclass(frozen=True)
class AggregateStats:
    mae: float
    rmse: float
    pearson: float


def mae(pairs: Iterable[tuple[float, float]]) -> float:
    ps = list(pairs)
    if not ps:
        return 0.0
    return sum(abs(a - b) for a, b in ps) / len(ps)


def rmse(pairs: Iterable[tuple[float, float]]) -> float:
    ps = list(pairs)
    if not ps:
        return 0.0
    return (sum((a - b) ** 2 for a, b in ps) / len(ps)) ** 0.5


def pearson(pairs: Iterable[tuple[float, float]]) -> float:
    """Pearson r. Returns 0.0 on degenerate (zero-variance) inputs."""
    ps = list(pairs)
    n = len(ps)
    if n < 2:
        return 0.0
    xs = [a for a, _ in ps]
    ys = [b for _, b in ps]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def aggregate(pairs: Iterable[tuple[float, float]]) -> AggregateStats:
    ps = list(pairs)
    return AggregateStats(mae=mae(ps), rmse=rmse(ps), pearson=pearson(ps))
