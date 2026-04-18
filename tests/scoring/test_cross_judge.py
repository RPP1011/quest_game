from __future__ import annotations
import pytest
from app.scoring.cross_judge import JudgePair, compute_agreement


def test_judge_pair_perfect_agreement():
    pair = JudgePair(
        gemma_scores={"prose_execution": 0.8, "subtext": 0.7},
        prometheus_scores={"prose_execution": 0.8, "subtext": 0.7},
    )
    assert pair.agreement == pytest.approx(0.0)
    assert pair.self_preference_flag is False


def test_judge_pair_mild_disagreement():
    pair = JudgePair(
        gemma_scores={"prose_execution": 0.8, "subtext": 0.7},
        prometheus_scores={"prose_execution": 0.7, "subtext": 0.6},
    )
    assert pair.agreement == pytest.approx(0.1)
    assert pair.self_preference_flag is False


def test_judge_pair_self_preference_flag():
    pair = JudgePair(
        gemma_scores={"prose_execution": 0.9, "subtext": 0.9},
        prometheus_scores={"prose_execution": 0.6, "subtext": 0.5},
    )
    assert pair.self_preference_flag is True


def test_compute_agreement():
    a = {"prose_execution": 0.8, "subtext": 0.7, "hook_quality": 0.6}
    b = {"prose_execution": 0.7, "subtext": 0.7, "hook_quality": 0.5}
    assert compute_agreement(a, b) == pytest.approx(1 / 15)
