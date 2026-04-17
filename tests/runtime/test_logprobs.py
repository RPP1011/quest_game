"""Tests for logprob infrastructure on InferenceClient."""
from __future__ import annotations
import math

import pytest

from app.runtime.client import ChatWithLogprobs, TokenLogprob


def _make_result(token_logprobs: list[TokenLogprob]) -> ChatWithLogprobs:
    content = "".join(t.token for t in token_logprobs)
    return ChatWithLogprobs(content=content, token_logprobs=token_logprobs)


def test_score_token_distribution_normalizes():
    """Distribution over candidates sums to 1.0."""
    result = _make_result([
        TokenLogprob(token="7", logprob=math.log(0.6), top_logprobs={
            "6": math.log(0.2), "7": math.log(0.6), "8": math.log(0.15), "9": math.log(0.05),
        }),
    ])
    dist = result.score_token_distribution(0, ["6", "7", "8", "9", "10"])
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert dist["7"] > dist["6"] > dist["8"] > dist["9"]
    assert dist["10"] == 0.0  # not in top_logprobs


def test_score_token_distribution_out_of_range():
    """Out-of-range position returns uniform."""
    result = _make_result([])
    dist = result.score_token_distribution(99, ["A", "B"])
    assert dist == {"A": 0.5, "B": 0.5}


def test_expected_score_spiky():
    """Spiky distribution at 7 → E[score] near 7/10, high confidence."""
    result = _make_result([
        TokenLogprob(token="7", logprob=math.log(0.95), top_logprobs={
            "6": math.log(0.02), "7": math.log(0.95), "8": math.log(0.03),
        }),
    ])
    e_score, confidence = result.expected_score(0, min_val=1, max_val=10)
    # E[score] should be near (7-1)/(10-1) = 0.667
    assert 0.60 < e_score < 0.75
    # Confidence should be high (spiky)
    assert confidence > 0.7


def test_expected_score_spread():
    """Spread distribution → E[score] near center, low confidence."""
    # Roughly uniform over 4,5,6,7,8
    lp = math.log(0.2)
    result = _make_result([
        TokenLogprob(token="6", logprob=lp, top_logprobs={
            "4": lp, "5": lp, "6": lp, "7": lp, "8": lp,
        }),
    ])
    e_score, confidence = result.expected_score(0, min_val=1, max_val=10)
    # E[score] should be near (6-1)/(10-1) = 0.556
    assert 0.45 < e_score < 0.65
    # Confidence should be low (spread)
    assert confidence < 0.5


def test_expected_score_pairwise_ab():
    """For pairwise A/B, expected_score isn't the right tool, but
    score_token_distribution works: P(A wins) from logprobs at the
    A/B position."""
    result = _make_result([
        TokenLogprob(token="A", logprob=math.log(0.73), top_logprobs={
            "A": math.log(0.73), "B": math.log(0.27),
        }),
    ])
    dist = result.score_token_distribution(0, ["A", "B"])
    assert abs(dist["A"] - 0.73) < 0.01
    assert abs(dist["B"] - 0.27) < 0.01


def test_expected_score_empty_top_logprobs():
    """If top_logprobs has none of the candidates, return uniform."""
    result = _make_result([
        TokenLogprob(token="x", logprob=-1.0, top_logprobs={
            "x": -1.0, "y": -2.0,
        }),
    ])
    dist = result.score_token_distribution(0, ["1", "2", "3"])
    assert dist == {"1": pytest.approx(1/3), "2": pytest.approx(1/3), "3": pytest.approx(1/3)}
