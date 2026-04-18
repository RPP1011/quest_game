from __future__ import annotations
import pytest
import numpy as np
from app.scoring.voice_drift import (
    function_word_distribution, kl_divergence, FUNCTION_WORDS,
)


def test_function_word_distribution_shape():
    text = "the dog sat on the mat and the cat"
    dist = function_word_distribution(text)
    assert dist.shape == (len(FUNCTION_WORDS),)
    assert dist.sum() == pytest.approx(1.0, abs=0.01)


def test_kl_identical_distributions():
    p = np.array([0.3, 0.3, 0.2, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_different_distributions():
    p = np.array([0.5, 0.3, 0.1, 0.1])
    q = np.array([0.1, 0.1, 0.4, 0.4])
    kl = kl_divergence(p, q)
    assert kl > 0.5


def test_same_text_zero_drift():
    text = "He walked to the door and she followed him through it."
    p = function_word_distribution(text)
    q = function_word_distribution(text)
    assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-6)


def test_different_style_detectable_drift():
    text_a = (
        "He was not sure if he could do it, but he had to try. "
        "She had told him that it would be difficult, and he "
        "believed her. They were in this together, after all."
    )
    text_b = (
        "Magnificent crystalline structures erupted skyward, "
        "iridescent perfection embodying transcendent beauty. "
        "Luminous ethereal phenomena cascaded downward perpetually."
    )
    p = function_word_distribution(text_a)
    q = function_word_distribution(text_b)
    kl = kl_divergence(p, q)
    assert kl > 0.1
