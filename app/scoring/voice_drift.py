"""Rolling function-word KL drift monitor.

Function words (articles, prepositions, auxiliaries, pronouns) are the
most reliable authorship signal because they're topic-independent.
KL divergence from a baseline distribution detects voice drift.

This is diagnostic only — annotates, does not block or revise.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

FUNCTION_WORDS = [
    "the", "a", "an", "of", "in", "to", "and", "but", "or", "not",
    "he", "she", "it", "they", "his", "her", "its", "was", "were",
    "had", "has", "have", "is", "are", "been", "be", "would", "could",
    "should", "will", "can", "do", "did", "that", "this", "which",
    "who", "what", "when", "where", "how", "if", "than", "then",
    "so", "as", "at", "by", "for", "from", "on", "with", "into",
    "no", "nor", "yet", "just", "only", "very", "too", "also",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "any", "own", "same", "about", "after", "before",
    "between", "through", "during", "without", "again", "once",
]


def function_word_distribution(text: str) -> np.ndarray:
    """Compute normalized frequency distribution over function words."""
    tokens = text.lower().split()
    counts = Counter(tokens)
    raw = np.array([counts.get(w, 0) for w in FUNCTION_WORDS], dtype=float)
    total = raw.sum()
    if total == 0:
        return np.ones(len(FUNCTION_WORDS)) / len(FUNCTION_WORDS)
    return raw / total


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL(P || Q) with epsilon smoothing."""
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def assess_drift(kl: float) -> str:
    """Interpret KL divergence as drift severity."""
    if kl < 0.05:
        return "stable"
    if kl < 0.15:
        return "mild_drift"
    return "voice_drift"
