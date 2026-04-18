"""Non-LLM structural prose metrics — guardrails, not targets.

Three deterministic measurements:
- Syntactic compression ratio (POS-tag gzip)
- MTLD (length-robust lexical diversity)
- MAUVE (distributional divergence from reference corpus)

These are alarms. Never optimize against them directly.
"""
from __future__ import annotations

import gzip
from collections import Counter

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


def syntactic_compression_ratio(prose: str) -> float:
    """Gzip compression ratio of the POS-tag sequence.

    Low ratio = repetitive syntax ("She X'd. She Y'd.").
    Threshold: < 0.35 is syntactically templated.
    """
    nlp = _get_nlp()
    doc = nlp(prose)
    pos_seq = " ".join(tok.pos_ for tok in doc)
    raw = pos_seq.encode("utf-8")
    if not raw:
        return 1.0
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)


def mtld_forward(words: list[str], ttr_threshold: float = 0.72) -> float:
    """One-direction MTLD pass."""
    factors = 0.0
    factor_length = 0
    types: set[str] = set()

    for word in words:
        types.add(word.lower())
        factor_length += 1
        ttr = len(types) / factor_length
        if ttr <= ttr_threshold:
            factors += 1
            types = set()
            factor_length = 0

    # Partial factor
    if factor_length > 0:
        ttr = len(types) / factor_length
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - ttr_threshold)

    return len(words) / factors if factors > 0 else float(len(words))


def mtld(prose: str, ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Length-robust type-token ratio. Returns the average of forward
    and backward passes. Higher = more diverse vocabulary.
    Threshold: < 50 suggests vocabulary collapse.
    """
    words = prose.split()
    if len(words) < 10:
        return 0.0
    forward = mtld_forward(words, ttr_threshold)
    backward = mtld_forward(words[::-1], ttr_threshold)
    return (forward + backward) / 2
