"""Pure-Python heuristic dimension scorers.

Each returns a float in [0, 1]. Tuning rationale lives next to each function
and in ``docs/calibration.md``.

These are intentionally NOT reused from ``app.engine.pipeline``'s rerank
scorer: the rerank scorer's weights evolve with craft goals, and letting the
calibration harness drift with it would mask regressions. If a coincidentally
overlapping metric ships in rerank later, keep both.
"""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Matches "...", \u201c...\u201d, \u2018...\u2019 spans of dialogue.
_QUOTE_SPANS = re.compile(
    r"\"[^\"\n]{1,500}\""
    r"|\u201c[^\u201d\n]{1,500}\u201d"
    r"|\u2018[^\u2019\n]{1,500}\u2019"
)

# Function-word list we strip from content-word overlap.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to from in on at by for with without into onto over under
    is are was were be been being am do does did doing have has had having will would should could
    shall may might can must not no yes as than that this these those there here it its his her
    their they them he she we us our your you i me my mine ours yours theirs himself herself itself
    themselves myself yourself ourselves
    """.split()
)


def clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def sentence_variance(text: str) -> float:
    """Std-dev of sentence lengths (words) normalized to [0, 1].

    Initial cap of std=15 saturated nearly every literary passage. Empirically,
    typical mid-chapter passages span: Hemingway ~4 std, Austen/Leonard ~8-12,
    McCarthy/Dostoevsky ~15-22, Joyce stream-of-consciousness ~30+. Capping at
    std=30 preserves separation between the clusters.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0
    lengths = [len(_tokens(s)) for s in sentences]
    sd = statistics.pstdev(lengths)
    return clip01(sd / 30.0)


def dialogue_ratio(text: str) -> float:
    """Fraction of characters inside quoted-speech spans. Clipped to [0, 1]."""
    if not text:
        return 0.0
    total = len(text)
    quoted = sum(m.end() - m.start() for m in _QUOTE_SPANS.finditer(text))
    return clip01(quoted / total)


def pacing(text: str) -> float:
    """Sentences per 100 words, normalized.

    3 sentences/100w -> 0.3, 10 sentences/100w -> 0.9, linear, clipped.
    Rationale: long flowing sentences (Woolf) => low pacing; short punchy
    crime prose (Leonard) => high pacing.
    """
    words = _tokens(text)
    sentences = _split_sentences(text)
    if not words:
        return 0.0
    density = len(sentences) / len(words) * 100.0
    # Linear: 3 -> 0.3, 10 -> 0.9, slope = (0.9 - 0.3) / 7
    slope = (0.9 - 0.3) / 7.0
    y = 0.3 + slope * (density - 3.0)
    return clip01(y)


def action_fidelity(passage: str, player_action: str) -> float:
    """Content-word overlap between player action and passage (quests only).

    Intent-capture proxy. Tokenize both, drop stopwords, compute
    |action_tokens intersect passage_tokens| / |action_tokens|. Returns 0.0
    when player_action is empty (non-quest usage should skip this dim).
    """
    action_toks = {
        t.lower() for t in _tokens(player_action) if t.lower() not in _STOPWORDS
    }
    if not action_toks:
        return 0.0
    passage_toks = {t.lower() for t in _tokens(passage)}
    hit = len(action_toks & passage_toks)
    return clip01(hit / len(action_toks))


HEURISTIC_DIMS: tuple[str, ...] = (
    "sentence_variance",
    "dialogue_ratio",
    "pacing",
)


def sensory_density(text: str) -> float:
    """Sensory keyword hits per 100 words, normalized to [0, 1].

    Reuses the sensory lexicon that backs
    ``validate_narrator_sensory_distribution`` (see ``app.planning.critics``).
    Critic reuse is intentional here — that lexicon is the one we trust for
    channel detection; counting its hits per length gives a density score
    without depending on a narrator target distribution.

    Tuning: 2 hits / 100 words → 0.3, 10 hits / 100 words → 0.95. Based on
    corpus sampling: Woolf ≥ 7/100w, Austen ≈ 2/100w, McCarthy ≥ 10/100w.
    """
    from app.planning.critics import _count_sensory_channels

    words = _tokens(text)
    if not words:
        return 0.0
    counts = _count_sensory_channels(text)
    total_hits = sum(counts.values())
    per_100 = total_hits * 100.0 / len(words)
    # Empirical on calibration passages: Austen ≈ 0.8/100w, Karamazov/PGTE
    # ~2-3/100w, Flaubert/Hemingway ~3-4/100w, Joyce/McCarthy ~5-6/100w. The
    # old 10/100w ceiling was unreachable — no passage approached it.
    lo, hi = 0.5, 6.0
    if per_100 <= lo:
        return clip01(per_100 * 0.2)
    if per_100 >= hi:
        return 0.95
    return clip01(0.2 + (per_100 - lo) / (hi - lo) * 0.75)


def run_heuristics(
    text: str,
    *,
    is_quest: bool = False,
    player_action: str | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "sentence_variance": sentence_variance(text),
        "dialogue_ratio": dialogue_ratio(text),
        "pacing": pacing(text),
        "sensory_density": sensory_density(text),
    }
    if is_quest and player_action:
        out["action_fidelity"] = action_fidelity(text, player_action)
    return out
