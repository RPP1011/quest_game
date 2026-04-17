"""Multi-chapter opening pattern critic.

Detects when consecutive chapters open with the same syntactic template
(e.g., "The *Bluebell* did not merely [verb]" × 3 chapters). This is a
lightweight heuristic — no LLM call — that runs post-write and produces
a warning the check stage can surface.

The critic compares the first sentence of the current chapter against
the first sentences of the prior N chapters using word-level trigram
overlap. If overlap exceeds a threshold, the opening is flagged as
repetitive.
"""
from __future__ import annotations

import re


def _first_sentence(text: str) -> str:
    """Extract the first sentence (up to the first period, !, or ?)."""
    text = text.strip()
    # Handle markdown italics/bold at start
    text = re.sub(r"^\*+", "", text)
    match = re.search(r"[.!?]", text)
    if match:
        return text[: match.end()].strip()
    # No sentence-ending punctuation in the first 200 chars — take the first line
    return text.split("\n")[0][:200].strip()


def _trigrams(text: str) -> set[tuple[str, ...]]:
    """Extract word-level trigrams from text (lowercased, punctuation stripped)."""
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 3:
        return set()
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def check_opening_repetition(
    current_prose: str,
    prior_chapter_proses: list[str],
    *,
    overlap_threshold: float = 0.25,
    min_prior: int = 1,
) -> list[dict]:
    """Check if the current chapter's opening repeats a prior pattern.

    Parameters
    ----------
    current_prose:
        The full prose of the current chapter.
    prior_chapter_proses:
        List of full prose texts from prior chapters (most recent last).
    overlap_threshold:
        Trigram Jaccard overlap above which the opening is flagged.
    min_prior:
        Minimum number of prior chapters needed before checking.

    Returns
    -------
    List of warning dicts ``{severity, category, message, similar_to}``.
    Empty if no repetition detected.
    """
    if len(prior_chapter_proses) < min_prior:
        return []

    current_opening = _first_sentence(current_prose)
    current_trigrams = _trigrams(current_opening)
    if not current_trigrams:
        return []

    issues: list[dict] = []
    for i, prior_prose in enumerate(prior_chapter_proses):
        prior_opening = _first_sentence(prior_prose)
        prior_tri = _trigrams(prior_opening)
        if not prior_tri:
            continue
        intersection = current_trigrams & prior_tri
        union = current_trigrams | prior_tri
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard >= overlap_threshold:
            chapters_back = len(prior_chapter_proses) - i
            issues.append({
                "severity": "warning",
                "category": "prose_quality",
                "message": (
                    f"Chapter opening is repetitive — {jaccard:.0%} trigram overlap "
                    f"with a chapter {chapters_back} chapter(s) ago. "
                    f"Current: \"{current_opening[:80]}\" / "
                    f"Prior: \"{prior_opening[:80]}\""
                ),
                "similar_to_chapters_back": chapters_back,
                "jaccard": round(jaccard, 3),
            })

    return issues
