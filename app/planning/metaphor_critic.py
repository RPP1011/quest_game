"""Metaphor-variety critic.

Detects when a single imagery family dominates a chapter's figurative
language. The canonical problem: Tristan's chapters average ~8 gambling
metaphors ("the odds," "the house," "roll the die," "the deck") per
chapter when Pale Lights uses 2–3. The tic makes the prose feel
machine-generated even when individual sentences are strong.

This is a keyword-based heuristic — no LLM call. Define imagery
families as word sets; count matches per family per chapter; flag when
any single family exceeds a threshold.

The critic returns warnings that the revise stage can act on:
"reduce gambling metaphors from 12 to 3–4; vary the imagery register."
"""
from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Imagery families — each maps a family name to a set of trigger phrases.
# Phrases are lowercase; matching is word-boundary-aware.
# ---------------------------------------------------------------------------

IMAGERY_FAMILIES: dict[str, list[str]] = {
    "gambling": [
        "the odds", "the house", "the bet", "the dice", "the deck",
        "the pot", "the hand", "the deal", "the stakes",
        "roll the die", "rolling the dice", "fold", "folding",
        "bluff", "bluffing", "ante", "wager", "gamble", "gambler",
        "winning hand", "losing hand", "bad hand", "bad bet",
        "high card", "low card", "the table", "coin flip",
        "coin spinning", "toss of a coin", "played the hand",
        "stacked deck", "stacked against", "the cut",
        "dealt", "dealer", "play the floor", "play the hand",
    ],
    "predator_prey": [
        "the rat", "the mouse", "the cat", "the hunter",
        "the prey", "the predator", "the hawk", "the wolf",
        "cornered rat", "cornered animal", "the hunt",
        "stalking", "circling", "the kill", "the trap",
        "snared", "the snare", "the cage", "the web",
    ],
    "water_ocean": [
        "the tide", "the current", "the swell", "the wave",
        "drowning", "sinking", "the deep", "the depths",
        "treading water", "the shore", "the surface",
        "undertow", "whirlpool", "the flood",
    ],
    "mechanical": [
        "the ticking", "the gears", "clockwork", "the mechanism",
        "the springs", "the cogs", "winding down", "winding up",
        "the mainspring", "the pendulum",
    ],
    "fire_light": [
        "the flame", "the fire", "burning", "smoldering",
        "the spark", "the ember", "the blaze", "incandescent",
        "white-hot", "molten",
    ],
    "weight_gravity": [
        "the weight", "heavy", "heaviness", "the burden",
        "the anchor", "anchored", "pulling down", "dragging",
        "gravity", "sinking feeling", "leaden",
    ],
}


def _count_family(text: str, phrases: list[str]) -> int:
    """Count how many times any phrase from a family appears in text."""
    text_lower = text.lower()
    count = 0
    for phrase in phrases:
        # Word-boundary match to avoid partial matches
        pattern = r"\b" + re.escape(phrase) + r"\b"
        count += len(re.findall(pattern, text_lower))
    return count


def check_metaphor_variety(
    prose: str,
    *,
    max_per_family: int = 5,
    families: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Check a chapter for metaphor-family dominance.

    Parameters
    ----------
    prose:
        Full chapter text.
    max_per_family:
        Maximum occurrences of any single family before flagging.
        Default 5 is calibrated to Pale Lights (~2-3 gambling refs
        per chapter; 5 allows some headroom).
    families:
        Override the default imagery families. Mostly for testing.

    Returns
    -------
    List of warning dicts ``{severity, category, message, family,
    count, suggested_fix}``. Empty if no family exceeds threshold.
    """
    use_families = families or IMAGERY_FAMILIES
    counts: dict[str, int] = {}
    for family_name, phrases in use_families.items():
        c = _count_family(prose, phrases)
        if c > 0:
            counts[family_name] = c

    issues: list[dict] = []
    for family_name, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > max_per_family:
            # Find some example matches for the message
            text_lower = prose.lower()
            examples: list[str] = []
            for phrase in use_families[family_name]:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                for m in re.finditer(pattern, text_lower):
                    # Get surrounding context
                    start = max(0, m.start() - 20)
                    end = min(len(prose), m.end() + 20)
                    examples.append(f"...{prose[start:end]}...")
                    if len(examples) >= 3:
                        break
                if len(examples) >= 3:
                    break

            issues.append({
                "severity": "warning",
                "category": "prose_quality",
                "message": (
                    f"Imagery family '{family_name}' appears {count}× in this "
                    f"chapter (threshold: {max_per_family}). This creates a "
                    f"repetitive register. Examples: "
                    + "; ".join(examples[:3])
                ),
                "family": family_name,
                "count": count,
                "suggested_fix": (
                    f"Reduce '{family_name}' imagery from {count} to "
                    f"2–3 instances. Replace excess occurrences with "
                    f"imagery from other registers (sensory, spatial, "
                    f"bodily, architectural). Keep the strongest 2–3 "
                    f"and vary the rest."
                ),
            })

    return issues
