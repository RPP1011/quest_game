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
    # Gambling: only unambiguous gambling phrases. Removed "the deck"
    # (ship deck), "the hand" (body part), "the table" (furniture),
    # "fold/folding" (fabric), "dealt" (generic), "the cut" (wound).
    "gambling": [
        "the odds", "the bet", "the dice", "the pot", "the stakes",
        "roll the die", "rolling the dice", "bad bet", "bad hand",
        "bluff", "bluffing", "ante", "wager", "gamble", "gambler",
        "winning hand", "losing hand", "high card", "low card",
        "coin flip", "coin spinning", "toss of a coin",
        "played the hand", "stacked deck", "stacked against",
        "play the hand", "the house always", "the house wins",
        "the house takes",
    ],
    # Predator/prey: removed "stalking" (can be literal), "circling"
    # (can be physical movement)
    "predator_prey": [
        "the rat", "the mouse", "the cat", "the hunter",
        "the prey", "the predator", "the hawk", "the wolf",
        "cornered rat", "cornered animal", "the hunt",
        "the kill", "the trap", "snared", "the snare",
        "the cage", "the web",
    ],
    # Water/ocean: only figurative uses. On a ship/underwater these
    # are literal, so this family should be weighted lower for
    # maritime scenes. For now, keep but note the limitation.
    "water_ocean": [
        "the tide", "the current", "the swell", "the wave",
        "drowning", "treading water", "undertow", "the flood",
    ],
    "mechanical": [
        "the ticking", "the gears", "clockwork", "the mechanism",
        "the springs", "the cogs", "winding down", "winding up",
        "the mainspring", "the pendulum",
    ],
    "fire_light": [
        "the flame", "the fire", "smoldering",
        "the spark", "the ember", "the blaze",
        "white-hot", "molten",
    ],
    # Weight/gravity: removed "heavy" and "the weight" (too common
    # as literal descriptors). Only clearly metaphorical uses.
    "weight_gravity": [
        "the burden", "the anchor", "anchored", "leaden",
        "sinking feeling", "crushing weight",
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
            # Find ALL matches with surrounding context for specific replacements
            text_lower = prose.lower()
            all_matches: list[dict] = []
            for phrase in use_families[family_name]:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                for m in re.finditer(pattern, text_lower):
                    start = max(0, m.start() - 30)
                    end = min(len(prose), m.end() + 30)
                    all_matches.append({
                        "phrase": phrase,
                        "context": prose[start:end],
                        "position": m.start(),
                    })

            # Sort by position, take excess matches (keep first max_per_family)
            all_matches.sort(key=lambda x: x["position"])
            to_replace = all_matches[max_per_family:]
            examples_to_show = to_replace[:8]

            # Build specific replacement instructions
            replacement_lines = []
            for match in examples_to_show:
                replacement_lines.append(
                    f'  - "...{match["context"]}..." — replace the '
                    f'"{match["phrase"]}" imagery with a non-{family_name} '
                    f"alternative (bodily, architectural, textile, spatial, "
                    f"or sensory)"
                )

            suggested_fix = (
                f"This chapter uses '{family_name}' imagery {count}× "
                f"(limit: {max_per_family}). Keep the {max_per_family} "
                f"strongest instances. Replace these specific excess "
                f"occurrences with imagery from a DIFFERENT register:\n"
                + "\n".join(replacement_lines)
                + f"\n\nDo NOT replace with another '{family_name}' phrase. "
                f"Each replacement must use a completely different imagery "
                f"family (e.g., bodily sensation, architecture, fabric/textile, "
                f"weather, animal behavior, or physical space)."
            )

            issues.append({
                "severity": "warning",
                "category": "prose_quality",
                "message": (
                    f"Imagery family '{family_name}' appears {count}× in this "
                    f"chapter (limit: {max_per_family}). "
                    f"{len(to_replace)} occurrences must be replaced with "
                    f"non-{family_name} imagery."
                ),
                "family": family_name,
                "count": count,
                "excess": len(to_replace),
                "suggested_fix": suggested_fix,
            })

    return issues
