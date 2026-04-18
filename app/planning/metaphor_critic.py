"""Metaphor-variety critic.

Detects when a single imagery family dominates a chapter's figurative
language. Two modes:

1. **LLM classification** (primary): sends the chapter to the model,
   which identifies every figurative use and classifies it by imagery
   family. Catches novel phrasings the keyword list can't enumerate.

2. **Keyword heuristic** (fallback): regex phrase matching against a
   fixed list. Used when no client is available (tests, offline).

The critic returns warnings that the revise stage can act on:
"reduce gambling metaphors from 12 to 3–4; vary the imagery register."
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Keyword families — kept as fallback + for tests
# ---------------------------------------------------------------------------

IMAGERY_FAMILIES: dict[str, list[str]] = {
    "gambling": [
        "the odds", "the bet", "the dice", "the pot", "the stakes",
        "roll the die", "rolling the dice", "bad bet", "bad hand",
        "bluff", "bluffing", "ante", "wager", "gamble", "gambler",
        "winning hand", "losing hand", "high card", "low card",
        "coin flip", "coin spinning", "toss of a coin",
        "played the hand", "stacked deck", "stacked against",
        "play the hand", "the house always", "the house wins",
        "the house takes", "high stakes", "flush", "double down",
        "all in", "long shot", "jackpot", "cash in", "wild card",
        "a bad beat", "the house is", "the house look",
    ],
    "predator_prey": [
        "the rat", "the mouse", "the cat", "the hunter",
        "the prey", "the predator", "the hawk", "the wolf",
        "cornered rat", "cornered animal", "the hunt",
        "the kill", "the trap", "snared", "the snare",
        "the cage", "the web",
    ],
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
        pattern = r"\b" + re.escape(phrase) + r"\b"
        count += len(re.findall(pattern, text_lower))
    return count


def _ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------------------
# LLM-based classification
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "critics" / "metaphor_variety.j2"


async def classify_metaphors_llm(
    client: "InferenceClient",
    prose: str,
) -> dict:
    """Use the model to classify all figurative language in the chapter.

    Returns dict with 'families' (name → {count, quotes}),
    'total_figurative', 'dominant_family', 'dominant_percentage'.
    """
    from jinja2 import Template
    from app.runtime.client import ChatMessage

    # Truncate very long prose to avoid blowing the context window.
    # 12k words (~48k chars) is plenty for classification; the model
    # sees enough to identify all imagery families and their frequency.
    MAX_CHARS = 48_000
    truncated = prose[:MAX_CHARS] if len(prose) > MAX_CHARS else prose

    template = Template(_PROMPT_PATH.read_text())
    prompt = template.render(prose=truncated)

    raw = await client.chat(
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=2000,
        temperature=0.0,
        thinking=False,
    )
    content = raw.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]

    return json.loads(content.strip())


def _build_llm_issues(
    classification: dict,
    max_per_family: int = 5,
) -> list[dict]:
    """Convert LLM classification result into critic issues."""
    families = classification.get("families", {})
    issues: list[dict] = []

    for family_name, data in sorted(
        families.items(), key=lambda x: -x[1].get("count", 0),
    ):
        count = data.get("count", 0)
        if count <= max_per_family:
            continue

        quotes = data.get("quotes", [])
        excess = count - max_per_family
        # Show the excess quotes as replacement targets
        excess_quotes = quotes[max_per_family:]
        examples_to_show = excess_quotes[:8]

        replacement_lines = []
        for quote in examples_to_show:
            replacement_lines.append(
                f'  - FIND: "{quote}" → REPLACE with a non-{family_name} '
                f"phrase. Change ONLY this phrase, not the surrounding text."
            )

        suggested_fix = (
            f"SURGICAL REPLACEMENT REQUIRED. This chapter uses "
            f"'{family_name}' imagery {count}x (limit: {max_per_family}). "
            f"Replace EXACTLY these phrases — change only the figurative "
            f"language, keep the surrounding sentence intact:\n"
            + "\n".join(replacement_lines)
            + f"\n\nRULES: (1) Do NOT rewrite paragraphs — swap only the "
            f"marked phrase. (2) Do NOT introduce any new '{family_name}' "
            f"imagery anywhere. (3) Each replacement must use a different "
            f"family: bodily, architectural, textile, spatial, weather, "
            f"or sensory."
        )

        issues.append({
            "severity": "error",
            "category": "prose_quality",
            "message": (
                f"Imagery family '{family_name}' appears {count}x in this "
                f"chapter (limit: {max_per_family}). "
                f"{excess} occurrences must be replaced with "
                f"non-{family_name} imagery."
            ),
            "family": family_name,
            "count": count,
            "excess": excess,
            "suggested_fix": suggested_fix,
        })

    return issues


async def check_metaphor_variety_llm(
    client: "InferenceClient",
    prose: str,
    *,
    max_per_family: int = 4,
) -> list[dict]:
    """LLM-based metaphor variety check. Primary path.

    Falls back to keyword heuristic if the LLM call fails.
    """
    try:
        classification = await classify_metaphors_llm(client, prose)
        return _build_llm_issues(classification, max_per_family)
    except Exception:
        # Fallback to keyword heuristic
        return check_metaphor_variety(prose, max_per_family=max_per_family)


# ---------------------------------------------------------------------------
# Keyword-based heuristic (fallback + tests)
# ---------------------------------------------------------------------------

def check_metaphor_variety(
    prose: str,
    *,
    max_per_family: int = 5,
    families: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Keyword-based metaphor variety check. Fallback path.

    Parameters
    ----------
    prose:
        Full chapter text.
    max_per_family:
        Maximum occurrences of any single family before flagging.
    families:
        Override the default imagery families. Mostly for testing.

    Returns
    -------
    List of warning dicts. Empty if no family exceeds threshold.
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

            all_matches.sort(key=lambda x: x["position"])
            to_replace = all_matches[max_per_family:]
            examples_to_show = to_replace[:8]

            replacement_lines = []
            for match in examples_to_show:
                replacement_lines.append(
                    f'  - "...{match["context"]}..." — replace the '
                    f'"{match["phrase"]}" imagery with a non-{family_name} '
                    f"alternative (bodily, architectural, textile, spatial, "
                    f"or sensory)"
                )

            suggested_fix = (
                f"This chapter uses '{family_name}' imagery {count}x "
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
                    f"Imagery family '{family_name}' appears {count}x in this "
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
