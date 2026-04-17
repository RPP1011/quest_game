"""Entity-description consistency critic.

Checks whether the prose describes seeded entities in ways that
contradict their seed descriptions. The canonical example: the Bluebell
is a "cog" in the seed but was rendered as "barge," "skiff," and "ship"
across chapters.

This is a keyword-based heuristic for common attribute types (vessel
type, physical description keywords, species). Not an LLM call — fast
and deterministic.

Integrated into the check stage or post-write critic pass.
"""
from __future__ import annotations

import re
from typing import Any

from app.world.schema import Entity, EntityType


# Attribute-specific contradiction rules. Each rule defines:
# - A set of seed keywords to look for in the entity description
# - A set of contradicting terms that should NOT appear in prose
#   when the entity is mentioned

VESSEL_TYPES = {
    "cog", "barge", "skiff", "galley", "sloop", "carrack", "caravel",
    "frigate", "longship", "dinghy", "pinnace", "brigantine", "ketch",
    "schooner", "clipper",
}

PHYSICAL_ATTRIBUTES = {
    # hair
    "dark-haired", "blonde", "red-haired", "silver-haired", "bald",
    # build
    "thin", "broad", "stocky", "tall", "short", "lean", "heavy",
    # skin
    "dark-skinned", "pale", "fair",
}


def _extract_keywords(description: str, vocabulary: set[str]) -> set[str]:
    """Extract known vocabulary terms from a description string."""
    desc_lower = description.lower()
    found = set()
    for term in vocabulary:
        if term.lower() in desc_lower:
            found.add(term.lower())
    return found


def _find_entity_mentions_with_context(
    prose: str, entity_name: str, window: int = 200,
) -> list[str]:
    """Find all prose windows around mentions of an entity name."""
    windows = []
    pattern = r"\b" + re.escape(entity_name) + r"\b"
    for match in re.finditer(pattern, prose, re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(prose), match.end() + window)
        windows.append(prose[start:end])
    return windows


def check_entity_consistency(
    prose: str,
    entities: list[Entity],
) -> list[dict]:
    """Check prose for entity-description contradictions.

    Returns a list of issues ``{severity, category, message, entity_id}``.
    """
    issues: list[dict] = []

    for entity in entities:
        desc = entity.data.get("description", "")
        if not desc:
            continue

        # Find mentions of this entity in the prose
        mentions = _find_entity_mentions_with_context(prose, entity.name)
        if not mentions:
            continue

        # Check vessel type consistency (for locations that are vessels)
        if entity.entity_type == EntityType.LOCATION:
            seed_vessels = _extract_keywords(desc, VESSEL_TYPES)
            if seed_vessels:
                for mention_ctx in mentions:
                    prose_vessels = _extract_keywords(mention_ctx, VESSEL_TYPES)
                    contradictions = prose_vessels - seed_vessels
                    if contradictions:
                        issues.append({
                            "severity": "warning",
                            "category": "continuity",
                            "message": (
                                f"{entity.name} is described as "
                                f"{', '.join(sorted(seed_vessels))} in the seed, "
                                f"but the prose calls it "
                                f"{', '.join(sorted(contradictions))}."
                            ),
                            "entity_id": entity.id,
                            "suggested_fix": (
                                f"Replace references to {entity.name} as "
                                f"{', '.join(sorted(contradictions))} with "
                                f"{', '.join(sorted(seed_vessels))}."
                            ),
                        })
                        break  # one issue per entity per check

        # Check character physical description keywords
        if entity.entity_type == EntityType.CHARACTER:
            seed_attrs = _extract_keywords(desc, PHYSICAL_ATTRIBUTES)
            if seed_attrs:
                for mention_ctx in mentions:
                    prose_attrs = _extract_keywords(mention_ctx, PHYSICAL_ATTRIBUTES)
                    # Only flag if a CONTRADICTING attribute is found
                    # (e.g., seed says "dark-haired" but prose says "blonde")
                    for attr_type in [
                        {"dark-haired", "blonde", "red-haired", "silver-haired", "bald"},
                        {"thin", "lean", "broad", "stocky", "heavy"},
                        {"dark-skinned", "pale", "fair"},
                        {"tall", "short"},
                    ]:
                        seed_in_type = seed_attrs & attr_type
                        prose_in_type = prose_attrs & attr_type
                        if seed_in_type and prose_in_type and seed_in_type != prose_in_type:
                            issues.append({
                                "severity": "warning",
                                "category": "continuity",
                                "message": (
                                    f"{entity.name} is {', '.join(sorted(seed_in_type))} "
                                    f"in the seed, but the prose describes them as "
                                    f"{', '.join(sorted(prose_in_type))}."
                                ),
                                "entity_id": entity.id,
                                "suggested_fix": (
                                    f"Update the description to match the seed: "
                                    f"{', '.join(sorted(seed_in_type))}."
                                ),
                            })
                            break

    return issues
