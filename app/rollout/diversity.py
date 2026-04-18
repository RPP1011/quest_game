"""Rollout diversity measurement.

Measures whether different virtual-player profiles produce meaningfully
different trajectories, or just correlated noise at 3× the cost.

Metrics:
- Action text Jaccard: word overlap between corresponding chapter actions
- Entity-mention Jaccard: per-chapter entity sets compared
- Hook-payoff overlap: which hooks paid off in each rollout
- Prose n-gram similarity: 4-gram overlap between corresponding chapters

If impulsive and cautious produce >80% identical trajectories, the
profiles aren't adding value.
"""
from __future__ import annotations

import re
from collections import defaultdict

from app.planning.metaphor_critic import _count_family, IMAGERY_FAMILIES
from app.world.schema import RolloutChapter
from app.world.state_manager import WorldStateManager


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def measure_rollout_diversity(
    world: WorldStateManager,
    rollout_id_a: str,
    rollout_id_b: str,
    quest_id: str,
) -> dict:
    """Compare two rollouts of the same candidate.

    Returns a dict with per-chapter and aggregate diversity metrics.
    Higher Jaccard = more similar = less diverse.
    """
    chs_a = {c.chapter_index: c for c in world.list_rollout_chapters(rollout_id_a)}
    chs_b = {c.chapter_index: c for c in world.list_rollout_chapters(rollout_id_b)}
    common_indices = sorted(set(chs_a) & set(chs_b))

    if not common_indices:
        return {"error": "no common chapter indices", "chapters": []}

    per_chapter: list[dict] = []
    action_jaccards: list[float] = []
    prose_jaccards: list[float] = []
    entity_jaccards: list[float] = []

    # Get entity names for mention detection
    entities = world.list_entities()
    entity_names = {e.name for e in entities if e.name}

    for idx in common_indices:
        a = chs_a[idx]
        b = chs_b[idx]

        # Action text similarity
        action_j = _jaccard(_words(a.player_action), _words(b.player_action))
        action_jaccards.append(action_j)

        # Prose 4-gram similarity
        prose_j = _jaccard(_ngrams(a.prose), _ngrams(b.prose))
        prose_jaccards.append(prose_j)

        # Entity mentions
        a_entities = {
            name for name in entity_names
            if re.search(r"\b" + re.escape(name) + r"\b", a.prose, re.IGNORECASE)
        }
        b_entities = {
            name for name in entity_names
            if re.search(r"\b" + re.escape(name) + r"\b", b.prose, re.IGNORECASE)
        }
        ent_j = _jaccard(a_entities, b_entities)
        entity_jaccards.append(ent_j)

        per_chapter.append({
            "chapter_index": idx,
            "action_jaccard": round(action_j, 3),
            "prose_4gram_jaccard": round(prose_j, 3),
            "entity_mention_jaccard": round(ent_j, 3),
            "action_a": a.player_action[:60],
            "action_b": b.player_action[:60],
        })

    # Hook payoff overlap
    hooks_a = {
        r["hook_id"] for r in world.list_hook_payoffs(quest_id)
        if r["rollout_id"] == rollout_id_a and r["paid_off_at_chapter"] is not None
    }
    hooks_b = {
        r["hook_id"] for r in world.list_hook_payoffs(quest_id)
        if r["rollout_id"] == rollout_id_b and r["paid_off_at_chapter"] is not None
    }
    hook_j = _jaccard(hooks_a, hooks_b)

    # Imagery family distribution comparison
    family_profiles_a: dict[str, int] = defaultdict(int)
    family_profiles_b: dict[str, int] = defaultdict(int)
    for idx in common_indices:
        for fam, phrases in IMAGERY_FAMILIES.items():
            family_profiles_a[fam] += _count_family(chs_a[idx].prose, phrases)
            family_profiles_b[fam] += _count_family(chs_b[idx].prose, phrases)

    import numpy as np
    mean_action = float(np.mean(action_jaccards))
    mean_prose = float(np.mean(prose_jaccards))
    mean_entity = float(np.mean(entity_jaccards))

    return {
        "rollout_a": rollout_id_a,
        "rollout_b": rollout_id_b,
        "common_chapters": len(common_indices),
        "aggregate": {
            "action_jaccard_mean": round(mean_action, 3),
            "prose_4gram_jaccard_mean": round(mean_prose, 3),
            "entity_mention_jaccard_mean": round(mean_entity, 3),
            "hook_payoff_jaccard": round(hook_j, 3),
        },
        "interpretation": {
            "action_diversity": "low" if mean_action > 0.5 else "moderate" if mean_action > 0.2 else "high",
            "prose_diversity": "low" if mean_prose > 0.3 else "moderate" if mean_prose > 0.1 else "high",
            "entity_diversity": "low" if mean_entity > 0.8 else "moderate" if mean_entity > 0.5 else "high",
            "overall": (
                "Profiles produce similar trajectories — 3× cost may not be justified"
                if mean_action > 0.5 and mean_prose > 0.2
                else "Profiles produce meaningfully different trajectories"
            ),
        },
        "imagery_families": {
            "rollout_a": dict(family_profiles_a),
            "rollout_b": dict(family_profiles_b),
        },
        "per_chapter": per_chapter,
    }
