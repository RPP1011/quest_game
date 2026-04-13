"""Character perceptual grounding for detail selection (Gap G9).

Perceptual data for a character is persisted on ``Entity.data["perception"]``
as a JSON dict matching the ``PerceptualProfile`` schema. It is supplied at
seed time (or via delta) — the craft planner does NOT invent it. When a
scene enters a character's POV, what they notice is grounded in this
persistent profile plus the scene's current emotional state.

Schema for ``Entity.data["perception"]`` (all fields optional, defaults
applied by ``PerceptualProfile``)::

    {
      "permanent_preoccupations": [str, ...],
      "emotional_preoccupations": { "<emotion>": [str, ...], ... },
      "detail_mode": "precise" | "impressionistic" | "obsessive" | ...,
      "triple_duty_targets": [str, ...]
    }
"""
from __future__ import annotations

from app.planning.schemas import DetailPrinciple, PerceptualProfile
from app.world.schema import Entity, EntityType


PERCEPTION_KEY = "perception"


# Map from CraftPlan ``DetailPrinciple.detail_mode`` (an enum) to the
# PerceptualProfile's free-form mode string. We only use this to pick a
# default when the profile declares a mode — otherwise we leave the scene's
# detail_mode untouched.
_DETAIL_MODE_TO_PLAN: dict[str, str] = {
    # All ``PerceptualProfile.detail_mode`` values map cleanly to the
    # ``character_revealing`` plan enum value — per G9, the plan's enum
    # captures functional role, not sensory style, so we default to the
    # character-revealing mode when grounding.
    "precise": "character_revealing",
    "impressionistic": "mood_setting",
    "obsessive": "character_revealing",
}


def perceptual_profile_for(entity: Entity | None) -> PerceptualProfile | None:
    """Read ``entity.data["perception"]`` and return a ``PerceptualProfile`` or None.

    Returns None if ``entity`` is None, not a CHARACTER, lacks a
    ``perception`` key, or the payload is not a dict.
    """
    if entity is None:
        return None
    if entity.entity_type != EntityType.CHARACTER:
        return None
    raw = entity.data.get(PERCEPTION_KEY)
    if not isinstance(raw, dict):
        return None
    return PerceptualProfile.model_validate(raw)


def current_preoccupations(
    profile: PerceptualProfile,
    *,
    primary_emotion: str | None = None,
    secondary_emotion: str | None = None,
) -> list[str]:
    """Compute the active preoccupation list for a scene.

    Always includes ``permanent_preoccupations``. Extends with
    ``emotional_preoccupations`` entries keyed by the scene's primary
    (and optionally secondary) emotion. Emotion keys are matched
    case-insensitively. Order preserved; duplicates removed.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(items: list[str]) -> None:
        for item in items:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)

    _add(profile.permanent_preoccupations)

    # case-insensitive lookup into emotional_preoccupations
    lookup = {k.strip().lower(): v for k, v in profile.emotional_preoccupations.items()}
    for emo in (primary_emotion, secondary_emotion):
        if not emo:
            continue
        hits = lookup.get(emo.strip().lower())
        if hits:
            _add(hits)

    return out


def default_detail_principle(
    character_id: str,
    profile: PerceptualProfile,
    *,
    primary_emotion: str | None = None,
    secondary_emotion: str | None = None,
) -> DetailPrinciple:
    """Seed a grounded ``DetailPrinciple`` for a POV character in a scene.

    The craft planner passes this to the prompt (as a default) and uses it
    to backfill any LLM-emitted DetailPrinciple that arrives with empty
    ``perceptual_preoccupations``.
    """
    preoccupations = current_preoccupations(
        profile,
        primary_emotion=primary_emotion,
        secondary_emotion=secondary_emotion,
    )
    mode_plan = _DETAIL_MODE_TO_PLAN.get(profile.detail_mode, "character_revealing")
    return DetailPrinciple(
        perceiving_character_id=character_id,
        perceptual_preoccupations=preoccupations,
        detail_mode=mode_plan,  # type: ignore[arg-type]
        triple_duty_targets=list(profile.triple_duty_targets),
    )
