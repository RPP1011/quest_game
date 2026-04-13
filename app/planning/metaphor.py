"""Character metaphor grounding for source-domain selection (Gap G10).

Metaphor data for a character is persisted on ``Entity.data["metaphor"]``
as a JSON dict matching the ``CharacterMetaphorProfile`` schema. It is
supplied at seed time (or via delta) — the craft planner does NOT invent
it. When a scene enters a character's POV, the source domains the
narrator draws from are grounded in this persistent profile plus the
scene's current emotional state (which activates a small computed set of
``current_domains``).

Schema for ``Entity.data["metaphor"]`` (all fields optional, defaults
applied by ``CharacterMetaphorProfile``)::

    {
      "permanent_domains": [str, ...],
      "forbidden_domains": [str, ...],
      "metaphor_density": 0.0..1.0,
      "extends_to_narration": bool
    }
"""
from __future__ import annotations

from app.planning.schemas import CharacterMetaphorProfile, MetaphorProfile
from app.world.schema import Entity, EntityType


METAPHOR_KEY = "metaphor"


# Heuristic lookup: emotion label → candidate metaphor source domains.
# These are *additions* to a character's permanent_domains, filtered by
# forbidden_domains. Deliberately small and clichéd — the intent is that
# characters who have the relevant life experience will find these
# activated, while characters without it (e.g. a sheltered scholar with
# "cold/stone" in forbidden_domains) will have them filtered out.
_EMOTION_DOMAIN_MAP: dict[str, list[str]] = {
    "dread":        ["cold", "stone", "narrow-spaces", "weight"],
    "fear":         ["cold", "small-animals", "edges", "shadow"],
    "terror":       ["cold", "drowning", "suffocation", "shadow"],
    "grief":        ["water", "weight", "absence", "erosion"],
    "sorrow":       ["water", "grey-weather", "absence"],
    "longing":      ["distance", "horizon", "water", "hollowness"],
    "anger":        ["fire", "metal", "storm", "teeth"],
    "rage":         ["fire", "flood", "wound", "iron"],
    "joy":          ["light", "birdsong", "warmth", "flowering"],
    "relief":       ["exhaled-breath", "opening", "warmth", "tide-retreating"],
    "hope":         ["dawn", "seedling", "opening", "thread"],
    "shame":        ["heat", "mirror", "weight", "small-rooms"],
    "disgust":      ["rot", "spoilage", "crawling-things"],
    "determination":["stone", "iron", "bone", "tether"],
    "awe":          ["cathedral", "stars", "tide", "silence"],
    "love":         ["warmth", "handhold", "hearth", "thread"],
    "jealousy":     ["blade", "mirror", "thorn", "fire"],
    "guilt":        ["stain", "weight", "bell", "shadow"],
    "loneliness":   ["empty-room", "winter-light", "distance"],
    "anxiety":      ["clock", "low-ceiling", "crowded-air", "pulse"],
    "contempt":     ["dust", "turned-back", "threadbare"],
    "curiosity":    ["threshold", "loose-thread", "lantern"],
}


def character_metaphor_profile_for(
    entity: Entity | None,
) -> CharacterMetaphorProfile | None:
    """Read ``entity.data["metaphor"]`` and return a ``CharacterMetaphorProfile``.

    Returns None if ``entity`` is None, not a CHARACTER, lacks a
    ``metaphor`` key, or the payload is not a dict.
    """
    if entity is None:
        return None
    if entity.entity_type != EntityType.CHARACTER:
        return None
    raw = entity.data.get(METAPHOR_KEY)
    if not isinstance(raw, dict):
        return None
    return CharacterMetaphorProfile.model_validate(raw)


def compute_current_domains(
    profile: CharacterMetaphorProfile,
    *,
    primary_emotion: str | None = None,
    secondary_emotion: str | None = None,
    max_domains: int = 4,
) -> list[str]:
    """Derive the emotion-activated metaphor domains for a scene.

    Heuristic: look up candidate domains for the scene's primary (and
    secondary) emotion from ``_EMOTION_DOMAIN_MAP``, keep only those the
    character has experience of (``permanent_domains``, case-insensitive
    match) or that are at least not ``forbidden_domains``. If the
    character's ``permanent_domains`` intersects the emotion's candidate
    list, prefer those. Otherwise fall back to candidates that are not
    forbidden — with the caveat that this is a heuristic and the craft
    planner is free to override.
    """
    permanent_l = {d.strip().lower() for d in profile.permanent_domains}
    forbidden_l = {d.strip().lower() for d in profile.forbidden_domains}

    # union candidates for both emotions (primary first)
    candidates: list[str] = []
    seen: set[str] = set()
    for emo in (primary_emotion, secondary_emotion):
        if not emo:
            continue
        for d in _EMOTION_DOMAIN_MAP.get(emo.strip().lower(), []):
            k = d.lower()
            if k not in seen:
                seen.add(k)
                candidates.append(d)

    # Prefer candidates that the character has experience of.
    preferred = [c for c in candidates if c.lower() in permanent_l]
    # Then allow non-forbidden candidates as plausible new associations.
    allowed = [
        c for c in candidates
        if c.lower() not in permanent_l and c.lower() not in forbidden_l
    ]
    out = preferred + allowed
    return out[:max_domains]


def default_metaphor_profile(
    character_id: str,
    profile: CharacterMetaphorProfile,
    *,
    primary_emotion: str | None = None,
    secondary_emotion: str | None = None,
) -> MetaphorProfile:
    """Seed a grounded ``MetaphorProfile`` (craft-plan output shape) for a POV scene.

    Maps the persistent float density onto the plan's Literal bucket.
    """
    current = compute_current_domains(
        profile,
        primary_emotion=primary_emotion,
        secondary_emotion=secondary_emotion,
    )
    density = profile.metaphor_density
    if density < 0.2:
        bucket = "sparse"
    elif density < 0.45:
        bucket = "occasional"
    elif density < 0.75:
        bucket = "regular"
    else:
        bucket = "rich"
    return MetaphorProfile(
        character_id=character_id,
        permanent_domains=list(profile.permanent_domains),
        current_domains=current,
        forbidden_domains=list(profile.forbidden_domains),
        metaphor_density=bucket,  # type: ignore[arg-type]
        extends_to_narration=profile.extends_to_narration,
    )
