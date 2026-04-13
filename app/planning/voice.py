"""Character voice grounding for free-indirect-style permeability (Gap G3).

Voice data for a character is persisted on ``Entity.data["voice"]`` as a
JSON dict matching the ``CharacterVoice`` schema. It is supplied at seed
time (or via delta) — the craft planner does NOT invent it. When a scene
enters a character's POV, the narrator absorbs some of that voice; the
amount and the specific vocabulary that bleeds are grounded in this
persistent data plus the narrator's style register.

Schema for ``Entity.data["voice"]`` (all fields optional, defaults applied
by ``CharacterVoice``)::

    {
      "vocabulary_level": "elevated" | "plain" | "coarse" | ...,
      "jargon_domains": [str, ...],
      "forbidden_words": [str, ...],
      "signature_phrases": [str, ...],
      "sentence_length_bias": "short_clipped" | "long_winding" | ...,
      "directness": 0.0..1.0,
      "uses_metaphor": true|false,
      "emotional_expression": str,
      "truth_tendency": str,
      "code_switching": [str, ...],
      "voice_samples": [str, ...]
    }

Additionally, a seed may supply per-character blended voice samples via
``Entity.data["blended_voice_samples"]`` — few-shot exemplars of the
narrator-plus-character blend for that character's POV passages. Those
are author-curated, not LLM-generated.
"""
from __future__ import annotations

from typing import Iterable

from app.planning.schemas import CharacterVoice, VoicePermeability
from app.world.schema import Entity, EntityType


VOICE_KEY = "voice"
BLENDED_SAMPLES_KEY = "blended_voice_samples"


def character_voice_for(entity: Entity | None) -> CharacterVoice | None:
    """Read ``entity.data["voice"]`` and return a ``CharacterVoice`` or None.

    Returns None if ``entity`` is None, not a CHARACTER, lacks a ``voice``
    key, or the payload is not a dict. Invalid fields fall back to
    pydantic defaults (extra keys are ignored by default).
    """
    if entity is None:
        return None
    if entity.entity_type != EntityType.CHARACTER:
        return None
    raw = entity.data.get(VOICE_KEY)
    if not isinstance(raw, dict):
        return None
    return CharacterVoice.model_validate(raw)


def blended_voice_samples_for(entity: Entity | None) -> list[str]:
    """Return author-curated blended narrator+character voice samples for this character.

    Lives on ``Entity.data["blended_voice_samples"]`` as a list of strings.
    Empty list if absent or malformed.
    """
    if entity is None:
        return []
    raw = entity.data.get(BLENDED_SAMPLES_KEY)
    if not isinstance(raw, list):
        return []
    return [s for s in raw if isinstance(s, str)]


def derive_bleed_vocabulary(voice: CharacterVoice) -> list[str]:
    """Words a POV character's register should bleed into the narration.

    Derived from jargon domains + signature phrases. Deterministic, grounded.
    """
    out: list[str] = []
    seen: set[str] = set()
    for src in (voice.jargon_domains, voice.signature_phrases):
        for w in src:
            key = w.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(w)
    return out


# Narrator register hints that a character-voice must NOT clash with.
# Keyed by the character's vocabulary_level — terms from incompatible
# registers become "excluded" during high permeability because they would
# break the free-indirect illusion (narrator pulling into a coarse POV
# should not use "heretofore"; an elevated POV should not use "ain't").
_REGISTER_INCOMPATIBLE_TERMS: dict[str, list[str]] = {
    "coarse": ["heretofore", "wherefore", "whilst", "thusly"],
    "plain": [],
    "elevated": ["ain't", "gonna", "wanna", "y'all"],
}


def derive_excluded_vocabulary(
    voice: CharacterVoice,
    narrator_register: object | None = None,
) -> list[str]:
    """Words that must NOT appear during high permeability for this character.

    Combines ``voice.forbidden_words`` with a small register-clash list
    derived from the character's ``vocabulary_level``. If a narrator
    ``StyleRegister`` is supplied and its ``diction_register`` contains a
    recognised keyword, mismatched incompatible terms are added.
    """
    out: list[str] = list(voice.forbidden_words)
    seen = {w.lower() for w in out}

    def _add(terms: Iterable[str]) -> None:
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)

    _add(_REGISTER_INCOMPATIBLE_TERMS.get(voice.vocabulary_level, []))

    # If narrator register is provided, add the clash list for the narrator's
    # opposite register when it conflicts with the character's.
    diction = getattr(narrator_register, "diction_register", None)
    if isinstance(diction, str):
        diction_l = diction.lower()
        if "elevated" in diction_l or "formal" in diction_l or "lyrical" in diction_l:
            if voice.vocabulary_level == "coarse":
                _add(_REGISTER_INCOMPATIBLE_TERMS["coarse"])
        if "coarse" in diction_l or "plain" in diction_l or "terse" in diction_l:
            if voice.vocabulary_level == "elevated":
                _add(_REGISTER_INCOMPATIBLE_TERMS["elevated"])

    return out


def _register_distance(
    narrator_register: object | None,
    voice: CharacterVoice,
) -> float:
    """Rough 0..1 distance between narrator register and character vocabulary_level.

    Used to set the baseline permeability — distant registers start lower
    (narrator is less willing to absorb the POV's voice).
    """
    diction = getattr(narrator_register, "diction_register", None)
    if not isinstance(diction, str):
        return 0.0
    diction_l = diction.lower()

    narrator_level = "plain"
    if any(k in diction_l for k in ("elevated", "formal", "lyrical", "archaic")):
        narrator_level = "elevated"
    elif any(k in diction_l for k in ("coarse", "vulgar", "gutter")):
        narrator_level = "coarse"
    elif any(k in diction_l for k in ("plain", "terse", "spare")):
        narrator_level = "plain"

    order = {"coarse": 0, "plain": 1, "elevated": 2}
    a = order.get(narrator_level, 1)
    b = order.get(voice.vocabulary_level, 1)
    return abs(a - b) / 2.0


def default_permeability(
    narrator_register: object | None,
    voice: CharacterVoice,
    *,
    blended_voice_samples: list[str] | None = None,
) -> VoicePermeability:
    """Compute a baseline ``VoicePermeability`` for a narrator-character pairing.

    Baseline drops as narrator↔character register distance grows. A more
    *direct* character pulls the narrator closer (higher baseline); a
    metaphor-rich character gives the narrator permission to lean in during
    interiority.

    ``triggers_high`` / ``triggers_low`` get defaults suitable for FIS;
    extend via ``voice.code_switching`` so scene triggers match the
    character's known context-shifts.
    """
    distance = _register_distance(narrator_register, voice)
    # Distant registers → lower baseline. Directness pulls it up.
    baseline = max(0.0, min(1.0, 0.35 - 0.25 * distance + 0.25 * (voice.directness - 0.5)))

    triggers_high = [
        "emotional extremity",
        "moment of decision",
        "interiority beat",
    ]
    if voice.uses_metaphor:
        triggers_high.append("metaphor-dense passage")
    # Code-switching contexts tell us when register shifts — treat them as
    # high-permeability moments (the character's voice is most distinct there).
    for ctx in voice.code_switching:
        triggers_high.append(f"code-switch: {ctx}")

    triggers_low = [
        "scene transition",
        "physical description",
        "establishing beat",
    ]

    bleed = derive_bleed_vocabulary(voice)
    excluded = derive_excluded_vocabulary(voice, narrator_register)

    samples = list(blended_voice_samples or [])
    if not samples:
        samples = list(voice.voice_samples)

    return VoicePermeability(
        baseline=baseline,
        current_target=baseline,
        triggers_high=triggers_high,
        triggers_low=triggers_low,
        bleed_vocabulary=bleed,
        excluded_vocabulary=excluded,
        blended_voice_samples=samples,
    )
