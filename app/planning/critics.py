"""Stub critics for the hierarchical planning pipeline.

All validators are pure functions (no LLM calls, no side effects).
Each returns a list[ValidationIssue] — never raises.
"""
from __future__ import annotations

import re

from app.planning.schemas import (
    ArcDirective,
    CraftPlan,
    DramaticPlan,
    EmotionalPlan,
)
from app.world.delta import ValidationIssue


# ---------------------------------------------------------------------------
# Core validators
# ---------------------------------------------------------------------------


def validate_arc(directive: ArcDirective) -> list[ValidationIssue]:
    """Validate an ArcDirective for schema-shape correctness."""
    issues: list[ValidationIssue] = []

    lo, hi = directive.tension_range
    if not (0.0 <= lo <= 1.0):
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"tension_range lower bound {lo!r} is outside [0.0, 1.0]",
                subject="tension_range",
            )
        )
    if not (0.0 <= hi <= 1.0):
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"tension_range upper bound {hi!r} is outside [0.0, 1.0]",
                subject="tension_range",
            )
        )
    if lo > hi:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"tension_range is reversed: min {lo!r} > max {hi!r}",
                subject="tension_range",
            )
        )

    if directive.current_phase in {"rising", "crisis"} and not directive.plot_objectives:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=(
                    f"phase '{directive.current_phase}' has no plot_objectives — "
                    "narrative momentum may stall"
                ),
                subject="plot_objectives",
            )
        )

    return issues


def validate_dramatic(
    plan: DramaticPlan,
    active_entity_ids: set[str],
    valid_tool_ids: set[str],
) -> list[ValidationIssue]:
    """Validate a DramaticPlan for character / tool referential integrity."""
    issues: list[ValidationIssue] = []

    # pov_character_id must be in active_entity_ids (when not None)
    for scene in plan.scenes:
        if scene.pov_character_id is not None and scene.pov_character_id not in active_entity_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"scene {scene.scene_id}: pov_character_id "
                        f"'{scene.pov_character_id}' not in active entities"
                    ),
                    subject=scene.pov_character_id,
                )
            )

        # characters_present — skip check when active_entity_ids is empty (fixture flexibility)
        if active_entity_ids:
            for char_id in scene.characters_present:
                if char_id not in active_entity_ids:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"scene {scene.scene_id}: characters_present "
                                f"'{char_id}' not in active entities"
                            ),
                            subject=char_id,
                        )
                    )

        # tools_used in each scene
        for tool_id in scene.tools_used:
            if tool_id not in valid_tool_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"scene {scene.scene_id}: tools_used "
                            f"'{tool_id}' not in valid tools"
                        ),
                        subject=tool_id,
                    )
                )

        # warn on missing dramatic fields
        if not scene.dramatic_question:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"scene {scene.scene_id}: missing dramatic_question",
                    subject=str(scene.scene_id),
                )
            )
        if not scene.outcome:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"scene {scene.scene_id}: missing outcome",
                    subject=str(scene.scene_id),
                )
            )
        if not scene.beats:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"scene {scene.scene_id}: missing beats",
                    subject=str(scene.scene_id),
                )
            )

    # top-level tools_selected
    for sel in plan.tools_selected:
        if sel.tool_id not in valid_tool_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"tools_selected: tool_id '{sel.tool_id}' not in valid tools",
                    subject=sel.tool_id,
                )
            )

    if not plan.suggested_choices:
        issues.append(
            ValidationIssue(
                severity="warning",
                message="suggested_choices is empty — player has no meaningful options",
                subject="suggested_choices",
            )
        )

    return issues


def validate_emotional(
    plan: EmotionalPlan,
    dramatic: DramaticPlan,
) -> list[ValidationIssue]:
    """Validate that emotional scene_ids exactly match dramatic scene_ids."""
    issues: list[ValidationIssue] = []

    dramatic_ids = {s.scene_id for s in dramatic.scenes}
    emotional_ids_list = [s.scene_id for s in plan.scenes]
    emotional_ids = set(emotional_ids_list)

    # Duplicate check
    seen: set[int] = set()
    for sid in emotional_ids_list:
        if sid in seen:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"emotional plan has duplicate scene_id {sid}",
                    subject=str(sid),
                )
            )
        seen.add(sid)

    # Missing emotional scenes
    for sid in dramatic_ids - emotional_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"dramatic scene {sid} has no emotional plan entry",
                subject=str(sid),
            )
        )

    # Extra emotional scenes not in dramatic
    for sid in emotional_ids - dramatic_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"emotional plan contains scene {sid} not present in dramatic plan",
                subject=str(sid),
            )
        )

    return issues


def validate_craft(
    plan: CraftPlan,
    dramatic: DramaticPlan,
) -> list[ValidationIssue]:
    """Validate CraftPlan scene coverage and register ratios."""
    issues: list[ValidationIssue] = []

    dramatic_ids = {s.scene_id for s in dramatic.scenes}
    craft_ids = {s.scene_id for s in plan.scenes}

    for sid in dramatic_ids - craft_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"dramatic scene {sid} has no craft plan entry",
                subject=str(sid),
            )
        )

    for sid in craft_ids - dramatic_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"craft plan contains scene {sid} not present in dramatic plan",
                subject=str(sid),
            )
        )

    for scene in plan.scenes:
        r = scene.register
        if not (0.0 <= r.concrete_abstract_ratio <= 1.0):
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"scene {scene.scene_id}: register.concrete_abstract_ratio "
                        f"{r.concrete_abstract_ratio!r} outside [0, 1]"
                    ),
                    subject=str(scene.scene_id),
                )
            )
        if not (0.0 <= r.dialogue_ratio <= 1.0):
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"scene {scene.scene_id}: register.dialogue_ratio "
                        f"{r.dialogue_ratio!r} outside [0, 1]"
                    ),
                    subject=str(scene.scene_id),
                )
            )

    # Warn if briefs list is missing entries for any scene
    brief_ids = {b.scene_id for b in plan.briefs}
    for sid in craft_ids - brief_ids:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=f"craft plan scene {sid} is missing a CraftBrief",
                subject=str(sid),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Wood-gap validators
# ---------------------------------------------------------------------------


def _word_in_prose(word: str, prose: str) -> bool:
    """Return True if *word* appears as a whole word in *prose* (case-insensitive)."""
    pattern = re.compile(r"\b" + re.escape(word.lower()) + r"\b")
    return bool(pattern.search(prose.lower()))


def validate_free_indirect_integrity(
    craft: CraftPlan,
    prose: str,
) -> list[ValidationIssue]:
    """Check bleed vocabulary presence and excluded vocabulary absence in prose."""
    issues: list[ValidationIssue] = []

    for scene in craft.scenes:
        vp = scene.voice_permeability
        if vp is None:
            continue

        if vp.current_target >= 0.5:
            # At least one bleed word should appear
            if vp.bleed_vocabulary and not any(
                _word_in_prose(w, prose) for w in vp.bleed_vocabulary
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message=(
                            f"scene {scene.scene_id}: voice_permeability current_target "
                            f"{vp.current_target} >= 0.5 but no bleed_vocabulary word "
                            f"found in prose"
                        ),
                        subject=str(scene.scene_id),
                    )
                )

        # Excluded vocabulary must never appear regardless of permeability level
        for word in vp.excluded_vocabulary:
            if _word_in_prose(word, prose):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"scene {scene.scene_id}: excluded_vocabulary word "
                            f"'{word}' found in prose"
                        ),
                        subject=str(scene.scene_id),
                    )
                )

    return issues


def validate_detail_characterization(
    craft: CraftPlan,
    prose: str,
) -> list[ValidationIssue]:
    """Warn when character-revealing detail mode finds no perceptual preoccupation in prose."""
    issues: list[ValidationIssue] = []
    prose_lower = prose.lower()

    for scene in craft.scenes:
        dp = scene.detail_principle
        if dp is None:
            continue
        if dp.detail_mode != "character_revealing":
            continue
        if not dp.perceptual_preoccupations:
            continue

        found = any(pp.lower() in prose_lower for pp in dp.perceptual_preoccupations)
        if not found:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"scene {scene.scene_id}: detail_mode is 'character_revealing' "
                        f"but no perceptual_preoccupation phrase found in prose"
                    ),
                    subject=str(scene.scene_id),
                )
            )

    return issues


def validate_metaphor_domains(
    craft: CraftPlan,
    prose: str,
) -> list[ValidationIssue]:
    """Warn if forbidden metaphor domains appear in prose."""
    issues: list[ValidationIssue] = []
    prose_lower = prose.lower()

    for scene in craft.scenes:
        for mp in scene.metaphor_profiles:
            for domain in mp.forbidden_domains:
                keyword = domain.lower()
                if keyword in prose_lower:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            message=(
                                f"scene {scene.scene_id}: forbidden metaphor domain "
                                f"'{domain}' (character '{mp.character_id}') "
                                f"appears in prose"
                            ),
                            subject=str(scene.scene_id),
                        )
                    )

    return issues


def validate_indirection(
    craft: CraftPlan,
    prose: str,
) -> list[ValidationIssue]:
    """Error if any what_not_to_say phrase appears verbatim (case-insensitive) in prose."""
    issues: list[ValidationIssue] = []
    prose_lower = prose.lower()

    for scene in craft.scenes:
        for instr in scene.indirection:
            for phrase in instr.what_not_to_say:
                if phrase.lower() in prose_lower:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"scene {scene.scene_id}: indirection violation — "
                                f"what_not_to_say phrase '{phrase}' appears verbatim "
                                f"in prose (character '{instr.character_id}')"
                            ),
                            subject=str(scene.scene_id),
                        )
                    )

    return issues


# ---- narrator-consistency critic ----

# Keyword lexicons per sensory channel. Small but representative; sufficient
# for a heuristic distribution check. Add / tune as corpus evolves.
_SENSORY_LEXICONS: dict[str, list[str]] = {
    "visual": [
        "see", "saw", "seen", "look", "looked", "looking", "watch", "watched",
        "glimpse", "stare", "stared", "gaze", "gazed", "light", "dark", "shadow",
        "bright", "dim", "color", "colour", "red", "blue", "green", "black",
        "white", "gleam", "gleamed", "flash", "flashed", "shine", "shone",
        "glow", "glowed", "silhouette", "shape",
    ],
    "auditory": [
        "hear", "heard", "hearing", "listen", "listened", "sound", "sounded",
        "noise", "silence", "silent", "quiet", "loud", "whisper", "whispered",
        "shout", "shouted", "scream", "screamed", "voice", "voices", "echo",
        "echoed", "ring", "rang", "hum", "hummed", "clatter", "crack", "thud",
        "rumble",
    ],
    "tactile": [
        "touch", "touched", "feel", "felt", "rough", "smooth", "warm", "cold",
        "hot", "cool", "wet", "dry", "soft", "hard", "sharp", "blunt", "press",
        "pressed", "grip", "gripped", "brush", "brushed", "scrape", "scraped",
        "cold", "heat", "chill",
    ],
    "olfactory": [
        "smell", "smelled", "smelt", "scent", "scented", "aroma", "odor",
        "odour", "stink", "stank", "fragrance", "reek", "reeked", "whiff",
        "perfume", "musk",
    ],
    "gustatory": [
        "taste", "tasted", "flavor", "flavour", "bitter", "sweet", "salty",
        "sour", "savory", "savoury", "tongue", "swallow", "swallowed", "sip",
        "sipped",
    ],
    "kinesthetic": [
        "step", "stepped", "walk", "walked", "run", "ran", "move", "moved",
        "turn", "turned", "reach", "reached", "lean", "leaned", "leant",
        "stretch", "stretched", "climb", "climbed", "fall", "fell", "rise",
        "rose", "balance", "sway", "swayed",
    ],
    "interoceptive": [
        "heart", "breath", "breathed", "breathing", "pulse", "ache", "ached",
        "stomach", "chest", "throat", "tight", "tightness", "flutter",
        "fluttered", "nausea", "dizzy", "dizziness", "hunger", "hungry",
        "thirst", "exhausted", "fatigue",
    ],
}


def _count_sensory_channels(prose: str) -> dict[str, int]:
    """Count whole-word keyword hits per sensory channel (case-insensitive)."""
    lower = prose.lower()
    counts: dict[str, int] = {}
    for channel, lex in _SENSORY_LEXICONS.items():
        c = 0
        for w in lex:
            # \b word boundaries
            if re.search(r"\b" + re.escape(w) + r"\b", lower):
                # count occurrences, not just presence
                c += len(re.findall(r"\b" + re.escape(w) + r"\b", lower))
        counts[channel] = c
    return counts


def _l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """L1 distance between two distributions over the same key set."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def validate_narrator_sensory_distribution(
    narrator: "object | None",
    prose: str,
    *,
    threshold: float = 0.6,
    min_total_hits: int = 6,
) -> list["ValidationIssue"]:
    """Compare prose's sensory-channel distribution to ``narrator.sensory_bias``.

    Returns a warning when the L1 distance between the normalised prose
    distribution and the narrator's target distribution exceeds ``threshold``.

    Parameters
    ----------
    narrator:
        A ``Narrator`` instance (or ``None``; in which case this returns []).
    prose:
        The generated prose to evaluate.
    threshold:
        L1 distance threshold (0..2). Default 0.6 — a distribution that
        disagrees on roughly 30% of its mass.
    min_total_hits:
        Skip the check when the prose has fewer sensory-keyword hits than
        this.  Short prose can't yield a reliable distribution.
    """
    issues: list[ValidationIssue] = []
    if narrator is None:
        return issues
    bias = getattr(narrator, "sensory_bias", None) or {}
    if not bias:
        return issues

    # Normalise target distribution (tolerate unnormalised input)
    bias_total = sum(bias.values())
    if bias_total <= 0:
        return issues
    target = {k: v / bias_total for k, v in bias.items()}

    counts = _count_sensory_channels(prose)
    total = sum(counts.values())
    if total < min_total_hits:
        return issues
    observed = {k: v / total for k, v in counts.items()}

    dist = _l1_distance(observed, target)
    if dist > threshold:
        # Identify the channels most out of line
        deltas = sorted(
            ((k, observed.get(k, 0.0) - target.get(k, 0.0)) for k in set(observed) | set(target)),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        top = ", ".join(
            f"{k}: obs={observed.get(k, 0.0):.2f} vs target={target.get(k, 0.0):.2f}"
            for k, _ in deltas[:3]
        )
        issues.append(ValidationIssue(
            severity="warning",
            message=(
                f"narrator sensory distribution drift: L1={dist:.2f} > {threshold:.2f} "
                f"({top})"
            ),
            subject="narrator.sensory_bias",
        ))
    return issues


def validate_voice_blend(
    craft: CraftPlan,  # noqa: ARG001
    prose: str,  # noqa: ARG001
) -> list[ValidationIssue]:
    """Stub — always returns [].

    Placeholder for future LLM-based voice-blend critic (P10.3).
    """
    return []


# ---- Heuristic quality critics (from voter-rollout experiment findings) ----

_SECOND_PERSON_RE = __import__("re").compile(r"\b(you|your|yours)\b", __import__("re").I)
_FIRST_PERSON_RE = __import__("re").compile(r"\b(I|me|my|mine)\b")


def validate_pov_adherence(
    prose: str,
    expected_pov: str = "second_person",
    min_ratio: float = 0.7,
) -> list[ValidationIssue]:
    """Warn when prose drifts out of its configured POV.

    For second-person POV (SV-quest default), compute
    ``|you|/(|you| + |I|)``. Ratios below ``min_ratio`` indicate drift.

    Based on the voter-rollout experiment: small models routinely collapse
    to first person under certain context conditions (e.g. action verbs
    like "confront" seemed to trigger it). This catches it.
    """
    issues: list[ValidationIssue] = []
    if expected_pov != "second_person":
        return issues  # only 2nd-person covered for v1
    yous = len(_SECOND_PERSON_RE.findall(prose))
    firsts = len(_FIRST_PERSON_RE.findall(prose))
    if yous + firsts == 0:
        return issues  # prose has neither; probably prose is all narration
    ratio = yous / (yous + firsts)
    if ratio < min_ratio:
        issues.append(ValidationIssue(
            severity="warning",
            message=(
                f"POV drift: 2nd-person ratio {ratio:.2f} below threshold "
                f"{min_ratio:.2f} ({yous} 'you', {firsts} 'I')."
            ),
        ))
    return issues


def validate_named_entity_presence(
    prose: str,
    active_entity_names: list[str],
    min_hits: int = 1,
) -> list[ValidationIssue]:
    """Warn when prose mentions zero active named characters/locations.

    Small models sometimes produce purely atmospheric prose that never
    names the entities present in the scene. That's a quality failure:
    the world is supposed to be occupied, not suggestive.
    """
    if not active_entity_names:
        return []
    import re as _re
    hits = [
        n for n in active_entity_names
        if _re.search(r"\b" + _re.escape(n) + r"\b", prose, _re.I)
    ]
    if len(hits) < min_hits:
        return [ValidationIssue(
            severity="warning",
            message=(
                f"No active named entity appears in prose "
                f"(expected at least {min_hits}; candidates: {active_entity_names})."
            ),
        )]
    return []


_STOPWORDS = {
    "about", "with", "from", "this", "that", "have", "were", "will", "been",
    "they", "your", "their", "them", "through", "while", "because", "should",
    "would", "could",
}


def validate_action_fidelity(
    prose: str,
    player_action: str,
    min_ratio: float = 0.25,
) -> list[ValidationIssue]:
    """Warn when the prose fails to execute the proposed player action.

    Extracts content words (≥4 chars, non-stopword) from the action and
    checks how many appear in the prose. Zero-hit means the pipeline
    produced a generic chapter that ignored what the player chose.
    """
    import re as _re
    tokens = [
        w.lower() for w in _re.findall(r"\b[A-Za-z]{4,}\b", player_action)
        if w.lower() not in _STOPWORDS
    ]
    if not tokens:
        return []
    hits = [
        w for w in tokens
        if _re.search(r"\b" + _re.escape(w) + r"\b", prose, _re.I)
    ]
    ratio = len(hits) / len(tokens)
    if ratio < min_ratio:
        return [ValidationIssue(
            severity="warning",
            message=(
                f"Action fidelity {ratio:.2f} below threshold {min_ratio:.2f}: "
                f"only {len(hits)}/{len(tokens)} action tokens appear in prose."
            ),
        )]
    return []
