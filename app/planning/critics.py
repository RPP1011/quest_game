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


def validate_voice_blend(
    craft: CraftPlan,  # noqa: ARG001
    prose: str,  # noqa: ARG001
) -> list[ValidationIssue]:
    """Stub — always returns [].

    Placeholder for future LLM-based voice-blend critic (P10.3).
    """
    return []
