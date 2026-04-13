"""Tests for app/planning/critics.py — core and Wood-gap validators."""
from __future__ import annotations

import pytest

from app.planning.critics import (
    validate_arc,
    validate_craft,
    validate_detail_characterization,
    validate_dramatic,
    validate_emotional,
    validate_free_indirect_integrity,
    validate_indirection,
    validate_metaphor_domains,
    validate_voice_blend,
)
from app.planning.schemas import (
    ActionResolution,
    ArcDirective,
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
    DetailPrinciple,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
    IndirectionInstruction,
    MetaphorProfile,
    PlotObjective,
    SceneRegister,
    VoicePermeability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arc(
    phase: str = "rising",
    tension_range: tuple[float, float] = (0.3, 0.7),
    plot_objectives: list | None = None,
) -> ArcDirective:
    objs = plot_objectives if plot_objectives is not None else [
        PlotObjective(description="Do the thing", urgency="immediate")
    ]
    return ArcDirective(
        current_phase=phase,
        phase_assessment="stable",
        tension_range=tension_range,
        plot_objectives=objs,
    )


def _dramatic_scene(
    scene_id: int = 1,
    pov: str | None = None,
    characters: list[str] | None = None,
    tools_used: list[str] | None = None,
) -> DramaticScene:
    return DramaticScene(
        scene_id=scene_id,
        pov_character_id=pov,
        characters_present=characters or [],
        dramatic_question="Will they survive?",
        outcome="They escape narrowly",
        beats=["enters room", "hears noise", "flees"],
        dramatic_function="tension escalation",
        tools_used=tools_used or [],
    )


def _dramatic_plan(
    scenes: list[DramaticScene] | None = None,
    suggested_choices: list[dict] | None = None,
    tools_selected: list | None = None,
) -> DramaticPlan:
    return DramaticPlan(
        action_resolution=ActionResolution(kind="success", narrative="They made it"),
        scenes=scenes or [_dramatic_scene()],
        ending_hook="The door opens",
        suggested_choices=suggested_choices if suggested_choices is not None else [
            {"title": "Run", "description": "Sprint away", "tags": []}
        ],
        tools_selected=tools_selected or [],
    )


def _emotional_plan(scene_ids: list[int] | None = None) -> EmotionalPlan:
    ids = scene_ids if scene_ids is not None else [1]
    return EmotionalPlan(
        scenes=[
            EmotionalScenePlan(
                scene_id=sid,
                primary_emotion="dread",
                intensity=0.6,
                entry_state="anxious",
                exit_state="terrified",
                transition_type="escalation",
                emotional_source="threat revealed",
            )
            for sid in ids
        ],
        update_emotional_arc="dread mounts",
        contrast_strategy="silence before noise",
    )


def _craft_plan(
    scene_ids: list[int] | None = None,
    briefs: bool = True,
    concrete_abstract_ratio: float = 0.6,
    dialogue_ratio: float = 0.3,
) -> CraftPlan:
    ids = scene_ids if scene_ids is not None else [1]
    scenes = [
        CraftScenePlan(
            scene_id=sid,
            register=SceneRegister(
                concrete_abstract_ratio=concrete_abstract_ratio,
                dialogue_ratio=dialogue_ratio,
            ),
        )
        for sid in ids
    ]
    brief_list = [CraftBrief(scene_id=sid, brief="Write tense prose.") for sid in ids] if briefs else []
    return CraftPlan(scenes=scenes, briefs=brief_list)


# ---------------------------------------------------------------------------
# validate_arc
# ---------------------------------------------------------------------------


def test_arc_ok_case():
    issues = validate_arc(_arc(phase="rising"))
    assert issues == []


def test_arc_tension_range_inverted_error():
    issues = validate_arc(_arc(tension_range=(0.8, 0.2)))
    severities = [i.severity for i in issues]
    assert "error" in severities
    assert any("reversed" in i.message for i in issues)


def test_arc_tension_range_out_of_bounds_error():
    issues = validate_arc(_arc(tension_range=(-0.1, 1.2)))
    severities = {i.severity for i in issues}
    assert "error" in severities
    assert any("0" in i.message or "1" in i.message for i in issues)


def test_arc_empty_plot_objectives_in_rising_warning():
    issues = validate_arc(_arc(phase="rising", plot_objectives=[]))
    assert any(i.severity == "warning" and "plot_objectives" in i.subject for i in issues)


def test_arc_empty_plot_objectives_in_crisis_warning():
    issues = validate_arc(_arc(phase="crisis", plot_objectives=[]))
    assert any(i.severity == "warning" for i in issues)


def test_arc_empty_plot_objectives_non_critical_phase_no_warning():
    # "falling" phase with empty objectives should NOT warn
    issues = validate_arc(_arc(phase="falling", plot_objectives=[]))
    assert not any("plot_objectives" in (i.subject or "") for i in issues)


# ---------------------------------------------------------------------------
# validate_dramatic
# ---------------------------------------------------------------------------


def test_dramatic_ok():
    plan = _dramatic_plan()
    issues = validate_dramatic(plan, {"char1"}, {"tool1"})
    assert issues == []


def test_dramatic_unknown_character_error():
    scene = _dramatic_scene(pov="ghost_char", characters=["ghost_char"])
    plan = _dramatic_plan(scenes=[scene])
    issues = validate_dramatic(plan, {"hero"}, set())
    assert any(i.severity == "error" and "ghost_char" in i.message for i in issues)


def test_dramatic_unknown_tool_error():
    scene = _dramatic_scene(tools_used=["mystery_tool"])
    plan = _dramatic_plan(scenes=[scene])
    issues = validate_dramatic(plan, set(), {"valid_tool"})
    assert any(i.severity == "error" and "mystery_tool" in i.message for i in issues)


def test_dramatic_empty_choices_warning():
    plan = _dramatic_plan(suggested_choices=[])
    issues = validate_dramatic(plan, set(), set())
    assert any(i.severity == "warning" and "suggested_choices" in (i.subject or "") for i in issues)


def test_dramatic_characters_present_skip_when_entity_ids_empty():
    """When active_entity_ids is empty, skip characters_present check (fixture flexibility)."""
    scene = _dramatic_scene(characters=["unknown_char"])
    plan = _dramatic_plan(scenes=[scene])
    issues = validate_dramatic(plan, set(), set())
    # No error about unknown_char — only possible warning about empty choices
    char_errors = [i for i in issues if i.severity == "error" and "unknown_char" in i.message]
    assert char_errors == []


def test_dramatic_top_level_tool_selection_error():
    from app.planning.schemas import ToolSelection
    plan = _dramatic_plan()
    plan = plan.model_copy(
        update={"tools_selected": [ToolSelection(tool_id="bad_tool", scene_id=1, application="x")]}
    )
    issues = validate_dramatic(plan, set(), {"good_tool"})
    assert any(i.severity == "error" and "bad_tool" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_emotional
# ---------------------------------------------------------------------------


def test_emotional_scene_ids_match():
    dramatic = _dramatic_plan(scenes=[_dramatic_scene(1), _dramatic_scene(2)])
    emotional = _emotional_plan([1, 2])
    issues = validate_emotional(emotional, dramatic)
    assert issues == []


def test_emotional_missing_scene_error():
    dramatic = _dramatic_plan(scenes=[_dramatic_scene(1), _dramatic_scene(2)])
    emotional = _emotional_plan([1])  # missing scene 2
    issues = validate_emotional(emotional, dramatic)
    assert any(i.severity == "error" and "2" in i.message for i in issues)


def test_emotional_extra_scene_error():
    dramatic = _dramatic_plan(scenes=[_dramatic_scene(1)])
    emotional = _emotional_plan([1, 99])  # extra scene 99
    issues = validate_emotional(emotional, dramatic)
    assert any(i.severity == "error" and "99" in i.message for i in issues)


def test_emotional_duplicate_error():
    dramatic = _dramatic_plan(scenes=[_dramatic_scene(1)])
    emotional = _emotional_plan([1, 1])  # duplicate
    issues = validate_emotional(emotional, dramatic)
    assert any(i.severity == "error" and "duplicate" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_craft
# ---------------------------------------------------------------------------


def test_craft_ids_match_ok():
    dramatic = _dramatic_plan()
    craft = _craft_plan([1])
    issues = validate_craft(craft, dramatic)
    assert issues == []


def test_craft_bad_concrete_abstract_ratio_error():
    dramatic = _dramatic_plan()
    craft = _craft_plan([1], concrete_abstract_ratio=1.5)
    issues = validate_craft(craft, dramatic)
    assert any(i.severity == "error" and "concrete_abstract_ratio" in i.message for i in issues)


def test_craft_bad_dialogue_ratio_error():
    dramatic = _dramatic_plan()
    craft = _craft_plan([1], dialogue_ratio=-0.1)
    issues = validate_craft(craft, dramatic)
    assert any(i.severity == "error" and "dialogue_ratio" in i.message for i in issues)


def test_craft_missing_brief_warning():
    dramatic = _dramatic_plan()
    craft = _craft_plan([1], briefs=False)
    issues = validate_craft(craft, dramatic)
    assert any(i.severity == "warning" and "brief" in i.message.lower() for i in issues)


def test_craft_missing_scene_error():
    dramatic = _dramatic_plan(scenes=[_dramatic_scene(1), _dramatic_scene(2)])
    craft = _craft_plan([1])
    issues = validate_craft(craft, dramatic)
    assert any(i.severity == "error" and "2" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Wood-gap: validate_free_indirect_integrity
# ---------------------------------------------------------------------------


def test_free_indirect_missing_bleed_warning():
    scene = CraftScenePlan(
        scene_id=1,
        voice_permeability=VoicePermeability(
            current_target=0.7,
            bleed_vocabulary=["reckon", "ain't"],
        ),
    )
    craft = CraftPlan(scenes=[scene])
    prose = "She walked into the room and observed the surroundings carefully."
    issues = validate_free_indirect_integrity(craft, prose)
    assert any(i.severity == "warning" and "bleed_vocabulary" in i.message for i in issues)


def test_free_indirect_bleed_present_no_warning():
    scene = CraftScenePlan(
        scene_id=1,
        voice_permeability=VoicePermeability(
            current_target=0.7,
            bleed_vocabulary=["reckon"],
        ),
    )
    craft = CraftPlan(scenes=[scene])
    # "reckon" appears as a whole word (not just as a substring of "reckoned")
    prose = "She had to reckon with the open door."
    issues = validate_free_indirect_integrity(craft, prose)
    assert not any(i.severity == "warning" for i in issues)


def test_free_indirect_excluded_vocab_error():
    scene = CraftScenePlan(
        scene_id=1,
        voice_permeability=VoicePermeability(
            current_target=0.3,
            excluded_vocabulary=["furthermore", "nevertheless"],
        ),
    )
    craft = CraftPlan(scenes=[scene])
    prose = "Furthermore, she decided to leave."
    issues = validate_free_indirect_integrity(craft, prose)
    assert any(i.severity == "error" and "furthermore" in i.message.lower() for i in issues)


def test_free_indirect_none_voice_permeability_ok():
    craft = CraftPlan(scenes=[CraftScenePlan(scene_id=1)])
    issues = validate_free_indirect_integrity(craft, "Any prose here.")
    assert issues == []


# ---------------------------------------------------------------------------
# Wood-gap: validate_detail_characterization
# ---------------------------------------------------------------------------


def test_detail_characterization_missing_preoccupation_warning():
    scene = CraftScenePlan(
        scene_id=1,
        detail_principle=DetailPrinciple(
            perceiving_character_id="aria",
            perceptual_preoccupations=["rust stains", "broken hinges"],
            detail_mode="character_revealing",
        ),
    )
    craft = CraftPlan(scenes=[scene])
    prose = "The door was plain and unremarkable."
    issues = validate_detail_characterization(craft, prose)
    assert any(i.severity == "warning" for i in issues)


def test_detail_characterization_present_ok():
    scene = CraftScenePlan(
        scene_id=1,
        detail_principle=DetailPrinciple(
            perceiving_character_id="aria",
            perceptual_preoccupations=["rust stains"],
            detail_mode="character_revealing",
        ),
    )
    craft = CraftPlan(scenes=[scene])
    prose = "She noticed the rust stains creeping down the frame."
    issues = validate_detail_characterization(craft, prose)
    assert issues == []


def test_detail_characterization_non_revealing_mode_skipped():
    scene = CraftScenePlan(
        scene_id=1,
        detail_principle=DetailPrinciple(
            perceiving_character_id="aria",
            perceptual_preoccupations=["rust stains"],
            detail_mode="mood_setting",
        ),
    )
    craft = CraftPlan(scenes=[scene])
    prose = "The room was empty."
    issues = validate_detail_characterization(craft, prose)
    assert issues == []


# ---------------------------------------------------------------------------
# Wood-gap: validate_metaphor_domains
# ---------------------------------------------------------------------------


def test_metaphor_forbidden_domain_warning():
    scene = CraftScenePlan(
        scene_id=1,
        metaphor_profiles=[
            MetaphorProfile(
                character_id="kai",
                permanent_domains=["sea", "weather"],
                forbidden_domains=["technology", "sports"],
            )
        ],
    )
    craft = CraftPlan(scenes=[scene])
    prose = "His thoughts moved like a sports car through traffic."
    issues = validate_metaphor_domains(craft, prose)
    assert any(i.severity == "warning" and "sports" in i.message for i in issues)


def test_metaphor_no_forbidden_domain_ok():
    scene = CraftScenePlan(
        scene_id=1,
        metaphor_profiles=[
            MetaphorProfile(
                character_id="kai",
                permanent_domains=["sea"],
                forbidden_domains=["technology"],
            )
        ],
    )
    craft = CraftPlan(scenes=[scene])
    prose = "The ship rose on the swell."
    issues = validate_metaphor_domains(craft, prose)
    assert issues == []


# ---------------------------------------------------------------------------
# Wood-gap: validate_indirection
# ---------------------------------------------------------------------------


def test_indirection_violation_error():
    scene = CraftScenePlan(
        scene_id=1,
        indirection=[
            IndirectionInstruction(
                character_id="val",
                unconscious_motive="jealousy",
                surface_manifestations=["sarcastic comments"],
                detail_tells=["avoids eye contact"],
                what_not_to_say=["she was jealous", "she envied him"],
                reader_should_infer="Val resents Dani's success",
            )
        ],
    )
    craft = CraftPlan(scenes=[scene])
    prose = "She was jealous of everything he had achieved."
    issues = validate_indirection(craft, prose)
    assert any(i.severity == "error" and "she was jealous" in i.message.lower() for i in issues)


def test_indirection_no_violation_ok():
    scene = CraftScenePlan(
        scene_id=1,
        indirection=[
            IndirectionInstruction(
                character_id="val",
                unconscious_motive="jealousy",
                surface_manifestations=["sarcasm"],
                detail_tells=["avoids eye contact"],
                what_not_to_say=["she was jealous"],
                reader_should_infer="reader infers it",
            )
        ],
    )
    craft = CraftPlan(scenes=[scene])
    prose = "She glanced away when he mentioned the award."
    issues = validate_indirection(craft, prose)
    assert issues == []


# ---------------------------------------------------------------------------
# Wood-gap: validate_voice_blend (stub)
# ---------------------------------------------------------------------------


def test_voice_blend_always_ok():
    craft = _craft_plan([1])
    issues = validate_voice_blend(craft, "Any prose at all.")
    assert issues == []


def test_voice_blend_always_ok_with_complex_craft():
    """Stub returns [] regardless of input complexity."""
    dramatic = _dramatic_plan(scenes=[_dramatic_scene(1), _dramatic_scene(2)])
    craft = _craft_plan([1, 2])
    issues = validate_voice_blend(craft, "Complex prose with many characters speaking.")
    assert issues == []


# ---------------------------------------------------------------------------
# validate_narrator_sensory_distribution
# ---------------------------------------------------------------------------
def test_narrator_sensory_distribution_none_narrator_is_no_issues():
    from app.planning.critics import validate_narrator_sensory_distribution
    assert validate_narrator_sensory_distribution(None, "any prose") == []


def test_narrator_sensory_distribution_empty_bias_is_no_issues():
    from app.craft.schemas import Narrator
    from app.planning.critics import validate_narrator_sensory_distribution
    n = Narrator()
    assert validate_narrator_sensory_distribution(n, "any prose") == []


def test_narrator_sensory_distribution_short_prose_skipped():
    from app.craft.schemas import Narrator
    from app.planning.critics import validate_narrator_sensory_distribution
    n = Narrator(sensory_bias={"visual": 1.0})
    # only 1 hit — below min_total_hits default
    assert validate_narrator_sensory_distribution(n, "He saw nothing.") == []


def test_narrator_sensory_distribution_matching_is_no_warning():
    from app.craft.schemas import Narrator
    from app.planning.critics import validate_narrator_sensory_distribution
    n = Narrator(sensory_bias={"visual": 1.0})
    # Many visual words — matches target
    prose = (
        "He saw the light. She looked at the shadow. "
        "The red glow of the lamp gleamed. He watched the dim color. "
        "The bright silhouette shone in the dark."
    )
    assert validate_narrator_sensory_distribution(n, prose) == []


def test_narrator_sensory_distribution_mismatch_warns():
    from app.craft.schemas import Narrator
    from app.planning.critics import validate_narrator_sensory_distribution
    # Target is heavily auditory but prose is heavily interoceptive+kinesthetic
    n = Narrator(sensory_bias={"auditory": 1.0})
    prose = (
        "His heart ached. His chest tightened. He breathed hard. "
        "He walked. He ran. He turned. His stomach fluttered. "
        "He leaned. His pulse raced. His breath came short."
    )
    issues = validate_narrator_sensory_distribution(n, prose)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "narrator sensory" in issues[0].message.lower()
