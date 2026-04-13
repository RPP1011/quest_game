"""Tests for app/planning/schemas.py — round-trips, defaults, required fields, enum validation."""

import pytest
from pydantic import ValidationError

from app.planning.schemas import (
    ActionResolution,
    ArcDirective,
    CharacterArcDirective,
    CharacterEmotionalState,
    CraftPlan,
    CraftScenePlan,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
    MotifInstruction,
    NegativeSpaceInstruction,
    ParallelInstruction,
    PassageOverride,
    PlotObjective,
    SceneRegister,
    TemporalStructure,
    ThemePriority,
    ThreadAdvance,
    ToolSelection,
    VoiceNote,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arc_directive_minimal() -> ArcDirective:
    return ArcDirective(
        current_phase="rising_action",
        phase_assessment="Tension is building.",
    )


def _dramatic_plan_minimal() -> DramaticPlan:
    scene = DramaticScene(
        scene_id=1,
        dramatic_question="Will the hero escape?",
        outcome="The hero narrowly escapes.",
        beats=["Hero spots guard", "Hero slips past"],
        dramatic_function="obstacle",
    )
    return DramaticPlan(
        action_resolution=ActionResolution(kind="partial", narrative="Hero escapes but loses gear."),
        scenes=[scene],
        ending_hook="A door slams behind them.",
        suggested_choices=[
            {"title": "Press on", "description": "Continue into the dark.", "tags": ["brave"]},
        ],
    )


def _emotional_plan_minimal() -> EmotionalPlan:
    esp = EmotionalScenePlan(
        scene_id=1,
        primary_emotion="fear",
        intensity=0.8,
        entry_state="nervous",
        exit_state="relieved",
        transition_type="shift",
        emotional_source="imminent danger",
    )
    return EmotionalPlan(
        scenes=[esp],
        update_emotional_arc="Fear gives way to cautious hope.",
        contrast_strategy="Follow fear with a beat of dark humour.",
    )


def _craft_plan_minimal() -> CraftPlan:
    csp = CraftScenePlan(scene_id=1)
    return CraftPlan(scenes=[csp])


# ---------------------------------------------------------------------------
# ArcDirective
# ---------------------------------------------------------------------------

class TestArcDirective:
    def test_round_trip_json(self):
        original = _arc_directive_minimal()
        restored = ArcDirective.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_defaults(self):
        d = _arc_directive_minimal()
        assert d.theme_priorities == []
        assert d.plot_objectives == []
        assert d.character_arcs == []
        assert d.tension_range == (0.3, 0.7)
        assert d.hooks_to_plant == []
        assert d.hooks_to_pay_off == []
        assert d.parallels_to_schedule == []

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ArcDirective(phase_assessment="Missing current_phase.")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ArcDirective(current_phase="rising_action")  # type: ignore[call-arg]

    def test_full_round_trip(self):
        d = ArcDirective(
            current_phase="crisis",
            phase_assessment="Peak tension.",
            theme_priorities=[ThemePriority(theme_id="t1", intensity="climactic", method_hint="Lean in")],
            plot_objectives=[PlotObjective(description="Stop the villain", urgency="immediate", plot_thread_id="pt1")],
            character_arcs=[CharacterArcDirective(
                character_id="hero",
                current_state="doubtful",
                target_state="resolved",
                key_moment="The speech",
            )],
            tension_range=(0.7, 0.9),
            hooks_to_plant=["The locked box"],
            hooks_to_pay_off=["The scar"],
            parallels_to_schedule=["mirror scene"],
        )
        restored = ArcDirective.model_validate_json(d.model_dump_json())
        assert restored == d

    def test_bad_intensity_enum(self):
        with pytest.raises(ValidationError):
            ThemePriority(theme_id="t1", intensity="catastrophic")  # type: ignore[arg-type]

    def test_bad_urgency_enum(self):
        with pytest.raises(ValidationError):
            PlotObjective(description="do thing", urgency="eventually")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DramaticPlan
# ---------------------------------------------------------------------------

class TestDramaticPlan:
    def test_round_trip_json(self):
        original = _dramatic_plan_minimal()
        restored = DramaticPlan.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_defaults(self):
        p = _dramatic_plan_minimal()
        assert p.update_tension_target == 0.5
        assert p.tools_selected == []
        assert p.thread_advances == []
        assert p.questions_opened == []
        assert p.questions_closed == []

    def test_dramatic_scene_requires_dramatic_question(self):
        with pytest.raises(ValidationError):
            DramaticScene(
                scene_id=1,
                # missing dramatic_question
                outcome="Hero escapes.",
                beats=["Beat 1"],
                dramatic_function="obstacle",
            )  # type: ignore[call-arg]

    def test_dramatic_scene_requires_outcome(self):
        with pytest.raises(ValidationError):
            DramaticScene(
                scene_id=1,
                dramatic_question="Will they make it?",
                # missing outcome
                beats=["Beat 1"],
                dramatic_function="obstacle",
            )  # type: ignore[call-arg]

    def test_dramatic_scene_requires_beats(self):
        with pytest.raises(ValidationError):
            DramaticScene(
                scene_id=1,
                dramatic_question="Will they make it?",
                outcome="Yes.",
                # missing beats
                dramatic_function="obstacle",
            )  # type: ignore[call-arg]

    def test_bad_action_resolution_kind_enum(self):
        with pytest.raises(ValidationError):
            ActionResolution(kind="fumble", narrative="Nothing happened.")  # type: ignore[arg-type]

    def test_bad_advance_type_enum(self):
        with pytest.raises(ValidationError):
            ThreadAdvance(thread_id="t1", advance_type="explodes", description="boom")  # type: ignore[arg-type]

    def test_full_round_trip(self):
        plan = DramaticPlan(
            action_resolution=ActionResolution(kind="success", narrative="Hero wins."),
            scenes=[
                DramaticScene(
                    scene_id=1,
                    pov_character_id="hero",
                    location="castle",
                    characters_present=["hero", "villain"],
                    dramatic_question="Will hero prevail?",
                    outcome="Hero prevails.",
                    beats=["Confrontation", "Strike"],
                    dramatic_function="climax",
                    tools_used=["contrast"],
                    tension_target=0.9,
                    what_can_go_wrong="Villain has backup.",
                    theme_ids=["power"],
                    reveals=["villain's motive"],
                    withholds=["hero's weakness"],
                )
            ],
            update_tension_target=0.8,
            ending_hook="A final warning echoes.",
            suggested_choices=[{"title": "Flee", "description": "Run away.", "tags": []}],
            tools_selected=[ToolSelection(tool_id="contrast", scene_id=1, application="Use light/dark")],
            thread_advances=[ThreadAdvance(thread_id="main", advance_type="progresses", description="Moved forward.")],
            questions_opened=["What is next?"],
            questions_closed=["Is villain defeated?"],
        )
        restored = DramaticPlan.model_validate_json(plan.model_dump_json())
        assert restored == plan


# ---------------------------------------------------------------------------
# EmotionalPlan
# ---------------------------------------------------------------------------

class TestEmotionalPlan:
    def test_round_trip_json(self):
        original = _emotional_plan_minimal()
        restored = EmotionalPlan.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_defaults(self):
        esp = EmotionalScenePlan(
            scene_id=1,
            primary_emotion="joy",
            entry_state="neutral",
            exit_state="elated",
            transition_type="escalation",
            emotional_source="unexpected kindness",
        )
        assert esp.secondary_emotion is None
        assert esp.intensity == 0.5
        assert esp.surface_vs_depth is None
        assert esp.character_emotions == {}

    def test_character_emotional_state_gap_none(self):
        ces = CharacterEmotionalState(internal="terrified", displayed="calm")
        assert ces.gap is None
        restored = CharacterEmotionalState.model_validate_json(ces.model_dump_json())
        assert restored.gap is None

    def test_bad_transition_type_enum(self):
        with pytest.raises(ValidationError):
            EmotionalScenePlan(
                scene_id=1,
                primary_emotion="fear",
                entry_state="nervous",
                exit_state="calm",
                transition_type="explosion",  # type: ignore[arg-type]
                emotional_source="danger",
            )

    def test_full_round_trip(self):
        plan = EmotionalPlan(
            scenes=[
                EmotionalScenePlan(
                    scene_id=1,
                    primary_emotion="dread",
                    secondary_emotion="curiosity",
                    intensity=0.9,
                    entry_state="tense",
                    exit_state="resigned",
                    transition_type="rupture",
                    emotional_source="betrayal revealed",
                    surface_vs_depth="Smiles while devastated inside",
                    character_emotions={
                        "hero": CharacterEmotionalState(
                            internal="heartbroken",
                            displayed="stoic",
                            gap="Hiding grief behind duty",
                        )
                    },
                )
            ],
            update_emotional_arc="Dread transitions to acceptance.",
            contrast_strategy="Follow heavy scene with a moment of levity.",
        )
        restored = EmotionalPlan.model_validate_json(plan.model_dump_json())
        assert restored == plan


# ---------------------------------------------------------------------------
# CraftPlan
# ---------------------------------------------------------------------------

class TestCraftPlan:
    def test_round_trip_json(self):
        original = _craft_plan_minimal()
        restored = CraftPlan.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_scene_register_defaults(self):
        reg = SceneRegister()
        assert reg.sentence_variance == "medium"
        assert reg.concrete_abstract_ratio == 0.6
        assert reg.interiority_depth == "medium"
        assert reg.sensory_density == "moderate"
        assert reg.dialogue_ratio == 0.3
        assert reg.pace == "measured"

    def test_craft_scene_plan_defaults(self):
        csp = CraftScenePlan(scene_id=42)
        assert csp.temporal.description == "linear present-scene"
        assert csp.register.pace == "measured"
        assert csp.passage_register_overrides == []
        assert csp.motif_instructions == []
        assert csp.narrator_focus == []
        assert csp.narrator_withholding == []
        assert csp.sensory_palette == {}
        assert csp.voice_notes == []
        assert csp.parallel_instruction is None
        assert csp.negative_space == []
        assert csp.opening_instruction is None
        assert csp.closing_instruction is None

    def test_bad_sentence_variance_enum(self):
        with pytest.raises(ValidationError):
            SceneRegister(sentence_variance="very_high")  # type: ignore[arg-type]

    def test_bad_interiority_depth_enum(self):
        with pytest.raises(ValidationError):
            SceneRegister(interiority_depth="extreme")  # type: ignore[arg-type]

    def test_bad_pace_enum(self):
        with pytest.raises(ValidationError):
            SceneRegister(pace="frantic")  # type: ignore[arg-type]

    def test_full_round_trip(self):
        plan = CraftPlan(
            scenes=[
                CraftScenePlan(
                    scene_id=1,
                    temporal=TemporalStructure(description="Present with one flashback"),
                    register=SceneRegister(
                        sentence_variance="high",
                        concrete_abstract_ratio=0.8,
                        interiority_depth="deep",
                        sensory_density="dense",
                        dialogue_ratio=0.4,
                        pace="compressed",
                    ),
                    passage_register_overrides=[
                        PassageOverride(
                            trigger="flashback begins",
                            new_register=SceneRegister(pace="dilated"),
                            duration="one paragraph",
                            reason="Slow time for memory",
                        )
                    ],
                    motif_instructions=[
                        MotifInstruction(
                            motif_id="m1",
                            placement="opening line",
                            semantic_value="death",
                            intensity=0.7,
                        )
                    ],
                    narrator_focus=["hero's hands"],
                    narrator_withholding=["villain's face"],
                    sensory_palette={"sight": "dim grey light", "sound": "distant thunder"},
                    voice_notes=[
                        VoiceNote(
                            character_id="hero",
                            instruction="Clipped sentences under stress.",
                            code_switching_active="formal register slips",
                        )
                    ],
                    parallel_instruction=ParallelInstruction(
                        parallel_id="p1",
                        source_description="Opening scene reunion",
                        inversion_axis="joy vs grief",
                        execution_guidance="Mirror the framing but invert the tone.",
                    ),
                    negative_space=[
                        NegativeSpaceInstruction(
                            beat_type="farewell",
                            what_is_absent="the words never spoken",
                            how_to_render="Pause, then cut away",
                        )
                    ],
                    opening_instruction="Begin mid-action.",
                    closing_instruction="End on a lingering image.",
                )
            ]
        )
        restored = CraftPlan.model_validate_json(plan.model_dump_json())
        assert restored == plan
