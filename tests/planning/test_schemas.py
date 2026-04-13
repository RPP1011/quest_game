"""Tests for app/planning/schemas.py — round-trips, defaults, required fields, enum validation."""

import pytest
from pydantic import ValidationError

from app.planning.schemas import (
    ActionResolution,
    ArcDirective,
    CharacterArcDirective,
    CharacterEmotionalState,
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
    MotifInstruction,
    NegativeSpaceInstruction,
    ParallelInstruction,
    PassageOverride,
    PassagePermeability,
    PlotObjective,
    SceneRegister,
    TemporalStructure,
    ThemePriority,
    ThreadAdvance,
    ToolSelection,
    VoiceNote,
    VoicePermeability,
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


# ---------------------------------------------------------------------------
# Wood-gap schemas
# ---------------------------------------------------------------------------

class TestVoicePermeability:
    def test_round_trip_json(self):
        vp = VoicePermeability(
            baseline=0.4,
            current_target=0.7,
            triggers_high=["emotional extremity", "moment of decision"],
            triggers_low=["scene transition"],
            bleed_vocabulary=["bloody", "damn"],
            excluded_vocabulary=["indeed", "furthermore"],
            blended_voice_samples=["She'd seen worse — hadn't she?"],
        )
        restored = VoicePermeability.model_validate_json(vp.model_dump_json())
        assert restored == vp

    def test_defaults(self):
        vp = VoicePermeability()
        assert vp.baseline == 0.3
        assert vp.current_target == 0.3
        assert vp.triggers_high == []
        assert vp.triggers_low == []
        assert vp.bleed_vocabulary == []
        assert vp.excluded_vocabulary == []
        assert vp.blended_voice_samples == []

    def test_baseline_bounds_enforced(self):
        with pytest.raises(ValidationError):
            VoicePermeability(baseline=-0.1)
        with pytest.raises(ValidationError):
            VoicePermeability(baseline=1.1)

    def test_current_target_bounds_enforced(self):
        with pytest.raises(ValidationError):
            VoicePermeability(current_target=-0.01)
        with pytest.raises(ValidationError):
            VoicePermeability(current_target=1.01)

    def test_boundary_values_valid(self):
        vp = VoicePermeability(baseline=0.0, current_target=1.0)
        assert vp.baseline == 0.0
        assert vp.current_target == 1.0


class TestPassagePermeability:
    def test_round_trip_json(self):
        pp = PassagePermeability(
            passage_description="Hero confronts the captain",
            target=0.8,
            character_id="hero",
            bleed_words=["bloody", "right"],
            reason="High emotional intensity pulls voice close",
        )
        restored = PassagePermeability.model_validate_json(pp.model_dump_json())
        assert restored == pp

    def test_defaults(self):
        pp = PassagePermeability(
            passage_description="Description passage",
            target=0.5,
            character_id="hero",
            reason="Moderate blend during exposition",
        )
        assert pp.bleed_words == []

    def test_target_bounds_enforced(self):
        with pytest.raises(ValidationError):
            PassagePermeability(
                passage_description="test",
                target=1.5,
                character_id="hero",
                reason="test",
            )
        with pytest.raises(ValidationError):
            PassagePermeability(
                passage_description="test",
                target=-0.1,
                character_id="hero",
                reason="test",
            )


class TestDetailPrinciple:
    def test_round_trip_json(self):
        dp = DetailPrinciple(
            perceiving_character_id="hero",
            perceptual_preoccupations=["exits", "weapons", "the captain's hands"],
            detail_mode="character_revealing",
            triple_duty_targets=["the locked door"],
        )
        restored = DetailPrinciple.model_validate_json(dp.model_dump_json())
        assert restored == dp

    def test_default_detail_mode(self):
        dp = DetailPrinciple(
            perceiving_character_id="hero",
            perceptual_preoccupations=["exits"],
        )
        assert dp.detail_mode == "character_revealing"

    def test_valid_detail_modes(self):
        for mode in ("character_revealing", "world_establishing", "thematic_resonant",
                     "mood_setting", "foreshadowing", "ironic"):
            dp = DetailPrinciple(
                perceiving_character_id="hero",
                perceptual_preoccupations=["something"],
                detail_mode=mode,  # type: ignore[arg-type]
            )
            assert dp.detail_mode == mode

    def test_bad_detail_mode_enum(self):
        with pytest.raises(ValidationError):
            DetailPrinciple(
                perceiving_character_id="hero",
                perceptual_preoccupations=["exits"],
                detail_mode="random",  # type: ignore[arg-type]
            )

    def test_triple_duty_default_empty(self):
        dp = DetailPrinciple(
            perceiving_character_id="hero",
            perceptual_preoccupations=["exits"],
        )
        assert dp.triple_duty_targets == []


class TestMetaphorProfile:
    def test_round_trip_json(self):
        mp = MetaphorProfile(
            character_id="hero",
            permanent_domains=["military", "weather"],
            current_domains=["fire"],
            forbidden_domains=["sailing"],
            metaphor_density="regular",
            extends_to_narration=False,
        )
        restored = MetaphorProfile.model_validate_json(mp.model_dump_json())
        assert restored == mp

    def test_default_metaphor_density(self):
        mp = MetaphorProfile(
            character_id="hero",
            permanent_domains=["military"],
        )
        assert mp.metaphor_density == "occasional"

    def test_defaults(self):
        mp = MetaphorProfile(
            character_id="hero",
            permanent_domains=["military"],
        )
        assert mp.current_domains == []
        assert mp.forbidden_domains == []
        assert mp.extends_to_narration is True

    def test_valid_metaphor_density_values(self):
        for density in ("sparse", "occasional", "regular", "rich"):
            mp = MetaphorProfile(
                character_id="hero",
                permanent_domains=["military"],
                metaphor_density=density,  # type: ignore[arg-type]
            )
            assert mp.metaphor_density == density

    def test_bad_metaphor_density_enum(self):
        with pytest.raises(ValidationError):
            MetaphorProfile(
                character_id="hero",
                permanent_domains=["military"],
                metaphor_density="overwhelming",  # type: ignore[arg-type]
            )


class TestIndirectionInstruction:
    def test_round_trip_json(self):
        ii = IndirectionInstruction(
            character_id="hero",
            unconscious_motive="Guilt about abandoning family",
            surface_manifestations=["checks exits obsessively", "flinches at children's voices"],
            detail_tells=["worn photograph in breast pocket", "hesitation at the gate"],
            what_not_to_say=["She missed them.", "Guilt gnawed at her."],
            reader_should_infer="Hero is running from something at home.",
        )
        restored = IndirectionInstruction.model_validate_json(ii.model_dump_json())
        assert restored == ii

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            IndirectionInstruction(
                character_id="hero",
                # missing unconscious_motive
                surface_manifestations=[],
                detail_tells=[],
                what_not_to_say=[],
                reader_should_infer="infer something",
            )  # type: ignore[call-arg]


class TestCraftBrief:
    def test_round_trip_json(self):
        cb = CraftBrief(
            scene_id=1,
            brief=(
                "This scene is a chess match played in pleasantries. The hero needs "
                "information the captain has; the captain suspects the hero knows more "
                "than she lets on. Write it in close third, pulled tight to her "
                "preoccupations: exits, the captain's hands, the weight of the satchel "
                "on her shoulder. Let the room do the work — the locked door, the "
                "untouched wine. She doesn't trust easily, so her metaphors run to "
                "weather and terrain, never warmth. End before resolution; the reader "
                "should feel the conversation could tip either way."
            ),
        )
        restored = CraftBrief.model_validate_json(cb.model_dump_json())
        assert restored == cb

    def test_requires_scene_id(self):
        with pytest.raises(ValidationError):
            CraftBrief(brief="A prose director's note.")  # type: ignore[call-arg]

    def test_requires_brief(self):
        with pytest.raises(ValidationError):
            CraftBrief(scene_id=1)  # type: ignore[call-arg]


class TestCraftScenePlanWoodFields:
    def test_backwards_compat_no_wood_fields(self):
        """Existing CraftScenePlan with no Wood fields still validates."""
        csp = CraftScenePlan(scene_id=5)
        assert csp.voice_permeability is None
        assert csp.passage_permeabilities == []
        assert csp.detail_principle is None
        assert csp.metaphor_profiles == []
        assert csp.indirection == []

    def test_backwards_compat_round_trip(self):
        """CraftScenePlan without Wood fields round-trips identically."""
        csp = CraftScenePlan(scene_id=5)
        restored = CraftScenePlan.model_validate_json(csp.model_dump_json())
        assert restored == csp

    def test_with_all_wood_fields_round_trip(self):
        csp = CraftScenePlan(
            scene_id=2,
            voice_permeability=VoicePermeability(
                baseline=0.2,
                current_target=0.6,
                triggers_high=["decision moment"],
                bleed_vocabulary=["damn"],
            ),
            passage_permeabilities=[
                PassagePermeability(
                    passage_description="Opening paragraph",
                    target=0.3,
                    character_id="hero",
                    reason="Establishing distance",
                )
            ],
            detail_principle=DetailPrinciple(
                perceiving_character_id="hero",
                perceptual_preoccupations=["exits", "hands"],
                detail_mode="character_revealing",
            ),
            metaphor_profiles=[
                MetaphorProfile(
                    character_id="hero",
                    permanent_domains=["military"],
                    metaphor_density="sparse",
                )
            ],
            indirection=[
                IndirectionInstruction(
                    character_id="hero",
                    unconscious_motive="Grief",
                    surface_manifestations=["avoids mirrors"],
                    detail_tells=["cracked locket"],
                    what_not_to_say=["She grieved."],
                    reader_should_infer="Hero is mourning.",
                )
            ],
        )
        restored = CraftScenePlan.model_validate_json(csp.model_dump_json())
        assert restored == csp


class TestCraftPlanWithBriefs:
    def test_briefs_default_empty(self):
        plan = CraftPlan(scenes=[CraftScenePlan(scene_id=1)])
        assert plan.briefs == []

    def test_with_scenes_and_briefs_round_trip(self):
        plan = CraftPlan(
            scenes=[
                CraftScenePlan(scene_id=1),
                CraftScenePlan(scene_id=2),
            ],
            briefs=[
                CraftBrief(scene_id=1, brief="Director's note for scene one."),
                CraftBrief(scene_id=2, brief="Director's note for scene two."),
            ],
        )
        restored = CraftPlan.model_validate_json(plan.model_dump_json())
        assert restored == plan
        assert len(restored.scenes) == 2
        assert len(restored.briefs) == 2
        assert restored.briefs[0].scene_id == 1
        assert restored.briefs[1].brief == "Director's note for scene two."
