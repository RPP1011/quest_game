from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---- shared mini-models ----
Intensity = Literal["background", "emerging", "foregrounded", "climactic"]
Urgency = Literal["immediate", "this_phase", "by_phase_end"]
AdvanceType = Literal["progresses", "complicates", "dormant", "resurfaces", "resolves"]
TransitionType = Literal["escalation", "shift", "rupture", "subsidence", "inversion", "complication"]


class ThemePriority(BaseModel):
    theme_id: str
    intensity: Intensity
    method_hint: str | None = None


class PlotObjective(BaseModel):
    description: str
    urgency: Urgency
    plot_thread_id: str | None = None


class CharacterArcDirective(BaseModel):
    character_id: str
    current_state: str
    target_state: str
    key_moment: str | None = None


# ---- ARC layer ----
class ArcDirective(BaseModel):
    current_phase: str                  # phase name from the structure
    phase_assessment: str
    theme_priorities: list[ThemePriority] = []
    plot_objectives: list[PlotObjective] = []
    character_arcs: list[CharacterArcDirective] = []
    tension_range: tuple[float, float] = (0.3, 0.7)
    hooks_to_plant: list[str] = []
    hooks_to_pay_off: list[str] = []
    parallels_to_schedule: list[str] = []


# ---- DRAMATIC layer ----
class ToolSelection(BaseModel):
    tool_id: str
    scene_id: int
    application: str


class ThreadAdvance(BaseModel):
    thread_id: str
    advance_type: AdvanceType
    description: str


class ActionResolution(BaseModel):
    kind: Literal["success", "failure", "partial", "deferred", "invalid"]
    narrative: str                      # one-sentence "what the action achieves"


class DramaticScene(BaseModel):
    scene_id: int
    pov_character_id: str | None = None
    location: str | None = None
    characters_present: list[str] = []
    dramatic_question: str
    outcome: str
    beats: list[str]
    dramatic_function: str
    tools_used: list[str] = []
    tension_target: float = 0.5
    what_can_go_wrong: str | None = None
    theme_ids: list[str] = []
    reveals: list[str] = []
    withholds: list[str] = []


class DramaticPlan(BaseModel):
    action_resolution: ActionResolution
    scenes: list[DramaticScene]
    update_tension_target: float = 0.5
    ending_hook: str
    suggested_choices: list[dict]       # reuse Choice shape: {title, description, tags}
    tools_selected: list[ToolSelection] = []
    thread_advances: list[ThreadAdvance] = []
    questions_opened: list[str] = []
    questions_closed: list[str] = []


# ---- EMOTIONAL layer ----
class CharacterEmotionalState(BaseModel):
    internal: str
    displayed: str
    gap: str | None = None


class EmotionalScenePlan(BaseModel):
    scene_id: int
    primary_emotion: str
    secondary_emotion: str | None = None
    intensity: float = 0.5
    entry_state: str
    exit_state: str
    transition_type: TransitionType
    emotional_source: str
    surface_vs_depth: str | None = None
    character_emotions: dict[str, CharacterEmotionalState] = {}


class EmotionalPlan(BaseModel):
    scenes: list[EmotionalScenePlan]
    update_emotional_arc: str
    contrast_strategy: str


# ---- CRAFT layer ----
class SceneRegister(BaseModel):
    sentence_variance: Literal["low", "medium", "high"] = "medium"
    concrete_abstract_ratio: float = 0.6  # 0..1, higher = more concrete
    interiority_depth: Literal["surface", "medium", "deep"] = "medium"
    sensory_density: Literal["sparse", "moderate", "dense"] = "moderate"
    dialogue_ratio: float = 0.3  # 0..1
    pace: Literal["compressed", "measured", "dilated"] = "measured"


class PassageOverride(BaseModel):
    trigger: str
    new_register: SceneRegister
    duration: str
    reason: str


class MotifInstruction(BaseModel):
    motif_id: str
    placement: str
    semantic_value: str
    intensity: float = 0.5


class VoiceNote(BaseModel):
    character_id: str
    instruction: str
    code_switching_active: str | None = None


class NegativeSpaceInstruction(BaseModel):
    beat_type: str
    what_is_absent: str
    how_to_render: str


class ParallelInstruction(BaseModel):
    parallel_id: str
    source_description: str
    inversion_axis: str
    execution_guidance: str


class TemporalStructure(BaseModel):
    """v1: single field describing the temporal shape of the scene in prose.
    Example: 'present-scene with one brief flashback before the final beat'."""
    description: str = "linear present-scene"


class CraftScenePlan(BaseModel):
    scene_id: int
    temporal: TemporalStructure = Field(default_factory=TemporalStructure)
    register: SceneRegister = Field(default_factory=SceneRegister)
    passage_register_overrides: list[PassageOverride] = []
    motif_instructions: list[MotifInstruction] = []
    narrator_focus: list[str] = []
    narrator_withholding: list[str] = []
    sensory_palette: dict[str, str] = {}
    voice_notes: list[VoiceNote] = []
    parallel_instruction: ParallelInstruction | None = None
    negative_space: list[NegativeSpaceInstruction] = []
    opening_instruction: str | None = None
    closing_instruction: str | None = None


class CraftPlan(BaseModel):
    scenes: list[CraftScenePlan]
