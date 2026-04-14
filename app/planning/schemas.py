from __future__ import annotations

import re as _re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _coerce_scene_id(v: object) -> int:
    """Coerce a scene_id to int.

    Day 11: small models routinely emit ``scene_id: "scene_1"``,
    ``scene_id: "1"``, or ``scene_id: 42`` (literal hallucinated default)
    despite the schema declaring ``int``. xgrammar/strict structured
    output catches the literal cases but the schema-name-as-string case
    still slips through when callers pre-validate. This validator pulls
    the first integer substring out of any input and coerces.
    """
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        m = _re.search(r"-?\d+", v)
        if m:
            return int(m.group(0))
    raise ValueError(f"cannot coerce scene_id from {v!r}")

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

    @field_validator("scene_id", mode="before")
    @classmethod
    def _coerce_scene_id(cls, v):
        return _coerce_scene_id(v)


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

    @field_validator("scene_id", mode="before")
    @classmethod
    def _coerce_scene_id(cls, v):
        return _coerce_scene_id(v)


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
    expectations_set: list[str] = []
    expectations_subverted: list[str] = []


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

    @field_validator("scene_id", mode="before")
    @classmethod
    def _coerce_scene_id(cls, v):
        return _coerce_scene_id(v)


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
    # --- Wood-gap fields ---
    voice_permeability: VoicePermeability | None = None
    passage_permeabilities: list[PassagePermeability] = []
    detail_principle: DetailPrinciple | None = None
    metaphor_profiles: list[MetaphorProfile] = []
    indirection: list[IndirectionInstruction] = []

    @field_validator("scene_id", mode="before")
    @classmethod
    def _coerce_scene_id(cls, v):
        return _coerce_scene_id(v)


class CraftPlan(BaseModel):
    scenes: list[CraftScenePlan]
    briefs: list[CraftBrief] = []


# ---- Wood gaps ----

class VoicePermeability(BaseModel):
    """How much the narrator's register absorbs the POV character's language.

    Not static. Fluctuates within a scene — pulls close during emotional
    intensity, retreats during transition or description. 0.0 = pure
    narrator, 1.0 = pure character stream-of-consciousness.
    """
    baseline: float = Field(default=0.3, ge=0.0, le=1.0)
    current_target: float = Field(default=0.3, ge=0.0, le=1.0)
    triggers_high: list[str] = []      # "emotional extremity", "moment of decision", ...
    triggers_low: list[str] = []       # "scene transition", "physical description", ...
    bleed_vocabulary: list[str] = []   # character-register words to appear in narration
    excluded_vocabulary: list[str] = []  # narrator words unsuitable during high permeability
    blended_voice_samples: list[str] = []  # few-shot samples of the blend


class PerceptualProfile(BaseModel):
    """Persistent per-character detail-selection profile (Gap G9).

    Grounds ``DetailPrinciple.perceptual_preoccupations`` in stable character
    data rather than per-scene LLM invention. A soldier notices exits and
    sightlines; a merchant notices quality and wealth. That preference is a
    property of the character, not of the scene.

    Intended to live on ``Entity.data["perception"]`` for CHARACTER entities
    and be loaded via ``app.planning.perception.perceptual_profile_for``.
    """
    permanent_preoccupations: list[str] = Field(default_factory=list)
    # emotion label (lowercase) → preoccupations activated when that emotion
    # is the scene's primary emotion. e.g. {"dread": ["exits", "footsteps"]}
    emotional_preoccupations: dict[str, list[str]] = Field(default_factory=dict)
    detail_mode: str = "precise"          # e.g. "precise", "impressionistic", "obsessive"
    triple_duty_targets: list[str] = Field(default_factory=list)


class CharacterMetaphorProfile(BaseModel):
    """Persistent per-character metaphor source domains (Gap G10).

    Grounds ``MetaphorProfile`` output on ``CraftScenePlan`` in stable
    character data. ``permanent_domains`` come from life experience;
    ``forbidden_domains`` are domains the character has no experience of.
    ``current_domains`` is *computed* at plan time from current emotional
    state and is NOT stored on this model.

    Intended to live on ``Entity.data["metaphor"]`` for CHARACTER entities
    and be loaded via ``app.planning.metaphor.character_metaphor_profile_for``.
    """
    permanent_domains: list[str] = Field(default_factory=list)
    forbidden_domains: list[str] = Field(default_factory=list)
    metaphor_density: float = Field(default=0.3, ge=0.0, le=1.0)
    extends_to_narration: bool = True


class CharacterVoice(BaseModel):
    """Persistent voice specification for a character.

    This is the grounding source for free-indirect-style bleed: the narrator
    pulls from a character's vocabulary, cadence, and metaphor habits when
    voice permeability is high. Intended to live on ``Entity.data["voice"]``
    for CHARACTER entities and be loaded via
    ``app.planning.voice.character_voice_for(entity)``.

    ``bleed_vocabulary`` is derived from ``jargon_domains`` + ``signature_phrases``;
    ``excluded_vocabulary`` is derived from ``forbidden_words`` plus any
    narrator-register-mismatched terms. Those are NOT stored on this model —
    they are computed when the craft planner needs them.
    """
    vocabulary_level: str = "plain"           # e.g. "elevated", "plain", "coarse"
    jargon_domains: list[str] = Field(default_factory=list)
    forbidden_words: list[str] = Field(default_factory=list)
    signature_phrases: list[str] = Field(default_factory=list)
    sentence_length_bias: str = "varied"      # e.g. "short_clipped", "long_winding"
    directness: float = Field(default=0.5, ge=0.0, le=1.0)
    uses_metaphor: bool = True
    emotional_expression: str = "mixed"       # e.g. "guarded", "effusive", "ironic"
    truth_tendency: str = "candid"            # e.g. "evasive", "candid", "embellishes"
    code_switching: list[str] = Field(default_factory=list)  # contexts where register shifts
    voice_samples: list[str] = Field(default_factory=list)


class PassagePermeability(BaseModel):
    """Per-passage voice permeability instruction."""
    passage_description: str
    target: float = Field(ge=0.0, le=1.0)
    character_id: str
    bleed_words: list[str] = []
    reason: str


class DetailPrinciple(BaseModel):
    """Governs detail selection for a scene — why THIS detail, not a generic one."""
    perceiving_character_id: str
    perceptual_preoccupations: list[str]   # what they're currently biased to notice
    detail_mode: Literal[
        "character_revealing", "world_establishing", "thematic_resonant",
        "mood_setting", "foreshadowing", "ironic",
    ] = "character_revealing"
    triple_duty_targets: list[str] = []    # moments where one detail serves multiple functions


class MetaphorProfile(BaseModel):
    """Metaphor source domains a character draws from. Metaphor IS characterization."""
    character_id: str
    permanent_domains: list[str]           # from life experience, stable
    current_domains: list[str] = []        # activated by current state
    forbidden_domains: list[str] = []      # domains this character has no experience of
    metaphor_density: Literal["sparse", "occasional", "regular", "rich"] = "occasional"
    extends_to_narration: bool = True      # bleeds into narrator during high permeability


class IndirectionInstruction(BaseModel):
    """How to render unconscious motive through behavior and detail, not exposition."""
    character_id: str
    unconscious_motive: str                # what the system knows but won't name
    surface_manifestations: list[str]      # observable behaviors that express the motive
    detail_tells: list[str]                # details whose inclusion carries the subtext
    what_not_to_say: list[str]             # narration to AVOID
    reader_should_infer: str               # the check criterion


class CraftBrief(BaseModel):
    """The prose form of the craft plan, for the WRITE stage.

    A director's note, not a parameter dump. The structured CraftScenePlan
    exists so critics can validate; the brief exists so the writer receives
    a unified creative vision rather than a checklist.
    """
    scene_id: int
    brief: str                             # 100-300 words of prose brief

    @field_validator("scene_id", mode="before")
    @classmethod
    def _coerce_scene_id(cls, v):
        return _coerce_scene_id(v)
