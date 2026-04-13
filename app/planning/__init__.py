"""Planning module — hierarchical narrative pipeline.

Exports the P10 public API for planning: four-layer hierarchy
(ARC → DRAMATIC → EMOTIONAL → CRAFT) with schema models and planner classes.
"""

# Planners
from app.planning.arc_planner import ArcPlanner
from app.planning.craft_planner import CraftPlanner
from app.planning.dramatic_planner import DramaticPlanner
from app.planning.emotional_planner import EmotionalPlanner

# All schemas from the core planning layer
from app.planning.schemas import (
    ArcDirective,
    AdvanceType,
    ActionResolution,
    CharacterArcDirective,
    CharacterEmotionalState,
    CharacterMetaphorProfile,
    CharacterVoice,
    PerceptualProfile,
    CraftBrief,
    CraftPlan,
    CraftScenePlan,
    DetailPrinciple,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
    IndirectionInstruction,
    Intensity,
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
    TransitionType,
    VoiceNote,
    VoicePermeability,
)

# World extensions used by planning
from app.planning.world_extensions import Motif, MotifDef, MotifOccurrence, Theme

# Voice grounding helpers (Gap G3)
from app.planning.voice import (
    blended_voice_samples_for,
    character_voice_for,
    default_permeability,
    derive_bleed_vocabulary,
    derive_excluded_vocabulary,
)

# Perception grounding helpers (Gap G9)
from app.planning.perception import (
    current_preoccupations,
    default_detail_principle,
    perceptual_profile_for,
)

# Metaphor grounding helpers (Gap G10)
from app.planning.metaphor import (
    character_metaphor_profile_for,
    compute_current_domains,
    default_metaphor_profile,
)

# Critics module — available as planning.critics
from app.planning import critics

__all__ = [
    # Planners
    "ArcPlanner",
    "DramaticPlanner",
    "EmotionalPlanner",
    "CraftPlanner",
    # Schemas
    "ArcDirective",
    "AdvanceType",
    "ActionResolution",
    "CharacterArcDirective",
    "CharacterEmotionalState",
    "CharacterMetaphorProfile",
    "CharacterVoice",
    "PerceptualProfile",
    "CraftBrief",
    "CraftPlan",
    "CraftScenePlan",
    "DetailPrinciple",
    "DramaticPlan",
    "DramaticScene",
    "EmotionalPlan",
    "EmotionalScenePlan",
    "IndirectionInstruction",
    "Intensity",
    "MetaphorProfile",
    "MotifInstruction",
    "NegativeSpaceInstruction",
    "ParallelInstruction",
    "PassageOverride",
    "PassagePermeability",
    "PlotObjective",
    "SceneRegister",
    "TemporalStructure",
    "ThemePriority",
    "ThreadAdvance",
    "ToolSelection",
    "TransitionType",
    "VoiceNote",
    "VoicePermeability",
    # World extensions
    "Theme",
    "Motif",
    "MotifDef",
    "MotifOccurrence",
    "blended_voice_samples_for",
    "character_voice_for",
    "default_permeability",
    "derive_bleed_vocabulary",
    "derive_excluded_vocabulary",
    "current_preoccupations",
    "default_detail_principle",
    "perceptual_profile_for",
    "character_metaphor_profile_for",
    "compute_current_domains",
    "default_metaphor_profile",
    # Critics module
    "critics",
]
