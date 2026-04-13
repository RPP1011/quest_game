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
from app.planning.world_extensions import MotifDef, Theme

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
    "MotifDef",
    # Critics module
    "critics",
]
