from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class EntityScope(str, Enum):
    ALL = "all"           # every non-destroyed entity
    ACTIVE = "active"     # only status=active
    RELEVANT = "relevant" # active + referenced within the recent window


class NarrativeMode(str, Enum):
    FULL = "full"
    SUMMARY = "summary"
    NONE = "none"


class ContextSpec(BaseModel):
    entity_scope: EntityScope = EntityScope.RELEVANT
    include_relationships: bool = True
    include_rules: bool = True
    timeline_window: int = 10  # last N timeline events

    narrative_mode: NarrativeMode = NarrativeMode.SUMMARY
    narrative_window: int = 3  # last N narrative records (full mode)

    prior_stages: list[str] = Field(default_factory=list)

    include_style: bool = False
    include_character_voices: bool = False
    include_anti_patterns: bool = False
    include_foreshadowing: bool = True
    include_plot_threads: bool = True


PLAN_SPEC = ContextSpec(
    entity_scope=EntityScope.RELEVANT,
    narrative_mode=NarrativeMode.SUMMARY,
    include_style=False,
    include_rules=True,
    include_foreshadowing=True,
    include_plot_threads=True,
)


WRITE_SPEC = ContextSpec(
    entity_scope=EntityScope.ACTIVE,
    narrative_mode=NarrativeMode.FULL,
    narrative_window=3,
    include_style=True,
    include_character_voices=True,
    include_anti_patterns=True,
    include_rules=False,
    include_foreshadowing=False,
    include_plot_threads=False,
    prior_stages=["plan"],
)


CHECK_SPEC = ContextSpec(
    entity_scope=EntityScope.ACTIVE,
    narrative_mode=NarrativeMode.FULL,
    narrative_window=2,
    include_style=False,
    include_rules=True,
    include_plot_threads=True,
    include_foreshadowing=True,
    prior_stages=["plan", "write"],
)


REVISE_SPEC = ContextSpec(
    entity_scope=EntityScope.ACTIVE,
    narrative_mode=NarrativeMode.FULL,
    narrative_window=2,
    include_style=True,
    include_character_voices=True,
    include_anti_patterns=True,
    include_rules=False,
    prior_stages=["plan", "write", "check"],
)
