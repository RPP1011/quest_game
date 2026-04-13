from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    FACTION = "faction"
    ITEM = "item"
    CONCEPT = "concept"


class EntityStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    DECEASED = "deceased"
    DESTROYED = "destroyed"


class HookStatus(str, Enum):
    PLANTED = "planted"
    REFERENCED = "referenced"
    PAID_OFF = "paid_off"
    ABANDONED = "abandoned"


class ThreadStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ArcPosition(str, Enum):
    RISING = "rising"
    CLIMAX = "climax"
    FALLING = "falling"
    DENOUEMENT = "denouement"


class Entity(BaseModel):
    id: str
    entity_type: EntityType
    name: str
    data: dict[str, Any] = Field(default_factory=dict)
    status: EntityStatus = EntityStatus.ACTIVE
    last_referenced_update: int | None = None
    created_at_update: int | None = None


class Relationship(BaseModel):
    source_id: str
    target_id: str
    rel_type: str
    data: dict[str, Any] = Field(default_factory=dict)
    established_at_update: int | None = None


class WorldRule(BaseModel):
    id: str
    category: str
    description: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    established_at_update: int | None = None


class TimelineEvent(BaseModel):
    update_number: int
    event_index: int
    description: str
    involved_entities: list[str] = Field(default_factory=list)
    causal_links: list[tuple[int, int]] = Field(default_factory=list)


class NarrativeRecord(BaseModel):
    update_number: int
    raw_text: str
    summary: str | None = None
    chapter_id: int | None = None
    state_diff: dict[str, Any] = Field(default_factory=dict)
    player_action: str | None = None
    pipeline_trace_id: str | None = None


class ForeshadowingHook(BaseModel):
    id: str
    description: str
    planted_at_update: int
    payoff_target: str
    status: HookStatus = HookStatus.PLANTED
    paid_off_at_update: int | None = None
    references: list[int] = Field(default_factory=list)


class PlotThread(BaseModel):
    id: str
    name: str
    description: str
    status: ThreadStatus = ThreadStatus.ACTIVE
    involved_entities: list[str] = Field(default_factory=list)
    arc_position: ArcPosition
    priority: int = Field(default=5, ge=1, le=10)
