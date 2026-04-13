from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
from .schema import Entity, Relationship, TimelineEvent


class EntityCreate(BaseModel):
    entity: Entity


class EntityUpdate(BaseModel):
    id: str
    patch: dict[str, Any]


class RelChange(BaseModel):
    action: Literal["add", "remove", "modify"]
    relationship: Relationship


class TimelineEventOp(BaseModel):
    event: TimelineEvent


class FSUpdate(BaseModel):
    id: str
    new_status: Literal["planted", "referenced", "paid_off", "abandoned"]
    paid_off_at_update: int | None = None
    add_reference: int | None = None  # append this update to `references`


class PTUpdate(BaseModel):
    id: str
    patch: dict[str, Any]


class StateDelta(BaseModel):
    entity_creates: list[EntityCreate] = Field(default_factory=list)
    entity_updates: list[EntityUpdate] = Field(default_factory=list)
    relationship_changes: list[RelChange] = Field(default_factory=list)
    timeline_events: list[TimelineEventOp] = Field(default_factory=list)
    foreshadowing_updates: list[FSUpdate] = Field(default_factory=list)
    plot_thread_updates: list[PTUpdate] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str
    subject: str | None = None  # entity id, rel key, etc.


class ValidationResult(BaseModel):
    issues: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)
