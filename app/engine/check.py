from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


Severity = Literal["info", "warning", "error", "critical"]
Category = Literal["continuity", "world_rule", "plan_adherence", "prose_quality"]


class CheckIssue(BaseModel):
    severity: Severity
    category: Category
    message: str
    suggested_fix: str | None = None


class CheckOutput(BaseModel):
    issues: list[CheckIssue] = Field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def has_fixable(self) -> bool:
        return any(i.severity in ("warning", "error") for i in self.issues)

    @property
    def all_trivial(self) -> bool:
        return all(i.severity == "info" for i in self.issues)


CHECK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
                    "category": {"type": "string",
                                 "enum": ["continuity", "world_rule", "plan_adherence", "prose_quality"]},
                    "message": {"type": "string"},
                    "suggested_fix": {"type": ["string", "null"]},
                },
                "required": ["severity", "category", "message"],
            },
        },
    },
    "required": ["issues"],
}
