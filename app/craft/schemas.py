from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


Scale = Literal["scene", "chapter", "campaign", "saga"]
ToolCategory = Literal[
    "foreshadowing", "pacing", "reversal", "dialogue",
    "character", "structural", "tension", "revelation",
]


class ArcPhase(BaseModel):
    name: str
    position: int
    tension_target: float = Field(ge=0.0, le=1.0)
    expected_beats: list[str] = Field(default_factory=list)
    description: str


class Structure(BaseModel):
    id: str
    name: str
    description: str
    scales: list[Scale]
    phases: list[ArcPhase]
    tension_curve: list[tuple[float, float]]  # (position_0_1, tension_0_1)

    @model_validator(mode="after")
    def _validate(self) -> "Structure":
        positions = [p.position for p in self.phases]
        if len(positions) != len(set(positions)):
            raise ValueError("structure phases must have unique positions")
        self.phases = sorted(self.phases, key=lambda p: p.position)
        if self.tension_curve:
            xs = [x for x, _ in self.tension_curve]
            if xs != sorted(xs):
                raise ValueError("tension_curve must be sorted by position")
        return self


class Tool(BaseModel):
    id: str
    name: str
    category: ToolCategory
    description: str
    preconditions: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    example_ids: list[str] = Field(default_factory=list)


class Example(BaseModel):
    id: str
    tool_ids: list[str] = Field(min_length=1)
    source: str
    scale: Scale
    snippet: str
    annotation: str


class StyleRegister(BaseModel):
    id: str
    name: str
    description: str
    sentence_variance: Literal["low", "medium", "high"]
    concrete_abstract_ratio: float = Field(ge=0.0, le=1.0)
    interiority_depth: Literal["surface", "medium", "deep"]
    pov_discipline: Literal["strict", "moderate", "loose"]
    diction_register: str
    voice_samples: list[str] = Field(min_length=1)


class Arc(BaseModel):
    id: str
    name: str
    scale: Scale
    structure_id: str
    current_phase_index: int = 0
    phase_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    theme: str | None = None
    parent_arc_id: str | None = None
    child_arc_ids: list[str] = Field(default_factory=list)
    plot_thread_ids: list[str] = Field(default_factory=list)
    pivot_update_numbers: list[int] = Field(default_factory=list)
    tension_observed: list[tuple[int, float]] = Field(default_factory=list)
    required_beats_remaining: list[str] = Field(default_factory=list)
