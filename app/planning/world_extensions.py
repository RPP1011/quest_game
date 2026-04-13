from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ThemeStance = Literal["exploring", "affirming", "questioning", "subverting"]


class Theme(BaseModel):
    """A debatable proposition the quest interrogates.

    A theme is a *claim*, not a topic — "loyalty demands self-erasure", not
    "loyalty". The stance tracks how the quest currently engages with the
    proposition; it may evolve over the course of play.
    """

    id: str
    proposition: str
    stance: ThemeStance = "exploring"
    motif_ids: list[str] = Field(default_factory=list)
    thesis_character_ids: list[str] = Field(default_factory=list)
    key_scenes: list[str] = Field(default_factory=list)


class MotifDef(BaseModel):
    id: str
    name: str
    description: str
    recurrences: list[int] = []  # update numbers where it appeared
