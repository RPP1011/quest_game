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


class Motif(BaseModel):
    """A recurring concrete image/element that carries evolving meaning.

    Persistent Motif Tracking (Gap G5). A motif has a fixed identity (id, name,
    description) and links to one or more Themes. Its ``semantic_range`` is the
    set of meanings it can carry across recurrences; each occurrence selects
    one value from that range. ``target_interval_min/max`` bound how often the
    motif should recur (in update numbers).
    """

    id: str
    name: str
    description: str
    theme_ids: list[str] = Field(default_factory=list)
    semantic_range: list[str] = Field(default_factory=list)
    target_interval_min: int = 2
    target_interval_max: int = 6


# Legacy alias retained for backward-compat refs.
MotifDef = Motif


class MotifOccurrence(BaseModel):
    """A single appearance of a motif in a specific update."""

    motif_id: str
    update_number: int
    context: str = ""
    semantic_value: str = ""
    intensity: float = 0.5
