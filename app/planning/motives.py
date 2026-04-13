"""G13 — Indirection Instruction Grounding.

The system stores each character's unconscious motives persistently (as
entity data). The craft planner reads them and derives
``IndirectionInstruction`` entries rather than inventing new motives per
scene.

This module mirrors the pattern used elsewhere (e.g. character voice
grounding in G3): a pydantic model for the stored shape, a helper that
reads entity ``data`` and returns active entries, and a tiny backfill
utility used by the craft planner when the LLM emits empty or generic
indirection instructions.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.world.schema import Entity


class UnconsciousMotive(BaseModel):
    """A persistent unconscious motive attached to a character.

    ``active_since_update`` is the update number at which the motive became
    active. ``resolved_at_update`` is None while the motive is active and is
    set to the update number once the character has worked through it.
    """

    id: str
    motive: str
    surface_manifestations: list[str] = Field(default_factory=list)
    detail_tells: list[str] = Field(default_factory=list)
    what_not_to_say: list[str] = Field(default_factory=list)
    active_since_update: int = 0
    resolved_at_update: int | None = None

    @property
    def is_active(self) -> bool:
        return self.resolved_at_update is None


def unconscious_motives_for(entity: Entity) -> list[UnconsciousMotive]:
    """Return the *active* unconscious motives stored on ``entity.data``.

    Reads from ``entity.data["unconscious_motives"]`` (a list of dicts) and
    filters out any entries that have been resolved. Unknown / malformed
    entries are skipped silently so a stray datum never breaks planning.
    """
    raw = entity.data.get("unconscious_motives") if isinstance(entity.data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[UnconsciousMotive] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            m = UnconsciousMotive.model_validate(item)
        except Exception:
            continue
        if m.is_active:
            out.append(m)
    return out


def pick_primary_motive(motives: list[UnconsciousMotive]) -> UnconsciousMotive | None:
    """Pick the most-active motive — the one with the lowest ``active_since_update``
    (i.e. the one that has been active the longest)."""
    if not motives:
        return None
    return min(motives, key=lambda m: m.active_since_update)


def apply_motive_resolutions(
    entity_data: dict[str, Any],
    resolutions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a new copy of ``entity_data`` with the given motive resolutions applied.

    ``resolutions`` items must have ``motive_id`` and ``resolved_at_update``
    keys. Unknown motive ids are ignored. The existing list ordering is
    preserved; only the matching entries are mutated.
    """
    motives = entity_data.get("unconscious_motives")
    if not isinstance(motives, list):
        return entity_data
    by_id = {r.get("motive_id"): r.get("resolved_at_update") for r in resolutions}
    new_motives: list[dict[str, Any]] = []
    changed = False
    for item in motives:
        if isinstance(item, dict) and item.get("id") in by_id and item.get("resolved_at_update") is None:
            new_item = {**item, "resolved_at_update": by_id[item["id"]]}
            new_motives.append(new_item)
            changed = True
        else:
            new_motives.append(item)
    if not changed:
        return entity_data
    return {**entity_data, "unconscious_motives": new_motives}
