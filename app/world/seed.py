from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from app.planning.world_extensions import Theme
from .delta import EntityCreate, RelChange, StateDelta
from .schema import (
    Entity,
    ForeshadowingHook,
    PlotThread,
    Relationship,
    WorldRule,
)


@dataclass
class SeedPayload:
    delta: StateDelta
    rules: list[WorldRule] = field(default_factory=list)
    foreshadowing: list[ForeshadowingHook] = field(default_factory=list)
    plot_threads: list[PlotThread] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)


def _coerce_theme(raw: Any, index: int) -> Theme:
    """Accept either a plain string (legacy seed format) or a structured
    Theme dict. Plain strings are auto-wrapped with a defaulted schema."""
    if isinstance(raw, str):
        return Theme(
            id=f"theme:{index}",
            proposition=raw,
            stance="exploring",
            motif_ids=[],
            thesis_character_ids=[],
            key_scenes=[],
        )
    if isinstance(raw, dict):
        data = dict(raw)
        # Backward-compat: older shape used {name, description}
        if "proposition" not in data:
            if "description" in data:
                data["proposition"] = data["description"]
            elif "name" in data:
                data["proposition"] = data["name"]
        data.setdefault("id", f"theme:{index}")
        data.setdefault("stance", "exploring")
        # Strip unknown legacy keys so model_validate doesn't choke.
        allowed = {
            "id", "proposition", "stance", "motif_ids",
            "thesis_character_ids", "key_scenes",
        }
        data = {k: v for k, v in data.items() if k in allowed}
        return Theme.model_validate(data)
    raise ValueError(f"theme entry must be string or object, got {type(raw).__name__}")


class SeedLoader:
    @staticmethod
    def load(path: str | Path) -> SeedPayload:
        p = Path(path)
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"seed file not valid JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError("seed root must be an object")

        try:
            entities = [Entity.model_validate(x) for x in raw.get("entities", [])]
            rels = [Relationship.model_validate(x) for x in raw.get("relationships", [])]
            rules = [WorldRule.model_validate(x) for x in raw.get("rules", [])]
            hooks = [ForeshadowingHook.model_validate(x) for x in raw.get("foreshadowing", [])]
            threads = [PlotThread.model_validate(x) for x in raw.get("plot_threads", [])]
            themes = [
                _coerce_theme(t, i) for i, t in enumerate(raw.get("themes", []))
            ]
        except ValidationError as e:
            raise ValueError(f"seed file schema error: {e}") from e

        delta = StateDelta(
            entity_creates=[EntityCreate(entity=e) for e in entities],
            relationship_changes=[RelChange(action="add", relationship=r) for r in rels],
        )
        return SeedPayload(
            delta=delta, rules=rules, foreshadowing=hooks, plot_threads=threads,
            themes=themes,
        )
