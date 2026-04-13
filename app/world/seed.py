from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import ValidationError
from app.craft.schemas import Narrator
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
    narrator: Narrator | None = None


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
            narrator_raw = raw.get("narrator")
            narrator = Narrator.model_validate(narrator_raw) if narrator_raw else None
        except ValidationError as e:
            raise ValueError(f"seed file schema error: {e}") from e

        delta = StateDelta(
            entity_creates=[EntityCreate(entity=e) for e in entities],
            relationship_changes=[RelChange(action="add", relationship=r) for r in rels],
        )
        return SeedPayload(
            delta=delta, rules=rules, foreshadowing=hooks, plot_threads=threads,
            narrator=narrator,
        )
