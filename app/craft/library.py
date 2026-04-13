from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .schemas import Arc, Example, Scale, Structure, StyleRegister, Tool, ToolCategory

_HOT_CATEGORIES = {"reversal", "tension", "pacing"}
_COLD_CATEGORIES = {"pacing", "character"}


def _score_tool(
    tool: Tool,
    phase_expected: set[str],
    required: set[str],
    recent: set[str],
    gap: float,
    patience_boost: bool = False,
) -> int:
    score = 0
    if tool.id in phase_expected:
        score += 3
    if tool.id in required:
        score += 5
    if gap > 0.15 and tool.category in _HOT_CATEGORIES:
        score += 2
    if gap < -0.15 and (tool.category in _COLD_CATEGORIES or tool.id == "scene_sequel"):
        score += 1
    if tool.id in recent:
        score -= 2
    # Gap G6: if the reader has been waiting too long without a major event,
    # nudge tools that tend to produce movement (reversal / tension).
    if patience_boost and tool.category in _HOT_CATEGORIES:
        score += 1
    return score


class CraftLibrary:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._structures: dict[str, Structure] = {}
        self._tools: dict[str, Tool] = {}
        self._examples: dict[str, Example] = {}
        self._styles: dict[str, StyleRegister] = {}
        self._load()

    def _load(self) -> None:
        self._structures = self._load_dir("structures", Structure)
        self._tools = self._load_dir("tools", Tool)
        self._styles = self._load_dir("styles", StyleRegister)
        self._examples = self._load_examples()

    def _load_dir(self, subdir: str, model) -> dict:
        out: dict[str, Any] = {}
        d = self._root / subdir
        if not d.is_dir():
            return out
        for path in sorted(d.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            if raw is None:
                continue
            obj = model.model_validate(raw)
            if obj.id in out:
                raise ValueError(
                    f"duplicate id {obj.id!r} in {subdir}: {path} and earlier file"
                )
            out[obj.id] = obj
        return out

    def _load_examples(self) -> dict[str, Example]:
        out: dict[str, Example] = {}
        d = self._root / "examples"
        if not d.is_dir():
            return out
        for path in sorted(d.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            if not raw:
                continue
            items = raw.get("examples") if isinstance(raw, dict) else raw
            for item in items or []:
                obj = Example.model_validate(item)
                if obj.id in out:
                    raise ValueError(f"duplicate example id {obj.id!r}")
                out[obj.id] = obj
        return out

    # ---- getters ----

    def structure(self, id: str) -> Structure:
        if id not in self._structures:
            raise KeyError(id)
        return self._structures[id]

    def tool(self, id: str) -> Tool:
        if id not in self._tools:
            raise KeyError(id)
        return self._tools[id]

    def example(self, id: str) -> Example:
        if id not in self._examples:
            raise KeyError(id)
        return self._examples[id]

    def style(self, id: str) -> StyleRegister:
        if id not in self._styles:
            raise KeyError(id)
        return self._styles[id]

    # ---- queries ----

    def structures(self, scale: Scale | None = None) -> list[Structure]:
        values = list(self._structures.values())
        if scale is None:
            return values
        return [s for s in values if scale in s.scales]

    def tools(self, category: ToolCategory | None = None) -> list[Tool]:
        values = list(self._tools.values())
        if category is None:
            return values
        return [t for t in values if t.category == category]

    def examples_for_tool(self, tool_id: str) -> list[Example]:
        return [e for e in self._examples.values() if tool_id in e.tool_ids]

    def all_structures(self) -> list[Structure]:
        return list(self._structures.values())

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def all_examples(self) -> list[Example]:
        return list(self._examples.values())

    def all_styles(self) -> list[StyleRegister]:
        return list(self._styles.values())

    def recommend_tools(
        self,
        arc: Arc,
        structure: Structure,
        recent_tool_ids: list[str] | None = None,
        limit: int = 5,
        updates_since_major_event: int | None = None,
        patience_threshold: int = 3,
    ) -> list[Tool]:
        from .arc import tension_gap as _tension_gap

        phase = structure.phases[min(arc.current_phase_index, len(structure.phases) - 1)]
        expected = set(phase.expected_beats)
        required = set(arc.required_beats_remaining)
        recent = set(recent_tool_ids or [])
        gap = _tension_gap(arc, structure)

        # Gap G6: patience boost when the reader has been waiting too long.
        patience_boost = (
            updates_since_major_event is not None
            and updates_since_major_event > patience_threshold
        )

        scored: list[tuple[int, str, Tool]] = []
        for tool in self.all_tools():
            score = _score_tool(
                tool, expected, required, recent, gap,
                patience_boost=patience_boost,
            )
            if score > 0:
                scored.append((score, tool.id, tool))

        required_order = {tid: i for i, tid in enumerate(arc.required_beats_remaining)}

        def sort_key(item: tuple[int, str, Tool]) -> tuple[int, int, str]:
            s, tid, _ = item
            req_rank = required_order.get(tid, len(required_order) + 1)
            return (-s, req_rank, tid)

        scored.sort(key=sort_key)
        return [t for _, _, t in scored[:limit]]
