from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .schemas import Example, Scale, Structure, StyleRegister, Tool, ToolCategory


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
