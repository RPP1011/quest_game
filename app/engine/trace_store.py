from __future__ import annotations
import os
from pathlib import Path
from .trace import PipelineTrace


class TraceStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, trace: PipelineTrace) -> Path:
        path = self._root / f"{trace.trace_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(trace.model_dump_json(indent=2))
        os.replace(tmp, path)
        return path

    def load(self, trace_id: str) -> PipelineTrace:
        path = self._root / f"{trace_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return PipelineTrace.model_validate_json(path.read_text())

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.json"))
