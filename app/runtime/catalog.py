from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


_QUANT_RE = re.compile(r"-(Q\d+[_A-Z0-9]*|F16|F32|BF16)\.gguf$", re.IGNORECASE)
_REPO_DIR_RE = re.compile(r"^models--(?P<org>[^-]+(?:-[^-]+)*?)--(?P<name>.+)$")


@dataclass(frozen=True)
class ModelInfo:
    repo_id: str
    filename: str
    path: Path
    size_bytes: int
    quant: str | None


class ModelCatalog:
    def __init__(self, cache_root: Path) -> None:
        self._cache_root = Path(cache_root)
        self._cache: list[ModelInfo] | None = None

    def scan(self, refresh: bool = False) -> list[ModelInfo]:
        if self._cache is not None and not refresh:
            return self._cache
        if not self._cache_root.is_dir():
            self._cache = []
            return self._cache
        results: list[ModelInfo] = []
        for repo_dir in self._cache_root.iterdir():
            if not repo_dir.is_dir():
                continue
            m = _REPO_DIR_RE.match(repo_dir.name)
            if not m:
                continue
            repo_id = f"{m['org']}/{m['name']}"
            snapshots = repo_dir / "snapshots"
            if not snapshots.is_dir():
                continue
            for gguf in snapshots.rglob("*.gguf"):
                results.append(
                    ModelInfo(
                        repo_id=repo_id,
                        filename=gguf.name,
                        path=gguf.resolve(),
                        size_bytes=gguf.stat().st_size,
                        quant=_extract_quant(gguf.name),
                    )
                )
        self._cache = sorted(results, key=lambda m: (m.repo_id, m.filename))
        return self._cache


def _extract_quant(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None
