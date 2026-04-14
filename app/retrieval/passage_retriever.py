"""Literary-corpus passage retriever (Wave 1b, metadata-only).

Loads the manifest + merged Claude labels, builds an in-memory index of
labeled passages, and serves filter-based retrieval queries.

Semantic (embedding) rerank is Wave 2; constructing with
``enable_semantic=True`` raises :class:`NotImplementedError` until that
wave wires it.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.calibration.loader import load_manifest

from .interface import Query, Result


DEFAULT_LABEL_GLOB = "/tmp/labels_claude_part_*.json"
DEFAULT_LABEL_ALL = "/tmp/labels_claude_all.json"


@dataclass
class _IndexedPassage:
    work_id: str
    passage_id: str
    is_quest: bool
    pov: str
    expected_scores: dict[str, float]  # work-level (manifest)
    actual_scores: dict[str, float]  # passage-level (Claude labels)
    passage_text: str

    def score_for(self, dim: str) -> float | None:
        """Passage-level (actual) score if present, else work-level (expected)."""
        if dim in self.actual_scores:
            return self.actual_scores[dim]
        if dim in self.expected_scores:
            return self.expected_scores[dim]
        return None


class PassageRetriever:
    """Serve metadata-filtered literary passages.

    Parameters
    ----------
    manifest_path:
        Path to ``data/calibration/manifest.yaml``.
    labels_dir:
        Directory that holds ``labels_claude_part_*.json`` shards (typically
        ``/tmp``). The merge logic matches
        ``tools/finetune/build_dataset.py::merge_labels``.
    passages_dir:
        Directory with ``<work_id>/<passage_id>.txt`` passage bodies.
    enable_semantic:
        Reserved for Wave 2; raises :class:`NotImplementedError` in Wave 1.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        labels_dir: str | Path,
        passages_dir: str | Path,
        enable_semantic: bool = False,
    ) -> None:
        if enable_semantic:
            raise NotImplementedError(
                "Semantic retrieval is Wave 2; pass enable_semantic=False."
            )

        self._manifest_path = Path(manifest_path)
        self._labels_dir = Path(labels_dir)
        self._passages_dir = Path(passages_dir)
        self._enable_semantic = enable_semantic

        self._index: list[_IndexedPassage] = self._build_index()

    # -- Loading --------------------------------------------------------

    def _merge_labels(self) -> list[dict[str, Any]]:
        """Mirror ``tools/finetune/build_dataset.py::merge_labels``.

        Reads every ``labels_claude_part_*.json`` shard under ``labels_dir``.
        If none are present, falls back to ``labels_claude_all.json`` if it
        exists.
        """
        merged: list[dict[str, Any]] = []
        shard_pattern = str(self._labels_dir / "labels_claude_part_*.json")
        shards = sorted(glob.glob(shard_pattern))

        for f in shards:
            data = json.loads(Path(f).read_text())
            stem = Path(f).stem.replace("labels_claude_part_", "")
            passages = data if isinstance(data, list) else data.get("passages", [])
            for p in passages:
                if "scores" in p and "dimensions" not in p:
                    p["dimensions"] = p.pop("scores")
                p.setdefault("work_id", stem)
                p.setdefault("is_quest", False)
                merged.append(p)

        if merged:
            return merged

        # Fallback: merged labels file from a prior build.
        fallback = self._labels_dir / "labels_claude_all.json"
        if fallback.is_file():
            data = json.loads(fallback.read_text())
            passages = data if isinstance(data, list) else data.get("passages", [])
            for p in passages:
                if "scores" in p and "dimensions" not in p:
                    p["dimensions"] = p.pop("scores")
                p.setdefault("is_quest", False)
            return list(passages)

        return []

    def _load_passage_text(self, work_id: str, passage_id: str) -> str | None:
        path = self._passages_dir / work_id / f"{passage_id}.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        # Skip YAML frontmatter (same convention as build_dataset.load_passage).
        if text.startswith("---"):
            _, _, rest = text[3:].partition("---")
            text = rest.lstrip()
        return text

    def _build_index(self) -> list[_IndexedPassage]:
        manifest = load_manifest(self._manifest_path)
        works_by_id = {w.id: w for w in manifest.works}

        labels = self._merge_labels()

        # Dedup labels to at most one entry per (work_id, passage_id). Last
        # shard wins, matching the iteration order in merge_labels.
        labels_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for p in labels:
            key = (p["work_id"], p["passage_id"])
            labels_by_key[key] = p

        index: list[_IndexedPassage] = []
        for (work_id, passage_id), label in labels_by_key.items():
            work = works_by_id.get(work_id)
            if work is None:
                # Label references a work the manifest doesn't know about;
                # skip — we need pov + is_quest metadata to index it.
                continue
            text = self._load_passage_text(work_id, passage_id)
            if text is None:
                continue
            actual = {k: float(v) for k, v in (label.get("dimensions") or {}).items()}
            index.append(
                _IndexedPassage(
                    work_id=work_id,
                    passage_id=passage_id,
                    is_quest=bool(label.get("is_quest", work.is_quest)),
                    pov=work.pov,
                    expected_scores=dict(work.expected),
                    actual_scores=actual,
                    passage_text=text,
                )
            )

        index.sort(key=lambda p: (p.work_id, p.passage_id))
        return index

    # -- Retrieval ------------------------------------------------------

    @property
    def index_size(self) -> int:
        return len(self._index)

    def _passes_filters(self, entry: _IndexedPassage, query: Query) -> bool:
        f = query.filters
        pov = f.get("pov")
        if pov is not None and entry.pov != pov:
            return False
        is_quest = f.get("is_quest")
        if is_quest is not None and entry.is_quest != is_quest:
            return False
        exclude_works = f.get("exclude_works") or set()
        if entry.work_id in exclude_works:
            return False
        for dim, (lo, hi) in (f.get("score_ranges") or {}).items():
            score = entry.score_for(dim)
            if score is None:
                return False
            if score < lo or score > hi:
                return False
        return True

    def _score(self, entry: _IndexedPassage, query: Query) -> float:
        """Start at 1.0, drop 0.2 per filter whose *midpoint* the entry misses."""
        score = 1.0
        f = query.filters
        for dim, (lo, hi) in (f.get("score_ranges") or {}).items():
            value = entry.score_for(dim)
            if value is None:
                score -= 0.2
                continue
            mid = (lo + hi) / 2.0
            half = max((hi - lo) / 2.0, 1e-6)
            rel = abs(value - mid) / half
            if rel > 0.5:
                score -= 0.2
        return max(score, 0.0)

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        candidates: list[tuple[float, _IndexedPassage]] = []
        for entry in self._index:
            if not self._passes_filters(entry, query):
                continue
            candidates.append((self._score(entry, query), entry))

        # Sort by score desc, then stable by (work_id, passage_id) asc.
        candidates.sort(key=lambda t: (-t[0], t[1].work_id, t[1].passage_id))

        results: list[Result] = []
        for score, entry in candidates[:k]:
            results.append(
                Result(
                    source_id=f"{entry.work_id}/{entry.passage_id}",
                    text=entry.passage_text,
                    score=score,
                    metadata={
                        "work_id": entry.work_id,
                        "passage_id": entry.passage_id,
                        "pov": entry.pov,
                        "is_quest": entry.is_quest,
                        "expected_scores": dict(entry.expected_scores),
                        "actual_scores": dict(entry.actual_scores),
                    },
                )
            )
        return results
