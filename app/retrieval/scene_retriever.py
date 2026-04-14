"""Scene-shape retriever (Wave 3c).

Mirrors :class:`app.retrieval.passage_retriever.PassageRetriever` but reads
from the arc-scale scene corpus instead of the passage corpus. Serves
scene-shape exemplars (full 2000-4000 word scenes) filtered by dramatic
function and/or arc-scale dimension score ranges so the
``DramaticPlanner`` can ground each scene against literature that
executed a similar function well.

Sources
-------
- ``data/calibration/scenes_manifest.yaml`` — per-work manifest of
  scene slots (``sNN``), POV, is_quest, expected arc scores.
- ``/tmp/labels_claude_arc_*.json`` — per-scene Claude-labeled arc
  dimensions (``tension_execution``, ``thematic_development``,
  ``scene_coherence``, ``emotional_arc``, ``consequence_weight``, and
  quest-only extras). Files may either be a bare list or
  ``{"passages": [...]}`` at the top level. Both shapes are handled.
- ``data/calibration/scenes/<work_id>/<scene_id>.txt`` — raw scene
  bodies. Gitignored, sampled on demand.

The retriever is intentionally permissive:

* Missing scene text on disk → that scene is silently skipped (same
  convention as :class:`PassageRetriever`).
* Labels that reference an unknown work → dropped.
* Labels without a numeric ``dimensions`` map → still indexed but
  excluded from any ``score_ranges`` filter.

Semantic mode is available behind the same lazy construction pattern
the passage retriever uses — the embedder is only instantiated when
``enable_semantic=True`` and the first :meth:`retrieve` call provides a
``seed_text``.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .embeddings import Embedder, EmbeddingCache
from .interface import Query, Result


DEFAULT_ARC_LABELS_GLOB = "/tmp/labels_claude_arc_*.json"
DEFAULT_EMBEDDING_CACHE_PATH = Path("data/calibration/scene_embeddings.npy")


@dataclass
class _IndexedScene:
    work_id: str
    scene_id: str
    is_quest: bool
    pov: str
    expected_scores: dict[str, float]  # work-level (manifest)
    actual_scores: dict[str, float]  # scene-level (arc labels)
    dramatic_function: str | None
    scene_text: str

    def score_for(self, dim: str) -> float | None:
        """Scene-level (actual) score if present, else work-level (expected)."""
        if dim in self.actual_scores:
            return self.actual_scores[dim]
        if dim in self.expected_scores:
            return self.expected_scores[dim]
        return None


@dataclass
class _SceneWork:
    id: str
    pov: str
    is_quest: bool
    expected: dict[str, float] = field(default_factory=dict)


def _load_scene_manifest(path: Path) -> dict[str, _SceneWork]:
    """Read ``scenes_manifest.yaml`` into per-work metadata.

    The scene manifest shape differs slightly from the passage manifest
    (no ``year`` field, scene-scale dims only) so we parse it inline
    rather than reusing :func:`app.calibration.loader.load_manifest`.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    works: dict[str, _SceneWork] = {}
    for w in raw.get("works", []) or []:
        wid = w["id"]
        works[wid] = _SceneWork(
            id=wid,
            pov=str(w.get("pov", "")),
            is_quest=bool(w.get("is_quest", False)),
            expected={
                k: float(v) for k, v in (w.get("expected") or {}).items()
            },
        )
    return works


class SceneShapeRetriever:
    """Serve scene-shape exemplars filtered by dramatic function or arc dims.

    Parameters
    ----------
    scenes_manifest_path:
        Path to ``data/calibration/scenes_manifest.yaml``.
    arc_labels_glob:
        Glob matching the arc-scale label files (default
        ``/tmp/labels_claude_arc_*.json``). Each file is either a JSON
        list of scene-label dicts or an object with a ``passages`` list.
    scenes_dir:
        Directory with ``<work_id>/<scene_id>.txt`` scene bodies.
    enable_semantic:
        If ``True``, hybrid mode is active: the retriever embeds the
        whole scene corpus on first retrieve and blends cosine
        similarity with the metadata score when ``Query.seed_text`` is
        set. Defaults to ``False`` for cheap test/CI paths.
    embedding_cache_path:
        ``.npy`` file persisting the embedding matrix. Only consulted
        when ``enable_semantic=True``.
    embedder:
        Optional injected :class:`Embedder` — test seam; when ``None``
        a default :class:`Embedder` is built lazily.
    """

    def __init__(
        self,
        scenes_manifest_path: str | Path,
        arc_labels_glob: str | Path,
        scenes_dir: str | Path,
        enable_semantic: bool = False,
        embedding_cache_path: str | Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._manifest_path = Path(scenes_manifest_path)
        self._arc_labels_glob = str(arc_labels_glob)
        self._scenes_dir = Path(scenes_dir)
        self._enable_semantic = enable_semantic
        self._embedding_cache_path = Path(
            embedding_cache_path
            if embedding_cache_path is not None
            else DEFAULT_EMBEDDING_CACHE_PATH
        )

        # Lazily constructed on first semantic retrieve.
        self._embedder: Embedder | None = embedder
        self._scene_vectors: dict[str, np.ndarray] | None = None

        self._index: list[_IndexedScene] = self._build_index()

    # -- Loading --------------------------------------------------------

    def _merge_arc_labels(self) -> list[dict[str, Any]]:
        """Load every arc-label file matched by the configured glob.

        Each file's ``passages`` list (or bare list) contributes its
        entries to the merged stream. Later files override earlier ones
        on ``(work_id, passage_id)`` collisions — matches the passage
        retriever's "last shard wins" contract.
        """
        merged: list[dict[str, Any]] = []
        for f in sorted(glob.glob(self._arc_labels_glob)):
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries = data if isinstance(data, list) else data.get("passages", [])
            for p in entries or []:
                if "scores" in p and "dimensions" not in p:
                    p["dimensions"] = p.pop("scores")
                merged.append(p)
        return merged

    def _load_scene_text(self, work_id: str, scene_id: str) -> str | None:
        path = self._scenes_dir / work_id / f"{scene_id}.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        # Skip YAML frontmatter (same convention as passage loader).
        if text.startswith("---"):
            _, _, rest = text[3:].partition("---")
            text = rest.lstrip()
        return text

    def _build_index(self) -> list[_IndexedScene]:
        works = _load_scene_manifest(self._manifest_path)

        labels = self._merge_arc_labels()
        # Dedup: last label wins on (work_id, passage_id).
        labels_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for p in labels:
            wid = p.get("work_id")
            pid = p.get("passage_id")
            if not wid or not pid:
                continue
            labels_by_key[(wid, pid)] = p

        index: list[_IndexedScene] = []
        for (work_id, scene_id), label in labels_by_key.items():
            work = works.get(work_id)
            if work is None:
                continue
            text = self._load_scene_text(work_id, scene_id)
            if text is None:
                continue
            actual: dict[str, float] = {}
            for k, v in (label.get("dimensions") or {}).items():
                try:
                    actual[k] = float(v)
                except (TypeError, ValueError):
                    continue
            dramatic_function = label.get("dramatic_function")
            if dramatic_function is not None:
                dramatic_function = str(dramatic_function)
            index.append(
                _IndexedScene(
                    work_id=work_id,
                    scene_id=scene_id,
                    is_quest=bool(label.get("is_quest", work.is_quest)),
                    pov=work.pov,
                    expected_scores=dict(work.expected),
                    actual_scores=actual,
                    dramatic_function=dramatic_function,
                    scene_text=text,
                )
            )

        index.sort(key=lambda s: (s.work_id, s.scene_id))
        return index

    # -- Retrieval ------------------------------------------------------

    @property
    def index_size(self) -> int:
        return len(self._index)

    def _passes_filters(self, entry: _IndexedScene, query: Query) -> bool:
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
        dramatic_function = f.get("dramatic_function")
        if dramatic_function is not None:
            # If this scene has a dramatic_function label, require a
            # match. If it doesn't, drop it — the caller explicitly
            # asked for labeled scenes.
            if entry.dramatic_function is None:
                return False
            if entry.dramatic_function != dramatic_function:
                return False
        for dim, (lo, hi) in (f.get("score_ranges") or {}).items():
            score = entry.score_for(dim)
            if score is None:
                return False
            if score < lo or score > hi:
                return False
        return True

    def _score(self, entry: _IndexedScene, query: Query) -> float:
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

    # -- Semantic lazy init --------------------------------------------

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _ensure_scene_vectors(self) -> dict[str, np.ndarray]:
        if self._scene_vectors is not None:
            return self._scene_vectors
        embedder = self._ensure_embedder()
        cache = EmbeddingCache(embedder)
        pairs: list[tuple[str, str]] = []
        for entry in self._index:
            if not entry.scene_text:
                continue
            key = f"{entry.work_id}/{entry.scene_id}"
            pairs.append((key, entry.scene_text))
        if not pairs:
            self._scene_vectors = {}
            return self._scene_vectors
        self._scene_vectors = cache.load_or_build(
            pairs, self._embedding_cache_path
        )
        return self._scene_vectors

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        # Metadata-only fast path: filter → midpoint-proximity ranking.
        if not self._enable_semantic:
            return self._retrieve_metadata_only(query, k=k)

        filtered: list[tuple[float, _IndexedScene]] = []
        for entry in self._index:
            if not self._passes_filters(entry, query):
                continue
            filtered.append((self._score(entry, query), entry))

        if not query.seed_text or not filtered:
            filtered.sort(key=lambda t: (-t[0], t[1].work_id, t[1].scene_id))
            return self._build_results(filtered[:k])

        vectors = self._ensure_scene_vectors()
        embedder = self._ensure_embedder()
        seed_vec = embedder.embed_one(query.seed_text)

        ranked: list[tuple[float, _IndexedScene]] = []
        for meta_score, entry in filtered:
            key = f"{entry.work_id}/{entry.scene_id}"
            scene_vec = vectors.get(key)
            if scene_vec is None:
                final = 0.5 * meta_score
            else:
                cosine = float(Embedder.cosine(scene_vec, seed_vec))
                cosine = max(cosine, 0.0)
                final = 0.5 * meta_score + 0.5 * cosine
            ranked.append((final, entry))

        ranked.sort(key=lambda t: (-t[0], t[1].work_id, t[1].scene_id))
        return self._build_results(ranked[:k])

    # -- Helpers --------------------------------------------------------

    def _retrieve_metadata_only(self, query: Query, *, k: int) -> list[Result]:
        candidates: list[tuple[float, _IndexedScene]] = []
        for entry in self._index:
            if not self._passes_filters(entry, query):
                continue
            candidates.append((self._score(entry, query), entry))
        candidates.sort(key=lambda t: (-t[0], t[1].work_id, t[1].scene_id))
        return self._build_results(candidates[:k])

    def _build_results(
        self, scored: list[tuple[float, _IndexedScene]]
    ) -> list[Result]:
        results: list[Result] = []
        for score, entry in scored:
            metadata: dict[str, Any] = {
                "work_id": entry.work_id,
                "scene_id": entry.scene_id,
                "pov": entry.pov,
                "is_quest": entry.is_quest,
                "scale": "scene",
                "expected_scores": dict(entry.expected_scores),
                "actual_scores": dict(entry.actual_scores),
            }
            if entry.dramatic_function is not None:
                metadata["dramatic_function"] = entry.dramatic_function
            results.append(
                Result(
                    source_id=f"{entry.work_id}/{entry.scene_id}",
                    text=entry.scene_text,
                    score=score,
                    metadata=metadata,
                )
            )
        return results
