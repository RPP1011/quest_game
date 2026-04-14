"""Literary-corpus passage retriever.

Loads the manifest + merged Claude labels, builds an in-memory index of
labeled passages, and serves filter-based retrieval queries.

Two modes:

* ``enable_semantic=False`` (Wave 1b): metadata-only — filter by POV,
  ``is_quest``, ``exclude_works`` and score ranges, then rank by
  midpoint proximity.
* ``enable_semantic=True`` (Wave 2a): hybrid — apply the same metadata
  filter first, then, if ``Query.seed_text`` is provided, embed it with
  MiniLM and rerank surviving candidates by a 50/50 blend of the
  metadata score and cosine similarity. Without ``seed_text`` the
  hybrid mode degrades gracefully to metadata-only ranking.

The semantic embedder and the on-disk :class:`~app.retrieval.embeddings.EmbeddingCache`
are constructed lazily: they are not touched until the first call to
:meth:`PassageRetriever.retrieve` when ``enable_semantic=True``.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.calibration.loader import load_manifest

from .embeddings import Embedder, EmbeddingCache
from .interface import Query, Result


DEFAULT_LABEL_GLOB = "/tmp/labels_claude_part_*.json"
DEFAULT_LABEL_ALL = "/tmp/labels_claude_all.json"
DEFAULT_EMBEDDING_CACHE_PATH = Path("data/calibration/passage_embeddings.npy")


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
    """Serve metadata-filtered literary passages with optional semantic rerank.

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
        If ``True``, hybrid mode is active: the retriever computes
        MiniLM embeddings for every indexed passage (lazily, on first
        ``retrieve`` call) and blends cosine similarity with the
        metadata score when ``Query.seed_text`` is set. The embedder
        and cache are constructed on demand, not during ``__init__``.
    embedding_cache_path:
        ``.npy`` file persisting the embedding matrix. A sibling
        ``.json`` file (same stem) holds the parallel id array. Only
        consulted when ``enable_semantic=True``. Defaults to
        ``data/calibration/passage_embeddings.npy``.
    embedder:
        Optional injected :class:`Embedder` — primarily a test seam.
        When ``None`` a default :class:`Embedder` is built lazily on
        first retrieve.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        labels_dir: str | Path,
        passages_dir: str | Path,
        enable_semantic: bool = False,
        embedding_cache_path: str | Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._manifest_path = Path(manifest_path)
        self._labels_dir = Path(labels_dir)
        self._passages_dir = Path(passages_dir)
        self._enable_semantic = enable_semantic
        self._embedding_cache_path = Path(
            embedding_cache_path
            if embedding_cache_path is not None
            else DEFAULT_EMBEDDING_CACHE_PATH
        )

        # Lazily constructed on first semantic retrieve. None until then.
        self._embedder: Embedder | None = embedder
        self._passage_vectors: dict[str, np.ndarray] | None = None

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

    # -- Semantic lazy init --------------------------------------------

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _ensure_passage_vectors(self) -> dict[str, np.ndarray]:
        """Load or build the passage embedding cache on first use.

        Passages with empty text are excluded from the cache (and
        therefore from semantic rerank). They still participate in
        metadata-only flow because the index keeps them.
        """
        if self._passage_vectors is not None:
            return self._passage_vectors

        embedder = self._ensure_embedder()
        cache = EmbeddingCache(embedder)
        pairs: list[tuple[str, str]] = []
        for entry in self._index:
            if not entry.passage_text:
                continue
            key = f"{entry.work_id}/{entry.passage_id}"
            pairs.append((key, entry.passage_text))
        if not pairs:
            self._passage_vectors = {}
            return self._passage_vectors

        self._passage_vectors = cache.load_or_build(
            pairs, self._embedding_cache_path
        )
        return self._passage_vectors

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        # Metadata-only fast path preserves the Wave 1b ranking byte-for-byte.
        if not self._enable_semantic:
            return self._retrieve_metadata_only(query, k=k)

        # Hybrid mode. Metadata filter first, same as before.
        filtered: list[tuple[float, _IndexedPassage]] = []
        for entry in self._index:
            if not self._passes_filters(entry, query):
                continue
            filtered.append((self._score(entry, query), entry))

        # Without a seed_text, or if the metadata filter dropped
        # everything, the hybrid mode degrades to metadata-only ranking
        # and does not spin up the embedder.
        if not query.seed_text or not filtered:
            filtered.sort(key=lambda t: (-t[0], t[1].work_id, t[1].passage_id))
            return self._build_results(filtered[:k])

        vectors = self._ensure_passage_vectors()
        embedder = self._ensure_embedder()
        seed_vec = embedder.embed_one(query.seed_text)

        ranked: list[tuple[float, _IndexedPassage]] = []
        for meta_score, entry in filtered:
            key = f"{entry.work_id}/{entry.passage_id}"
            passage_vec = vectors.get(key)
            if passage_vec is None:
                # Passage text was empty at build time — skip from semantic
                # rerank but keep a metadata-only contribution so we still
                # have a candidate (half-weighted).
                final = 0.5 * meta_score
            else:
                cosine = float(Embedder.cosine(passage_vec, seed_vec))
                # MiniLM cosine lives in [-1, 1]; clamp negatives to 0 so the
                # composite stays in [0, 1].
                cosine = max(cosine, 0.0)
                final = 0.5 * meta_score + 0.5 * cosine
            ranked.append((final, entry))

        ranked.sort(key=lambda t: (-t[0], t[1].work_id, t[1].passage_id))
        return self._build_results(ranked[:k])

    # -- Helpers --------------------------------------------------------

    def _retrieve_metadata_only(self, query: Query, *, k: int) -> list[Result]:
        candidates: list[tuple[float, _IndexedPassage]] = []
        for entry in self._index:
            if not self._passes_filters(entry, query):
                continue
            candidates.append((self._score(entry, query), entry))

        # Sort by score desc, then stable by (work_id, passage_id) asc.
        candidates.sort(key=lambda t: (-t[0], t[1].work_id, t[1].passage_id))
        return self._build_results(candidates[:k])

    def _build_results(
        self, scored: list[tuple[float, _IndexedPassage]]
    ) -> list[Result]:
        results: list[Result] = []
        for score, entry in scored:
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
