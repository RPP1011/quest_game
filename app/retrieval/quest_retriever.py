"""This-quest retriever (Wave 3a).

Reads the ``narrative_embeddings`` SQLite table (written post-commit by the
extract stage, see :meth:`Pipeline._persist_narrative_embedding`) and serves
in-quest callback retrieval for the writer.

The retriever supports the filter keys documented in the retrieval design:

* ``Query.seed_text`` — required; callbacks are always semantic.
* ``filters["entity_mentions"]: set[str]`` — boost records whose
  ``text_preview`` mentions any of these entities (case-insensitive
  substring match).
* ``filters["max_updates_ago"]: int`` — skip ancient records (keep only
  those within the given update window from the current update).
* ``filters["last_n_records"]: int`` — cap the candidate pool to the
  most-recent N records before semantic reranking.

Wave 3b notes
-------------
This retriever ships as a lightweight stub compatible with the spec
interface. Wave 3a (if it lands separately) can add richer scoring
(e.g. recency decay, entity-weight tuning); nothing in the Wave 3b
writer-integration surface depends on those refinements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.world.state_manager import WorldStateManager

from .embeddings import Embedder
from .interface import Query, Result


class QuestRetriever:
    """Retrieve previously-committed narrative passages for a single quest.

    Parameters
    ----------
    world:
        The :class:`WorldStateManager` holding the per-quest SQLite
        state. ``QuestRetriever`` reads ``narrative_embeddings`` via
        :meth:`WorldStateManager.list_narrative_embeddings`.
    quest_id:
        The quest whose records to search. Required at construction time
        because the ``narrative_embeddings`` table is partitioned by
        ``quest_id`` and callers always know which quest they are in.
    embedder:
        Optional :class:`Embedder` used to embed ``Query.seed_text``. If
        omitted, a default one is built on first use (lazy — no model
        load at construction).
    current_update:
        Optional baseline for ``max_updates_ago`` filtering. If ``None``
        (default), the filter is interpreted relative to the newest
        persisted record.
    """

    def __init__(
        self,
        world: WorldStateManager,
        quest_id: str,
        *,
        embedder: Embedder | None = None,
        current_update: int | None = None,
    ) -> None:
        self._world = world
        self._quest_id = quest_id
        self._embedder: Embedder | None = embedder
        self._current_update = current_update

    # -- Helpers --------------------------------------------------------

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def _passes_update_filter(
        self,
        row: dict[str, Any],
        filters: dict[str, Any],
        *,
        baseline_update: int | None,
    ) -> bool:
        max_updates_ago = filters.get("max_updates_ago")
        if max_updates_ago is None or baseline_update is None:
            return True
        row_update = int(row["update_number"])
        return (baseline_update - row_update) <= int(max_updates_ago)

    def _entity_boost(
        self, row: dict[str, Any], entity_mentions: set[str]
    ) -> float:
        """Cheap boost: +0.1 per distinct entity whose name appears in the preview.

        Capped at 0.5 so the semantic signal remains dominant.
        """
        if not entity_mentions:
            return 0.0
        preview = (row.get("text_preview") or "").lower()
        hits = 0
        for name in entity_mentions:
            if not name:
                continue
            if name.lower() in preview:
                hits += 1
        return min(0.1 * hits, 0.5)

    # -- Retrieval ------------------------------------------------------

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        if not query.seed_text:
            # Per spec: seed_text is required for QuestRetriever.
            return []

        filters = query.filters or {}
        last_n = filters.get("last_n_records")
        try:
            limit = int(last_n) if last_n is not None else None
        except (TypeError, ValueError):
            limit = None

        rows = self._world.list_narrative_embeddings(self._quest_id, limit=limit)
        if not rows:
            return []

        # Resolve baseline for max_updates_ago (fall back to newest row).
        baseline_update = self._current_update
        if baseline_update is None:
            baseline_update = max(int(r["update_number"]) for r in rows)

        rows = [
            r for r in rows
            if self._passes_update_filter(
                r, filters, baseline_update=baseline_update
            )
        ]
        if not rows:
            return []

        embedder = self._ensure_embedder()
        seed_vec = embedder.embed_one(query.seed_text)

        entity_mentions = set(filters.get("entity_mentions") or set())

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            vec = np.asarray(row["embedding"], dtype=np.float32)
            if vec.size == 0:
                continue
            cosine = float(Embedder.cosine(vec, seed_vec))
            cosine = max(cosine, 0.0)
            boost = self._entity_boost(row, entity_mentions)
            final = min(cosine + boost, 1.0)
            scored.append((final, row))

        # Rank by score desc; break ties by most recent update first.
        scored.sort(
            key=lambda t: (
                -t[0],
                -int(t[1]["update_number"]),
                -int(t[1]["scene_index"]),
            )
        )
        top = scored[:k]

        results: list[Result] = []
        for score, row in top:
            results.append(
                Result(
                    source_id=(
                        f"{self._quest_id}/"
                        f"{int(row['update_number'])}/"
                        f"{int(row['scene_index'])}"
                    ),
                    text=row.get("text_preview", ""),
                    score=score,
                    metadata={
                        "quest_id": self._quest_id,
                        "update_number": int(row["update_number"]),
                        "scene_index": int(row["scene_index"]),
                    },
                )
            )
        return results
