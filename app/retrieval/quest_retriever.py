"""Per-quest narrative retriever (Wave 3a).

Reads the ``narrative_embeddings`` table (schema shipped in Wave 1c) for
the current quest, optionally applies temporal filters, and returns the
top-``k`` records ranked by cosine similarity against an embedded
``Query.seed_text``.

Filter semantics (all optional except ``seed_text``):

* ``entity_mentions: set[str]`` — case-insensitive substring match
  against the record's ``text_preview``. Matching records get a
  ``1.5`` score multiplier; non-matching records keep their raw
  cosine score.
* ``max_updates_ago: int`` — drop any record whose
  ``update_number`` is older than ``max_updates_ago`` updates behind
  the newest indexed record. ``current_max_update`` is derived from
  the rows themselves (no explicit ``current_update`` parameter).
* ``last_n_records: int`` — before semantic ranking, truncate the
  candidate pool to the ``N`` most-recent rows (ordered newest-first
  by update/scene). Applied after ``max_updates_ago``.

``seed_text`` is required because callbacks are inherently semantic;
without it, :meth:`retrieve` returns ``[]``. The writer hook that
populates ``narrative_embeddings`` ships in Wave 3b — this retriever is
standalone infrastructure until then.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.world.state_manager import WorldStateManager

from .embeddings import Embedder
from .interface import Query, Result


_ENTITY_BOOST = 1.5


class QuestRetriever:
    """Serve per-quest narrative callbacks via cosine rerank over embeddings.

    Parameters
    ----------
    world:
        The :class:`WorldStateManager` backing the current quest. Used
        read-only via ``list_narrative_embeddings``.
    quest_id:
        The quest scope. The retriever never looks at rows from other
        quests.
    embedder:
        Optional injected :class:`Embedder`. Tests pass a deterministic
        embedder; production leaves this ``None`` and an
        :class:`Embedder` is lazily constructed on first ``retrieve``.
    """

    def __init__(
        self,
        world: WorldStateManager,
        quest_id: str,
        embedder: Embedder | None = None,
    ) -> None:
        self._world = world
        self._quest_id = quest_id
        self._embedder: Embedder | None = embedder

    # -- Embedder lifecycle --------------------------------------------

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    # -- Filtering -----------------------------------------------------

    @staticmethod
    def _apply_temporal_filters(
        rows: list[dict[str, Any]],
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply ``max_updates_ago`` + ``last_n_records`` to ``rows``.

        ``rows`` comes in newest-first (per ``list_narrative_embeddings``
        ordering). Both filters preserve that order.
        """
        if not rows:
            return rows

        max_updates_ago = filters.get("max_updates_ago")
        if max_updates_ago is not None:
            current_max = max(r["update_number"] for r in rows)
            cutoff = current_max - int(max_updates_ago)
            rows = [r for r in rows if r["update_number"] >= cutoff]

        last_n = filters.get("last_n_records")
        if last_n is not None:
            rows = rows[: int(last_n)]

        return rows

    @staticmethod
    def _entity_boost_multiplier(
        preview: str, entity_mentions: set[str] | list[str] | None,
    ) -> float:
        if not entity_mentions:
            return 1.0
        lowered = preview.lower()
        for mention in entity_mentions:
            if not mention:
                continue
            if mention.lower() in lowered:
                return _ENTITY_BOOST
        return 1.0

    # -- Retrieval -----------------------------------------------------

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        # Callbacks are always semantic — without a seed we have nothing
        # to rank against, so bail early.
        if not query.seed_text:
            return []

        rows = self._world.list_narrative_embeddings(self._quest_id)
        if not rows:
            return []

        filters: dict[str, Any] = dict(query.filters or {})
        rows = self._apply_temporal_filters(rows, filters)
        if not rows:
            return []

        # ``rows`` is newest-first. Record each row's recency rank (0 =
        # newest) before semantic ranking scrambles the order.
        for recency_rank, row in enumerate(rows):
            row["recency_rank"] = recency_rank

        embedder = self._ensure_embedder()
        seed_vec = embedder.embed_one(query.seed_text)

        entity_mentions = filters.get("entity_mentions")

        matrix = np.stack([
            np.asarray(r["embedding"], dtype=np.float32) for r in rows
        ])
        cosines = Embedder.cosine(matrix, seed_vec)
        # ``cosine`` returns a scalar when both operands are 1-D; with a
        # single candidate the stack is 2-D so cosines is already 1-D of
        # length 1. Normalize to a Python list for easy zipping.
        cosine_values = np.asarray(cosines, dtype=np.float32).reshape(-1).tolist()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row, raw_cosine in zip(rows, cosine_values):
            # Clamp negatives so the multiplicative boost always grows the
            # score rather than flipping its sign.
            base = max(float(raw_cosine), 0.0)
            final = base * self._entity_boost_multiplier(
                row["text_preview"], entity_mentions
            )
            scored.append((final, row))

        # Sort by score desc, ties broken by recency (newest first) for
        # stability.
        scored.sort(
            key=lambda t: (-t[0], t[1]["recency_rank"]),
        )
        return self._build_results(scored[:k])

    # -- Result construction -------------------------------------------

    def _build_results(
        self, scored: list[tuple[float, dict[str, Any]]],
    ) -> list[Result]:
        results: list[Result] = []
        for score, row in scored:
            update_number = int(row["update_number"])
            scene_index = int(row["scene_index"])
            results.append(
                Result(
                    source_id=(
                        f"{self._quest_id}/u{update_number}/s{scene_index}"
                    ),
                    text=row["text_preview"],
                    score=score,
                    metadata={
                        "update_number": update_number,
                        "scene_index": scene_index,
                        "recency_rank": int(row["recency_rank"]),
                    },
                )
            )
        return results
