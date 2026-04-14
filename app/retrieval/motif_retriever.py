"""Motif recurrence retriever (Wave 4a).

Structural retriever (no embeddings) that surfaces motifs which are due or
overdue for re-appearance, based on the gap between the current update
and each motif's last recorded occurrence in the ``motif_occurrences``
table.

Due-ness semantics
------------------
For each motif in the quest we look at its most recent occurrence:

* **Overdue** — ``(current_update - last_update) > target_interval_max``;
  if the motif has never been recorded, we treat it as overdue once
  ``current_update >= target_interval_min`` (it's past the earliest
  time it should have appeared).
* **Due** — ``(current_update - last_update) > target_interval_min`` but
  not yet overdue.
* **Fresh** — anything else (recently seen or not yet eligible).

"Fresh" motifs are dropped from results entirely; the craft planner
only cares about motifs the prose should consider weaving in. Among
the survivors, overdue outranks due, and ties are broken by the oldest
``last_update`` (motifs silent longest get preference).

Query inputs
------------
``Query.filters["current_update"]`` (``int``) is the pipeline's notion of
"now". If absent, we default to ``0``, which makes every motif look early
(nothing due, never-seen motifs overdue only when their
``target_interval_min`` is ``0``). Tests rely on this default behavior.

Structured query — ``query.seed_text`` is ignored, no embedder is needed.
"""

from __future__ import annotations

from typing import Any

from app.planning.world_extensions import Motif, MotifOccurrence
from app.world.state_manager import WorldStateManager

from .interface import Query, Result


_OVERDUE_SCORE = 1.0
_DUE_SCORE = 0.6


class MotifRetriever:
    """Return motifs whose recurrence window has lapsed.

    Parameters
    ----------
    world:
        The :class:`WorldStateManager` backing the quest. Used read-only
        via ``list_motifs`` and ``list_motif_occurrences`` — no embedder
        needed because this retriever ranks on structured metadata.
    quest_id:
        The quest scope. Motifs from other quests never surface.
    """

    def __init__(self, world: WorldStateManager, quest_id: str) -> None:
        self._world = world
        self._quest_id = quest_id

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        current_update = int((query.filters or {}).get("current_update", 0))

        motifs = self._world.list_motifs(self._quest_id)
        if not motifs:
            return []

        scored: list[
            tuple[float, int, Motif, MotifOccurrence | None, list[MotifOccurrence], str]
        ] = []
        for motif in motifs:
            occurrences = self._world.list_motif_occurrences(
                self._quest_id, motif.id
            )
            # ``list_motif_occurrences`` returns ASC by update_number then id.
            # The newest entry is therefore the tail.
            last = occurrences[-1] if occurrences else None
            status = self._status(motif, last, current_update)
            if status == "fresh":
                continue
            score = _OVERDUE_SCORE if status == "overdue" else _DUE_SCORE
            # Tie-break: oldest last_update ranks first. Treat "never
            # occurred" as the oldest possible (negative infinity surrogate
            # = -1) so a never-seen overdue motif beats a recently-seen
            # overdue motif.
            last_update = last.update_number if last is not None else -1
            scored.append((score, last_update, motif, last, occurrences, status))

        if not scored:
            return []

        # Sort by score desc, then by oldest last_update asc, then motif id
        # asc for stability.
        scored.sort(key=lambda t: (-t[0], t[1], t[2].id))

        return self._build_results(scored[:k], current_update)

    # -- Helpers --------------------------------------------------------

    @staticmethod
    def _status(
        motif: Motif,
        last: MotifOccurrence | None,
        current_update: int,
    ) -> str:
        if last is None:
            if current_update >= motif.target_interval_min:
                return "overdue"
            return "fresh"
        gap = current_update - last.update_number
        if gap > motif.target_interval_max:
            return "overdue"
        if gap > motif.target_interval_min:
            return "due"
        return "fresh"

    def _build_results(
        self,
        scored: list[
            tuple[float, int, Motif, MotifOccurrence | None, list[MotifOccurrence], str]
        ],
        current_update: int,
    ) -> list[Result]:
        results: list[Result] = []
        for score, _last_update, motif, last, occurrences, status in scored:
            intervals_since_last: int | None
            last_update_number: int | None
            last_semantic_value: str | None
            if last is None:
                intervals_since_last = None
                last_update_number = None
                last_semantic_value = None
            else:
                last_update_number = int(last.update_number)
                last_semantic_value = last.semantic_value
                intervals_since_last = int(current_update - last.update_number)

            recent_contexts = self._recent_contexts(occurrences)

            metadata: dict[str, Any] = {
                "motif_id": motif.id,
                "name": motif.name,
                "last_update_number": last_update_number,
                "last_semantic_value": last_semantic_value,
                "intervals_since_last": intervals_since_last,
                "target_interval_min": motif.target_interval_min,
                "target_interval_max": motif.target_interval_max,
                "status": status,
                "recent_contexts": recent_contexts,
            }
            results.append(
                Result(
                    source_id=f"motif/{self._quest_id}/{motif.id}",
                    text=motif.description,
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    @staticmethod
    def _recent_contexts(
        occurrences: list[MotifOccurrence],
    ) -> list[dict[str, Any]]:
        """Return up to the last 2 occurrences as plain dicts (newest last).

        The planner uses these to see how the motif has been most recently
        worn — its semantic value, the surrounding context, and how hard it
        landed. Empty list when the motif has never appeared.
        """
        if not occurrences:
            return []
        tail = occurrences[-2:]
        return [
            {
                "update_number": int(occ.update_number),
                "context": occ.context,
                "semantic_value": occ.semantic_value,
                "intensity": float(occ.intensity),
            }
            for occ in tail
        ]
