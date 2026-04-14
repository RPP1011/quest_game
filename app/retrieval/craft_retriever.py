"""Craft-tool exemplar retriever (Wave 2c).

Thin adapter around :class:`app.craft.library.CraftLibrary`. Converts the
craft library's per-tool ``Example`` list into the unified
``Query``/``Result`` retrieval interface so the pipeline's writer/craft
planner can consume craft examples the same way it consumes
literary-corpus passages and quest callbacks.

Schema note
-----------
The existing :class:`app.craft.schemas.Example` model only carries these
fields:

    id, tool_ids, source, scale, snippet, annotation

Of the filter dimensions listed in the retrieval design (``pov``,
``scale``, ``register``), **only ``scale`` is natively present** on
``Example``. ``pov`` and ``register`` are accepted on the query for
forward-compatibility but are ignored today (no-ops); they become active
if/when the ``Example`` schema grows those fields. We filter on the
fields that exist and ignore the rest, per spec instructions.

Retrieval is unranked — every matching example scores ``1.0``. The craft
library already curates examples per tool, so there is no corpus-wide
signal to rank on; callers take the first ``k`` after filtering.
"""

from __future__ import annotations

from typing import Any

from app.craft.library import CraftLibrary
from app.craft.schemas import Example

from .interface import Query, Result


class CraftRetriever:
    """Retrieve craft-tool examples filtered by tool id (+ optional scale).

    Parameters
    ----------
    craft_library:
        The already-loaded :class:`CraftLibrary`. ``CraftRetriever`` never
        mutates it.
    """

    def __init__(self, craft_library: CraftLibrary) -> None:
        self._library = craft_library

    # -- Retrieval ------------------------------------------------------

    def _passes_filters(self, example: Example, filters: dict[str, Any]) -> bool:
        scale = filters.get("scale")
        if scale is not None and example.scale != scale:
            return False
        # ``pov`` and ``register`` are accepted but no-op — the current
        # ``Example`` schema has no matching fields. Documented above.
        return True

    def _build_result(self, example: Example, tool_id: str) -> Result:
        return Result(
            source_id=f"craft/{tool_id}/{example.id}",
            text=example.snippet,
            score=1.0,
            metadata={
                "tool_id": tool_id,
                "example_id": example.id,
                "tool_ids": list(example.tool_ids),
                "source": example.source,
                "scale": example.scale,
                "annotation": example.annotation,
            },
        )

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        tool_id = query.filters.get("tool_id")
        if not tool_id or not isinstance(tool_id, str):
            return []

        # ``examples_for_tool`` is a pure filter over the in-memory dict;
        # unknown tool ids just yield ``[]`` which we pass through.
        try:
            examples = self._library.examples_for_tool(tool_id)
        except Exception:
            # Defensive — current implementation doesn't raise, but an
            # unknown tool id should always look like "no results".
            return []
        if not examples:
            return []

        filters = query.filters
        hits: list[Result] = []
        for example in examples:
            if not self._passes_filters(example, filters):
                continue
            hits.append(self._build_result(example, tool_id))
            if len(hits) >= k:
                break
        return hits
