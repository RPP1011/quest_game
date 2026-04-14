"""Shared interfaces for retrievers.

NOTE (Wave 1b): This file is intended to be owned by Wave 1a. It lands here
as a local placeholder so ``passage_retriever.py`` has a stable shape to
import. When Wave 1a's commit lands, the merge step should reconcile the
two versions — fields added upstream (e.g. ``seed_text``) will extend this
dataclass; ``PassageRetriever`` only consumes a subset so it should merge
cleanly either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class QueryFilters:
    """Metadata filter slice of a retrieval query.

    All fields are optional; omitted filters are not applied. See
    ``PassageRetriever.retrieve`` for which ones it honors.
    """

    pov: str | None = None
    is_quest: bool | None = None
    # Dimension name -> inclusive (low, high) range.
    score_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # work_ids to drop from results.
    exclude_works: set[str] = field(default_factory=set)
    # Entity-name boosts (used by QuestRetriever; ignored here).
    entity_mentions: set[str] = field(default_factory=set)
    # Per-quest retriever temporal filters (ignored by PassageRetriever).
    max_updates_ago: int | None = None
    last_n_records: int | None = None
    # CraftRetriever filters (ignored by PassageRetriever).
    tool_id: str | None = None
    scale: str | None = None
    register: str | None = None


@dataclass
class Query:
    """A retrieval query. Fields not used by a given retriever are ignored."""

    filters: QueryFilters = field(default_factory=QueryFilters)
    seed_text: str | None = None
    # Free-form context the retriever may use or surface in logs.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """A single retrieval hit."""

    work_id: str
    passage_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]: ...
