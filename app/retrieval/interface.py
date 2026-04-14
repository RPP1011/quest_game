"""Retrieval interface: ``Query``, ``Result``, and the ``Retriever`` Protocol.

The pipeline builds a :class:`Query` describing what grounding it needs
(semantic seed text plus structured filters), hands it to a concrete
retriever implementation, and receives a list of :class:`Result`
wrappers with a score and source metadata.

Schema convention matches ``app/planning/schemas.py`` — pydantic
``BaseModel`` with ``from __future__ import annotations``.

``QueryFilters`` is a typed-key reference for the most common
``Query.filters`` shape; concrete retrievers may also accept
retriever-specific keys via the open ``dict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@dataclass
class QueryFilters:
    """Reference shape for the common keys consumers put in ``Query.filters``.

    All fields are optional; omitted filters are not applied. Each
    retriever consumes a subset:
      * ``PassageRetriever`` (Wave 1b): pov, is_quest, score_ranges,
        exclude_works.
      * ``QuestRetriever`` (Wave 3): entity_mentions, max_updates_ago,
        last_n_records.
      * ``CraftRetriever`` (Wave 2): tool_id, scale, register, pov.

    Use ``filters.to_dict()`` when constructing a ``Query`` if you want
    typed-key safety; otherwise pass a plain dict.
    """

    pov: str | None = None
    is_quest: bool | None = None
    score_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    exclude_works: set[str] = field(default_factory=set)
    entity_mentions: set[str] = field(default_factory=set)
    max_updates_ago: int | None = None
    last_n_records: int | None = None
    tool_id: str | None = None
    scale: str | None = None
    register: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is None or v == {} or v == set():
                continue
            out[k] = v
        return out


class Query(BaseModel):
    """A retrieval request.

    ``seed_text`` is the semantic anchor (optional — some retrievers
    only use ``filters`` and ``k``). ``filters`` is an open-ended dict
    because each retriever consumes a different subset of keys (POV,
    score ranges, entity mentions, tool id, etc.) — see
    :class:`QueryFilters` for the typed reference. ``k`` is the default
    number of results to return; callers can override per-call via the
    ``retrieve`` method's ``k`` kwarg.
    """

    seed_text: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    k: int = 3


class Result(BaseModel):
    """A single retrieval hit.

    ``source_id`` uniquely identifies the passage/record in its source
    corpus (e.g. ``"<work_id>/<passage_id>"`` for the literary corpus
    or ``"<quest_id>/<update>/<scene>"`` for the quest corpus).
    ``text`` is the retrieved prose; ``score`` is the retriever's
    composite match score (higher is better; 1.0 is a perfect match).
    ``metadata`` is the per-retriever payload callers need for prompt
    injection — POV, narrator type, score breakdown, entity mentions,
    tool id, etc.
    """

    source_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Structural interface implemented by each concrete retriever.

    Implementations are expected to be async (they may perform I/O
    against SQLite, numpy-backed embedding caches, or — in future —
    remote services).
    """

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        ...
