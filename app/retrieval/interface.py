"""Retrieval interface: ``Query``, ``Result``, and the ``Retriever`` Protocol.

The pipeline builds a :class:`Query` describing what grounding it needs
(semantic seed text plus structured filters), hands it to a concrete
retriever implementation, and receives a list of :class:`Result`
wrappers with a score and source metadata.

Schema convention matches ``app/planning/schemas.py`` — pydantic
``BaseModel`` with ``from __future__ import annotations``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Query(BaseModel):
    """A retrieval request.

    ``seed_text`` is the semantic anchor (optional — some retrievers
    only use ``filters`` and ``k``). ``filters`` is an open-ended dict
    because each retriever consumes a different subset of keys (POV,
    score ranges, entity mentions, tool id, etc.). ``k`` is the default
    number of results to return; callers can override per-call via the
    ``retrieve`` method's ``k`` kwarg.
    """

    seed_text: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    k: int = 3


class Result(BaseModel):
    """A single retrieval hit.

    ``source_id`` uniquely identifies the passage/record in its source
    corpus (e.g. ``"joyce_dubliners/passage_004"`` for the literary
    corpus or ``"<quest_id>/<update>/<scene>"`` for the quest corpus).
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
