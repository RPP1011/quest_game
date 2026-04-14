"""Retrieval layer — context-adaptive grounding for the quest pipeline.

See ``docs/superpowers/specs/2026-04-14-retrieval-layer-design.md`` for the
overall design. Combines the Wave 1a consumer interface (``Query``,
``Result``, ``Retriever`` Protocol) and embedding/cache layer with the
Wave 1b ``PassageRetriever`` (literary corpus, metadata-only mode).
Public names will grow as each retrieval wave adds its retriever.
"""

from __future__ import annotations

from app.retrieval.embeddings import Embedder, EmbeddingCache
from app.retrieval.interface import Query, QueryFilters, Result, Retriever
from app.retrieval.passage_retriever import PassageRetriever

__all__ = [
    "Query",
    "QueryFilters",
    "Result",
    "Retriever",
    "Embedder",
    "EmbeddingCache",
    "PassageRetriever",
]
