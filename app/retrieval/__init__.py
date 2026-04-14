"""Retrieval layer — literary + per-quest context grounding.

See ``docs/superpowers/specs/2026-04-14-retrieval-layer-design.md`` for the
overall design. Wave 1b provides ``PassageRetriever`` in metadata-only mode.
"""
from .interface import Query, QueryFilters, Result, Retriever
from .passage_retriever import PassageRetriever

__all__ = [
    "Query",
    "QueryFilters",
    "Result",
    "Retriever",
    "PassageRetriever",
]
