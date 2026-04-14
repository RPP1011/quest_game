"""Retrieval layer — context-adaptive grounding for the quest pipeline.

Wave 1a ships the consumer-facing interface (``Query``, ``Result``,
``Retriever`` Protocol) and the embedding/cache layer used by later
waves. Retriever implementations and pipeline integration land in
subsequent waves.

Public names are re-exported here and will be extended as each wave
adds its retriever.
"""

from __future__ import annotations

from app.retrieval.embeddings import Embedder, EmbeddingCache
from app.retrieval.interface import Query, Result, Retriever

__all__ = [
    "Query",
    "Result",
    "Retriever",
    "Embedder",
    "EmbeddingCache",
]
