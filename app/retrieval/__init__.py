"""Retrieval layer — context-adaptive grounding for the quest pipeline.

See ``docs/superpowers/specs/2026-04-14-retrieval-layer-design.md`` for the
overall design. Public names grow as each retrieval wave lands.

Import order matters: leaf retrievers without planning deps are imported
first, so that when a deeper retriever triggers ``app.world`` (and
therefore ``app.planning``) re-entry into ``app.retrieval`` can resolve
every name the planners ask for.
"""

from __future__ import annotations

from app.retrieval.interface import Query, QueryFilters, Result, Retriever
from app.retrieval.embeddings import Embedder, EmbeddingCache
from app.retrieval.craft_retriever import CraftRetriever
from app.retrieval.motif_retriever import MotifRetriever
from app.retrieval.passage_retriever import PassageRetriever
from app.retrieval.scene_retriever import SceneShapeRetriever
from app.retrieval.voice_retriever import VoiceRetriever
# World-touching retrievers imported last to keep the cycle resolvable
# during planner re-entry.
from app.retrieval.quest_retriever import QuestRetriever
from app.retrieval.foreshadowing_retriever import ForeshadowingRetriever

__all__ = [
    "Query",
    "QueryFilters",
    "Result",
    "Retriever",
    "Embedder",
    "EmbeddingCache",
    "PassageRetriever",
    "CraftRetriever",
    "MotifRetriever",
    "QuestRetriever",
    "SceneShapeRetriever",
    "VoiceRetriever",
    "ForeshadowingRetriever",
]
