"""Retrieval layer — context-adaptive grounding for the quest pipeline.

See ``docs/superpowers/specs/2026-04-14-retrieval-layer-design.md`` for the
overall design. Public names grow as each retrieval wave lands.
"""

from __future__ import annotations

# Interface + leaf retrievers with no planning deps are imported first,
# so that when a deeper retriever triggers ``app.world`` (and therefore
# ``app.planning``) re-entry into ``app.retrieval`` can resolve every
# name ``dramatic_planner`` / ``craft_planner`` ask for.
from app.retrieval.interface import Query, QueryFilters, Result, Retriever
from app.retrieval.embeddings import Embedder, EmbeddingCache
from app.retrieval.craft_retriever import CraftRetriever
from app.retrieval.passage_retriever import PassageRetriever
from app.retrieval.scene_retriever import SceneShapeRetriever
# ``quest_retriever`` and ``foreshadowing_retriever`` reach into
# ``app.world``. Importing them last keeps the earlier names available
# to any planner module that re-imports from ``app.retrieval`` during
# the world→seed→planning re-entry.
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
    "QuestRetriever",
    "SceneShapeRetriever",
    "ForeshadowingRetriever",
]
