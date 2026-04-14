"""Tests for ``QuestRetriever`` (Wave 3a).

Fixtures seed a :class:`WorldStateManager` with 5 pre-baked
``narrative_embeddings`` rows. A deterministic ``_FakeEmbedder`` projects
a tiny vocabulary onto orthogonal 8-dim axes so cosine similarity is
trivially predictable per test: the seed text and the matching row share
a keyword token and therefore a dimension, which ranks them above other
candidates.

The writer hook that populates ``narrative_embeddings`` in production
ships in Wave 3b. This suite exercises the read path only, using
manually inserted rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.retrieval import Embedder, Query, QueryFilters, QuestRetriever
from app.world.db import open_db
from app.world.state_manager import WorldStateManager


# -- Fake embedder -------------------------------------------------------
#
# Maps a closed vocabulary of "topic" words to orthogonal 8-d axes. A
# text with topic k gets the unit vector along axis k (dim 7 reserved as
# a shared/background component to keep cosines non-degenerate).
_TOPIC_AXES: dict[str, int] = {
    "ocean": 0,
    "forest": 1,
    "train": 2,
    "kitchen": 3,
    "library": 4,
}
_EMBED_DIM = 8


def _vector_for(text: str) -> np.ndarray:
    """Project ``text`` onto the topic axes by presence of keywords.

    Case-insensitive substring match. Sums axes for every topic word
    that appears in ``text`` and L2-normalizes. Texts with no keyword
    fall back to a zero vector, which has cosine 0 against everything.
    """
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    lowered = text.lower()
    for word, axis in _TOPIC_AXES.items():
        if word in lowered:
            vec[axis] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


class _FakeEmbedder(Embedder):
    """Deterministic topic-axis embedder — no model load, no torch."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        # Intentionally bypass ``Embedder.__init__`` so we never touch
        # ``sentence-transformers``.
        self._model_name = "fake"
        self._model = None

    def embed_one(self, text: str) -> np.ndarray:  # type: ignore[override]
        return _vector_for(text)

    def embed_many(self, texts):  # type: ignore[override]
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        return np.stack([_vector_for(t) for t in texts])


# -- World fixture -------------------------------------------------------


_PREVIEWS: list[tuple[int, int, str]] = [
    # (update_number, scene_index, text_preview)
    (1, 0, "The ocean crashed beneath the cliffs as gulls wheeled overhead."),
    (2, 0, "A quiet forest path wound through pines and moss-soft stones."),
    (3, 0, "The train rumbled east across the plains, smoke unspooling behind it."),
    (4, 0, "She stirred soup in the warm kitchen while rain struck the window."),
    (5, 0, "Alyss stood in the library; the shelves sagged with heavy books."),
]


@pytest.fixture()
def world(tmp_path: Path):
    """In-file WorldStateManager pre-seeded with 5 narrative_embeddings rows."""
    db_path = tmp_path / "quest.db"
    conn = open_db(db_path)
    sm = WorldStateManager(conn)
    for update_number, scene_index, preview in _PREVIEWS:
        sm.upsert_narrative_embedding(
            quest_id="q1",
            update_number=update_number,
            scene_index=scene_index,
            embedding=_vector_for(preview),
            text_preview=preview,
        )
    # Add one unrelated quest row to prove scope isolation is honored via
    # ``list_narrative_embeddings(quest_id=...)``.
    sm.upsert_narrative_embedding(
        quest_id="other_quest",
        update_number=99,
        scene_index=0,
        embedding=_vector_for("Dragons! Dragons everywhere."),
        text_preview="Dragons! Dragons everywhere.",
    )
    try:
        yield sm
    finally:
        conn.close()


@pytest.fixture()
def embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture()
def retriever(world: WorldStateManager, embedder: _FakeEmbedder) -> QuestRetriever:
    return QuestRetriever(world, "q1", embedder=embedder)


# -- Tests ---------------------------------------------------------------


async def test_missing_seed_text_returns_empty(retriever: QuestRetriever) -> None:
    """Callbacks are inherently semantic; no seed_text means no results."""
    assert await retriever.retrieve(Query(), k=5) == []
    # Empty string counts as missing per ``not query.seed_text``.
    assert await retriever.retrieve(Query(seed_text=""), k=5) == []


async def test_seed_text_topical_match_ranks_first(
    retriever: QuestRetriever,
) -> None:
    """The row whose preview shares the seed's topic word must rank first."""
    q = Query(seed_text="Steam trains thundering across the railway.")
    results = await retriever.retrieve(q, k=5)

    assert len(results) == 5
    top = results[0]
    # Row 3 is the only "train" preview; it must win the ranking.
    assert top.metadata["update_number"] == 3
    assert top.metadata["scene_index"] == 0
    assert top.source_id == "q1/u3/s0"
    # Score is the raw cosine for the top hit (no entity boost here).
    assert top.score == pytest.approx(1.0)
    # Every other candidate must score strictly lower.
    for r in results[1:]:
        assert r.score < top.score


async def test_max_updates_ago_drops_old_rows(
    retriever: QuestRetriever,
) -> None:
    """`max_updates_ago=1` keeps only rows within 1 update of the newest."""
    # Newest row is update 5; ``max_updates_ago=1`` keeps updates {4, 5}.
    # Seed touches only the kitchen topic axis so the kitchen row (update 4)
    # ranks first; library (update 5) has zero cosine and comes second.
    q = Query(
        seed_text="A warm kitchen filled with steam from the kettle.",
        filters=QueryFilters(max_updates_ago=1).to_dict(),
    )
    results = await retriever.retrieve(q, k=10)

    kept_updates = {r.metadata["update_number"] for r in results}
    assert kept_updates == {4, 5}
    # The kitchen preview is the topical match for the seed and must rank
    # ahead of the library preview on the topic axis.
    assert results[0].metadata["update_number"] == 4
    # Train (update 3) is older than the cutoff and must be dropped.
    assert 3 not in kept_updates
    assert 2 not in kept_updates
    assert 1 not in kept_updates


async def test_entity_mentions_boost_moves_borderline_match_up(
    world: WorldStateManager, embedder: _FakeEmbedder,
) -> None:
    """A 1.5x boost from entity_mentions lifts a weaker semantic match above a strong one."""
    # Use a seed that blends two topics so cosine scores are non-trivial.
    # With no boost: the kitchen preview is the strongest semantic match
    # (topic axis for 'kitchen' in the seed). With ``entity_mentions`` for
    # 'Alyss' (present only in the library preview), the 1.5x boost should
    # tip the library row above the kitchen row despite kitchen's higher
    # base cosine.
    retriever = QuestRetriever(world, "q1", embedder=embedder)

    # Baseline: no entity boost. Kitchen row wins.
    baseline = await retriever.retrieve(
        Query(seed_text="A quiet evening in the warm kitchen."),
        k=5,
    )
    assert baseline[0].metadata["update_number"] == 4, (
        f"baseline: expected kitchen top; got {[r.metadata for r in baseline]}"
    )

    # With boost: the library row mentions 'Alyss' and receives the 1.5x
    # multiplier. Even if its cosine is 0 (no topic axis overlap), the
    # multiplier alone is not enough — we need at least some overlap. Use
    # a seed that also touches the library axis so the library row has
    # nonzero base cosine, then watch the boost promote it above kitchen.
    boosted = await retriever.retrieve(
        Query(
            seed_text=(
                "She sat in the warm kitchen, thinking of books in a library."
            ),
            filters=QueryFilters(entity_mentions={"Alyss"}).to_dict(),
        ),
        k=5,
    )
    # Top result must be the library row (update 5), not the kitchen row.
    assert boosted[0].metadata["update_number"] == 5, (
        f"boosted: expected library (Alyss boost); got {[r.metadata for r in boosted]}"
    )
    # And the score reflects the 1.5x multiplier: the boosted row's score
    # must be strictly greater than its un-boosted cosine.
    library_preview = _PREVIEWS[4][2]  # library row
    raw_cosine = float(
        Embedder.cosine(
            _vector_for(library_preview),
            _vector_for(
                "She sat in the warm kitchen, thinking of books in a library."
            ),
        )
    )
    assert boosted[0].score == pytest.approx(raw_cosine * 1.5)


async def test_entity_mentions_case_insensitive(
    retriever: QuestRetriever,
) -> None:
    """Entity match is case-insensitive substring against text_preview."""
    # The library preview contains 'Alyss'. Query for 'ALYSS' (upcased).
    # Both library and non-library rows contribute; the boost should ensure
    # the library row's final score shows the 1.5x multiplier.
    results = await retriever.retrieve(
        Query(
            seed_text="books on shelves in the library",
            filters=QueryFilters(entity_mentions={"ALYSS"}).to_dict(),
        ),
        k=5,
    )
    # Library row must be first and carry the boosted score.
    top = results[0]
    assert top.metadata["update_number"] == 5
    library_preview = _PREVIEWS[4][2]
    raw_cosine = float(
        Embedder.cosine(
            _vector_for(library_preview),
            _vector_for("books on shelves in the library"),
        )
    )
    assert top.score == pytest.approx(raw_cosine * 1.5)


async def test_k_limits_results(retriever: QuestRetriever) -> None:
    q = Query(seed_text="trains across the plains")
    results = await retriever.retrieve(q, k=2)
    assert len(results) == 2
    # Top-2 are the highest scorers — order preserved.
    assert results[0].score >= results[1].score


async def test_last_n_records_truncates_candidate_pool(
    retriever: QuestRetriever,
) -> None:
    """`last_n_records=2` keeps only the 2 most-recent candidates pre-ranking."""
    # Train is update 3 — older than the last 2 records (updates 4, 5).
    # Even though "train" is the topical match, it must be excluded.
    q = Query(
        seed_text="The train rolled past the station.",
        filters=QueryFilters(last_n_records=2).to_dict(),
    )
    results = await retriever.retrieve(q, k=5)
    assert len(results) == 2
    kept_updates = {r.metadata["update_number"] for r in results}
    assert kept_updates == {4, 5}


async def test_result_shape(retriever: QuestRetriever) -> None:
    """source_id, text, and metadata match the spec."""
    q = Query(seed_text="ocean and cliffs")
    results = await retriever.retrieve(q, k=1)
    assert len(results) == 1
    r = results[0]
    # Ocean row is update 1, scene 0, which is the oldest row — recency_rank 4.
    assert r.source_id == "q1/u1/s0"
    assert r.text == _PREVIEWS[0][2]
    assert r.metadata["update_number"] == 1
    assert r.metadata["scene_index"] == 0
    # recency_rank is 0 for the newest row; ocean is the oldest of 5 rows.
    assert r.metadata["recency_rank"] == 4


async def test_quest_scope_isolation(
    world: WorldStateManager, embedder: _FakeEmbedder,
) -> None:
    """Retrieving for q1 never sees ``other_quest`` rows."""
    retriever = QuestRetriever(world, "q1", embedder=embedder)
    q = Query(seed_text="dragons")
    results = await retriever.retrieve(q, k=10)
    # The "dragons" preview lives under other_quest and must not surface.
    for r in results:
        assert r.source_id.startswith("q1/")
        assert "dragon" not in r.text.lower()


async def test_no_rows_returns_empty(
    embedder: _FakeEmbedder, tmp_path: Path,
) -> None:
    """Empty ``narrative_embeddings`` table means empty result."""
    conn = open_db(tmp_path / "empty.db")
    try:
        sm = WorldStateManager(conn)
        retriever = QuestRetriever(sm, "q1", embedder=embedder)
        results = await retriever.retrieve(Query(seed_text="anything"), k=5)
        assert results == []
    finally:
        conn.close()
