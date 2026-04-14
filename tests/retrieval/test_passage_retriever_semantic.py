"""Tests for ``PassageRetriever`` Wave 2a semantic (hybrid) mode.

Covers:
* ``enable_semantic=True`` no longer raises.
* ``seed_text`` drives semantic rerank — a passage whose content
  matches the seed outranks unrelated passages with equal metadata.
* No ``seed_text`` + ``enable_semantic=True`` matches metadata-only
  output (no hidden embedding call necessary).
* Embedding cache persists to ``cache_path`` and a second
  :class:`PassageRetriever` loads identical vectors from disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.retrieval import Embedder, PassageRetriever, Query, QueryFilters


# -- Fixtures -------------------------------------------------------------
#
# Tiny 4-passage corpus with distinct topical content so MiniLM has
# signal to discriminate. Two passages are in the same work with the
# same POV + scores, so metadata-only ranking ties them: semantic rerank
# is what makes the topical match win.


def _write_manifest(path: Path) -> None:
    data = {
        "version": 1,
        "scoring": {"critic_error_weight": 0.25, "critic_warning_weight": 0.10},
        "works": [
            {
                "id": "mixed_work",
                "title": "Mixed Work",
                "author": "Anon",
                "year": 1900,
                "pov": "third_limited",
                "is_quest": False,
                "expected": {
                    "voice_distinctiveness": 0.8,
                    "clarity": 0.8,
                },
                "passages": [
                    {"id": p, "sha256": "PENDING", "expected_high": [], "expected_low": []}
                    for p in ("ocean", "forest", "train", "kitchen")
                ],
            },
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


# Four distinct topics. Each passage has identical dimension scores so
# metadata-only ranking is a tie; the only discriminator is semantics.
_PASSAGES: dict[str, str] = {
    "ocean": (
        "The sea thundered against the cliffs, white spray climbing the rocks. "
        "Seagulls wheeled above the tideline, and the salt wind scoured her face."
    ),
    "forest": (
        "Pines rose on every side, silent and green, and the forest floor was thick "
        "with needles. Somewhere a woodpecker knocked twice and then fell still."
    ),
    "train": (
        "The locomotive hauled its cars east across the plains, black smoke trailing. "
        "Steel wheels sang on steel rails, hour after hour after hour."
    ),
    "kitchen": (
        "She set a kettle on the stove. The gas hissed blue and the kitchen filled "
        "with the smell of tea and of bread warming in the oven."
    ),
}


def _write_labels(labels_dir: Path) -> None:
    shard = {
        "passages": [
            {
                "work_id": "mixed_work",
                "passage_id": pid,
                "is_quest": False,
                "dimensions": {
                    "voice_distinctiveness": 0.8,
                    "clarity": 0.8,
                },
                "rationale": f"mixed_work {pid}",
            }
            for pid in _PASSAGES
        ],
    }
    (labels_dir / "labels_claude_part_alpha.json").write_text(json.dumps(shard))


def _write_passages(passages_dir: Path) -> None:
    wdir = passages_dir / "mixed_work"
    wdir.mkdir(parents=True, exist_ok=True)
    for pid, body in _PASSAGES.items():
        (wdir / f"{pid}.txt").write_text(body + "\n", encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.yaml"
    labels = tmp_path / "labels"
    passages = tmp_path / "passages"
    labels.mkdir()
    passages.mkdir()
    _write_manifest(manifest)
    _write_labels(labels)
    _write_passages(passages)
    return manifest, labels, passages


# One shared embedder across the module — MiniLM loads exactly once.
@pytest.fixture(scope="module")
def shared_embedder() -> Embedder:
    return Embedder()


# -- Tests ----------------------------------------------------------------


def test_enable_semantic_does_not_raise(
    corpus: tuple[Path, Path, Path], tmp_path: Path, shared_embedder: Embedder
) -> None:
    manifest, labels, passages = corpus
    cache_path = tmp_path / "cache" / "passage_embeddings.npy"
    # Must not raise NotImplementedError any more (Wave 2a turned this on).
    PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=shared_embedder,
    )


def test_init_is_lazy(
    corpus: tuple[Path, Path, Path], tmp_path: Path, shared_embedder: Embedder
) -> None:
    """Constructing with ``enable_semantic=True`` must not touch the cache.

    The embedding cache + model should only be materialized on the first
    ``retrieve()`` call.
    """
    manifest, labels, passages = corpus
    cache_path = tmp_path / "cache" / "passage_embeddings.npy"

    PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=shared_embedder,
    )
    assert not cache_path.exists(), "cache built in __init__; must be lazy"


async def test_seed_text_drives_semantic_rerank(
    corpus: tuple[Path, Path, Path], tmp_path: Path, shared_embedder: Embedder
) -> None:
    manifest, labels, passages = corpus
    cache_path = tmp_path / "cache" / "passage_embeddings.npy"

    retriever = PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=shared_embedder,
    )
    q = Query(seed_text="Steam trains across the railroad plains.")
    results = await retriever.retrieve(q, k=4)

    # All four indexed passages should come back, but the top hit must be
    # the train passage — it is the only topical match for the seed.
    assert len(results) == 4
    top = results[0]
    assert (top.metadata["work_id"], top.metadata["passage_id"]) == (
        "mixed_work",
        "train",
    )
    # Train must rank strictly ahead of unrelated passages.
    train_score = top.score
    other_scores = [r.score for r in results[1:]]
    assert all(train_score > s for s in other_scores), (
        f"expected train top; got {[(r.metadata['passage_id'], r.score) for r in results]}"
    )


async def test_no_seed_text_matches_metadata_only(
    corpus: tuple[Path, Path, Path], tmp_path: Path, shared_embedder: Embedder
) -> None:
    manifest, labels, passages = corpus
    cache_path = tmp_path / "cache" / "passage_embeddings.npy"

    semantic = PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=shared_embedder,
    )
    metadata_only = PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=False,
    )

    q = Query(
        filters=QueryFilters(
            score_ranges={"voice_distinctiveness": (0.7, 1.0)},
        ).to_dict()
    )
    sem_results = await semantic.retrieve(q, k=10)
    md_results = await metadata_only.retrieve(q, k=10)

    sem_keys = [(r.metadata["work_id"], r.metadata["passage_id"]) for r in sem_results]
    md_keys = [(r.metadata["work_id"], r.metadata["passage_id"]) for r in md_results]
    assert sem_keys == md_keys
    for s, m in zip(sem_results, md_results):
        assert s.score == pytest.approx(m.score)
    # Without a seed_text the embedding cache must not have been built either.
    assert not cache_path.exists()


async def test_cache_persists_and_reloads_identical(
    corpus: tuple[Path, Path, Path], tmp_path: Path, shared_embedder: Embedder
) -> None:
    manifest, labels, passages = corpus
    cache_path = tmp_path / "cache" / "passage_embeddings.npy"

    first = PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=shared_embedder,
    )
    # Triggering retrieve with a seed_text materializes the cache on disk.
    await first.retrieve(Query(seed_text="any prompt"), k=1)
    assert cache_path.exists()
    ids_path = cache_path.with_suffix(".json")
    assert ids_path.exists()

    stored_ids = json.loads(ids_path.read_text())
    # All four passages are represented in the cache (none had empty text).
    assert set(stored_ids) == {f"mixed_work/{pid}" for pid in _PASSAGES}
    vectors_first = np.load(cache_path)
    assert vectors_first.shape == (4, 384)

    # Use a spy embedder on the second instance to prove we don't re-embed.
    class _SpyEmbedder(Embedder):
        def __init__(self, inner: Embedder) -> None:
            super().__init__()
            self._inner = inner
            self.embed_many_calls = 0
            self.embed_one_calls = 0

        def embed_many(self, texts):  # type: ignore[override]
            self.embed_many_calls += 1
            return self._inner.embed_many(texts)

        def embed_one(self, text):  # type: ignore[override]
            self.embed_one_calls += 1
            return self._inner.embed_one(text)

    spy = _SpyEmbedder(shared_embedder)
    second = PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=spy,
    )
    await second.retrieve(Query(seed_text="another prompt"), k=1)

    # Second instance must have skipped the bulk embed of the corpus
    # (served from disk) and only embedded the query seed.
    assert spy.embed_many_calls == 0
    assert spy.embed_one_calls == 1

    # Cache contents on disk must be bit-identical after the reload.
    vectors_second = np.load(cache_path)
    np.testing.assert_array_equal(vectors_first, vectors_second)


async def test_filters_still_apply_in_semantic_mode(
    corpus: tuple[Path, Path, Path], tmp_path: Path, shared_embedder: Embedder
) -> None:
    manifest, labels, passages = corpus
    cache_path = tmp_path / "cache" / "passage_embeddings.npy"

    retriever = PassageRetriever(
        manifest,
        labels,
        passages,
        enable_semantic=True,
        embedding_cache_path=cache_path,
        embedder=shared_embedder,
    )
    # exclude_works should drop everything: the only indexed work is mixed_work.
    q = Query(
        seed_text="Steam trains across the railroad plains.",
        filters=QueryFilters(exclude_works={"mixed_work"}).to_dict(),
    )
    results = await retriever.retrieve(q, k=4)
    assert results == []
    # Empty candidate set must also avoid spinning up the embedding cache.
    assert not cache_path.exists()
