"""Tests for app/retrieval/embeddings.py — Embedder + EmbeddingCache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.retrieval.embeddings import EMBED_DIM, Embedder, EmbeddingCache


# ---------------------------------------------------------------------------
# Module-scoped embedder fixture.  Loads MiniLM once for the whole file.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


@pytest.fixture(scope="module")
def toy_sentences() -> list[str]:
    # First two are near-similar (both about cats sleeping in sunbeams);
    # third is about an unrelated topic.
    return [
        "The cat curled in the sunbeam, asleep.",
        "A small cat dozed in a patch of afternoon sun.",
        "Diesel engines hauled the freight across the plains.",
    ]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


def test_embed_one_shape(embedder: Embedder, toy_sentences: list[str]):
    vec = embedder.embed_one(toy_sentences[0])
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32


def test_embed_many_shape(embedder: Embedder, toy_sentences: list[str]):
    mat = embedder.embed_many(toy_sentences)
    assert mat.shape == (3, EMBED_DIM)
    assert mat.dtype == np.float32


def test_embed_many_empty_returns_empty_array(embedder: Embedder):
    # Must not load the model for an empty batch.
    mat = embedder.embed_many([])
    assert mat.shape == (0, EMBED_DIM)


def test_cosine_self_is_one(embedder: Embedder, toy_sentences: list[str]):
    v = embedder.embed_one(toy_sentences[0])
    sim = Embedder.cosine(v, v)
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_cosine_similar_vs_different(embedder: Embedder, toy_sentences: list[str]):
    v_cat1 = embedder.embed_one(toy_sentences[0])
    v_cat2 = embedder.embed_one(toy_sentences[1])
    v_train = embedder.embed_one(toy_sentences[2])
    sim_similar = Embedder.cosine(v_cat1, v_cat2)
    sim_different = Embedder.cosine(v_cat1, v_train)
    assert sim_similar > sim_different, (
        f"expected near-similar > different; got {sim_similar=} vs {sim_different=}"
    )


def test_cosine_batch_vector_vs_matrix(embedder: Embedder, toy_sentences: list[str]):
    mat = embedder.embed_many(toy_sentences)
    query = embedder.embed_one(toy_sentences[0])
    sims = Embedder.cosine(mat, query)
    assert isinstance(sims, np.ndarray)
    assert sims.shape == (3,)
    # First index is self-similarity — highest of the three.
    assert np.argmax(sims) == 0
    assert sims[0] == pytest.approx(1.0, abs=1e-5)


def test_cosine_batch_matrix_vs_matrix(embedder: Embedder, toy_sentences: list[str]):
    mat = embedder.embed_many(toy_sentences)
    sims = Embedder.cosine(mat, mat)
    assert sims.shape == (3, 3)
    # Diagonal is self-similarity.
    for i in range(3):
        assert sims[i, i] == pytest.approx(1.0, abs=1e-5)


def test_cosine_zero_vector_is_safe():
    z = np.zeros(EMBED_DIM, dtype=np.float32)
    nonzero = np.ones(EMBED_DIM, dtype=np.float32)
    assert Embedder.cosine(z, nonzero) == 0.0
    assert Embedder.cosine(z, z) == 0.0


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


def test_cache_persists_and_reloads(
    embedder: Embedder, toy_sentences: list[str], tmp_path: Path
):
    cache = EmbeddingCache(embedder)
    pairs = [(f"toy_{i}", text) for i, text in enumerate(toy_sentences)]
    cache_path = tmp_path / "vecs.npy"

    built = cache.load_or_build(pairs, cache_path)
    assert set(built.keys()) == {"toy_0", "toy_1", "toy_2"}
    for v in built.values():
        assert v.shape == (EMBED_DIM,)

    assert cache_path.exists()
    ids_path = cache_path.with_suffix(".json")
    assert ids_path.exists()
    stored_ids = json.loads(ids_path.read_text())
    assert set(stored_ids) == {"toy_0", "toy_1", "toy_2"}

    # Second call with the same pairs should load from disk (not rebuild).
    # Swap in a no-embed embedder to prove the model is never touched.
    class _FailingEmbedder:
        def embed_many(self, texts):  # pragma: no cover - must not run
            raise AssertionError("cache must not rebuild when keys match")

    cache2 = EmbeddingCache(_FailingEmbedder())  # type: ignore[arg-type]
    reloaded = cache2.load_or_build(pairs, cache_path)
    assert set(reloaded.keys()) == set(built.keys())
    for key in built:
        np.testing.assert_array_equal(reloaded[key], built[key])


def test_cache_reload_is_order_independent(
    embedder: Embedder, toy_sentences: list[str], tmp_path: Path
):
    cache = EmbeddingCache(embedder)
    pairs = [(f"toy_{i}", text) for i, text in enumerate(toy_sentences)]
    cache_path = tmp_path / "vecs.npy"

    built = cache.load_or_build(pairs, cache_path)

    # Reload with reversed order — should hit disk cache, and returned dict
    # should map each key to the same vector it was built with.
    class _FailingEmbedder:
        def embed_many(self, texts):  # pragma: no cover
            raise AssertionError("cache must not rebuild when keys match")

    cache2 = EmbeddingCache(_FailingEmbedder())  # type: ignore[arg-type]
    reloaded = cache2.load_or_build(list(reversed(pairs)), cache_path)
    for key in built:
        np.testing.assert_array_equal(reloaded[key], built[key])


def test_cache_invalidates_on_key_set_change(
    embedder: Embedder, toy_sentences: list[str], tmp_path: Path
):
    cache = EmbeddingCache(embedder)
    first_pairs = [(f"toy_{i}", text) for i, text in enumerate(toy_sentences)]
    cache_path = tmp_path / "vecs.npy"
    cache.load_or_build(first_pairs, cache_path)

    # New pair introduced: the cache must rebuild.
    new_pairs = first_pairs + [("toy_3", "A fresh sentence to force invalidation.")]

    # Use a spy embedder to confirm a rebuild happens.
    class _SpyEmbedder:
        def __init__(self, inner: Embedder) -> None:
            self.inner = inner
            self.calls = 0

        def embed_many(self, texts):
            self.calls += 1
            return self.inner.embed_many(texts)

    spy = _SpyEmbedder(embedder)
    cache_spy = EmbeddingCache(spy)  # type: ignore[arg-type]
    rebuilt = cache_spy.load_or_build(new_pairs, cache_path)
    assert spy.calls == 1
    assert set(rebuilt.keys()) == {"toy_0", "toy_1", "toy_2", "toy_3"}

    # Removing a key also forces a rebuild.
    shrunk_pairs = first_pairs[:2]
    spy2 = _SpyEmbedder(embedder)
    cache_spy2 = EmbeddingCache(spy2)  # type: ignore[arg-type]
    shrunk = cache_spy2.load_or_build(shrunk_pairs, cache_path)
    assert spy2.calls == 1
    assert set(shrunk.keys()) == {"toy_0", "toy_1"}


def test_cache_duplicate_keys_raise(embedder: Embedder, tmp_path: Path):
    cache = EmbeddingCache(embedder)
    pairs = [("k", "a"), ("k", "b")]
    cache_path = tmp_path / "vecs.npy"
    with pytest.raises(ValueError, match="unique keys"):
        cache.load_or_build(pairs, cache_path)
