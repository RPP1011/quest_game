"""Embedding computation + on-disk cache for the retrieval layer.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` (22MB, 384-dim, CPU
friendly) per the retrieval design. The model is loaded lazily on
first use and cached for the lifetime of the :class:`Embedder`.

:class:`EmbeddingCache` persists vectors to a ``.npy`` file alongside a
sibling ``.json`` id index. The two files together describe a fixed
``(key -> vector)`` mapping. If the set of keys requested at load time
differs from the persisted set, the cache rebuilds from scratch;
otherwise the vectors are loaded straight off disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import-time avoidance only
    from sentence_transformers import SentenceTransformer


_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_SEQ_LENGTH = 512
EMBED_DIM = 384


class Embedder:
    """Wraps ``sentence-transformers`` with a lazy-loaded model.

    Inputs are truncated to 512 tokens (the MiniLM default context
    window) via ``model.max_seq_length``. Outputs are L2-normalized
    ``float32`` arrays so that cosine similarity reduces to a plain
    dot product.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    # ---------------------------------------------------------------
    # Model lifecycle
    # ---------------------------------------------------------------
    def _load(self) -> SentenceTransformer:
        if self._model is None:
            # Imported lazily to avoid pulling torch at import time.
            from sentence_transformers import SentenceTransformer

            # MiniLM is 22MB; CPU is plenty fast and avoids fighting
            # vllm for GPU memory. Override with device="cuda" when
            # vllm is not running (e.g. test suite with no LLM server).
            # ``QUEST_EMBEDDER_DEVICE`` env var is the recommended lever.
            import os as _os
            device = self._device or _os.environ.get("QUEST_EMBEDDER_DEVICE", "cpu")
            model = SentenceTransformer(self._model_name, device=device)
            model.max_seq_length = _MAX_SEQ_LENGTH
            self._model = model
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    # ---------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------
    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a ``(EMBED_DIM,)`` float32 vector."""
        vec = self.embed_many([text])[0]
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into a ``(N, EMBED_DIM)`` float32 array.

        The output is L2-normalized so cosine similarity == dot product.
        An empty input returns an empty ``(0, EMBED_DIM)`` array without
        loading the model.
        """
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        model = self._load()
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    # ---------------------------------------------------------------
    # Similarity
    # ---------------------------------------------------------------
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float | np.ndarray:
        """Cosine similarity between two vectors, or batch variants.

        - ``a`` and ``b`` both 1-D: returns a scalar ``float``.
        - ``a`` 2-D ``(N, D)`` and ``b`` 1-D ``(D,)``: returns ``(N,)``.
        - ``a`` 2-D ``(N, D)`` and ``b`` 2-D ``(M, D)``: returns ``(N, M)``.

        Accepts un-normalized inputs; divides by the L2 norms.
        """
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim == 1 and b.ndim == 1:
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        if a.ndim == 1:
            a = a[np.newaxis, :]
            squeeze_rows = True
        else:
            squeeze_rows = False
        if b.ndim == 1:
            b = b[np.newaxis, :]
            squeeze_cols = True
        else:
            squeeze_cols = False
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        a_norm = np.where(a_norm == 0.0, 1.0, a_norm)
        b_norm = np.where(b_norm == 0.0, 1.0, b_norm)
        sim = (a / a_norm) @ (b / b_norm).T
        if squeeze_rows and squeeze_cols:
            return float(sim[0, 0])
        if squeeze_rows:
            return sim[0]
        if squeeze_cols:
            return sim[:, 0]
        return sim


class EmbeddingCache:
    """On-disk embedding cache keyed by a stable set of ``(key, text)`` pairs.

    Two files make up one cache:

    - ``<cache_path>``: numpy ``(N, 384)`` float32 vectors.
    - ``<cache_path>.with_suffix(".json")``: JSON-encoded parallel
      array of string ids ``["<key_0>", "<key_1>", ...]``.

    If a cache exists and its id set matches the requested keys
    exactly, vectors are loaded and returned mapped key->vector. If the
    id set differs, the cache is rebuilt using the provided
    :class:`Embedder`.

    Order of the returned dict matches the order of the input pairs.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    @staticmethod
    def _ids_path_for(cache_path: Path) -> Path:
        return cache_path.with_suffix(".json")

    # ---------------------------------------------------------------
    # Load / build
    # ---------------------------------------------------------------
    def load_or_build(
        self,
        pairs: Iterable[tuple[str, str]],
        cache_path: str | Path,
    ) -> dict[str, np.ndarray]:
        """Return ``{key: vector}`` for each pair, building/loading as needed.

        Parameters
        ----------
        pairs:
            Iterable of ``(key, text)`` pairs. Keys must be unique and
            stable across runs; they drive cache invalidation.
        cache_path:
            Path to the ``.npy`` file. A sibling ``.json`` file at
            ``cache_path.with_suffix('.json')`` holds the id array.

        Behaviour
        ---------
        - If both files exist and the on-disk id set matches the input
          keys (order-independent), vectors are loaded from disk.
        - Otherwise, texts are embedded and persisted.
        """
        cache_path = Path(cache_path)
        ids_path = self._ids_path_for(cache_path)

        pair_list = list(pairs)
        keys = [k for k, _ in pair_list]
        if len(set(keys)) != len(keys):
            dupes = [k for k in keys if keys.count(k) > 1]
            raise ValueError(f"EmbeddingCache pairs must have unique keys; duplicates: {sorted(set(dupes))}")

        if cache_path.exists() and ids_path.exists():
            try:
                stored_ids = json.loads(ids_path.read_text())
            except (OSError, json.JSONDecodeError):
                stored_ids = None
            if isinstance(stored_ids, list) and set(stored_ids) == set(keys):
                vectors = np.load(cache_path)
                index = {sid: i for i, sid in enumerate(stored_ids)}
                return {k: vectors[index[k]] for k in keys}

        # Rebuild
        texts = [t for _, t in pair_list]
        vectors = self._embedder.embed_many(texts)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vectors)
        # np.save appends ``.npy`` if missing; normalize the written path for the
        # ids file so .json sits next to the actually-written .npy.
        actual_cache_path = cache_path if cache_path.suffix == ".npy" else cache_path.with_suffix(".npy")
        actual_ids_path = self._ids_path_for(actual_cache_path)
        actual_ids_path.write_text(json.dumps(keys))
        return {k: vectors[i] for i, k in enumerate(keys)}
