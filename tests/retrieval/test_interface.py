"""Tests for app/retrieval/interface.py — Query / Result dataclasses."""
from __future__ import annotations

from typing import get_type_hints

import pytest

from app.retrieval.interface import Query, Result, Retriever


def test_query_defaults_and_filter_access():
    q = Query()
    assert q.seed_text is None
    assert q.filters == {}
    assert q.k == 3


def test_query_with_all_fields():
    q = Query(
        seed_text="a lonely corridor at dusk",
        filters={
            "pov": "second_person",
            "score_ranges": {"voice_distinctiveness": (0.7, 1.0)},
            "exclude_works": {"joyce_ulysses"},
        },
        k=5,
    )
    assert q.seed_text == "a lonely corridor at dusk"
    assert q.k == 5
    assert q.filters["pov"] == "second_person"
    # score_ranges subkey access
    low, high = q.filters["score_ranges"]["voice_distinctiveness"]
    assert low == 0.7 and high == 1.0
    # set semantics survive round-trip
    assert "joyce_ulysses" in q.filters["exclude_works"]


def test_query_filters_are_mutable_dict():
    q = Query()
    q.filters["pov"] = "third_limited"
    assert q.filters == {"pov": "third_limited"}


def test_result_fields():
    r = Result(
        source_id="joyce_dubliners/passage_004",
        text="He watched the street from the window.",
        score=0.87,
        metadata={
            "work_id": "joyce_dubliners",
            "pov": "third_limited",
            "scores": {"voice_distinctiveness": 0.82},
        },
    )
    assert r.source_id == "joyce_dubliners/passage_004"
    assert r.text.startswith("He watched")
    assert r.score == pytest.approx(0.87)
    assert r.metadata["work_id"] == "joyce_dubliners"
    assert r.metadata["scores"]["voice_distinctiveness"] == pytest.approx(0.82)


def test_result_defaults_metadata_to_empty_dict():
    r = Result(source_id="x/y", text="hello", score=0.5)
    assert r.metadata == {}


def test_retriever_protocol_is_runtime_checkable():
    """A minimal async-retrieve object should structurally match ``Retriever``."""

    class _Stub:
        async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
            return []

    assert isinstance(_Stub(), Retriever)


def test_retriever_protocol_rejects_non_matching():
    class _NotRetriever:
        def something_else(self) -> None:
            return None

    assert not isinstance(_NotRetriever(), Retriever)


def test_retriever_protocol_has_retrieve_method():
    hints = get_type_hints(Retriever.retrieve)
    # Just assert the method exists with an annotated query parameter.
    assert "query" in hints
