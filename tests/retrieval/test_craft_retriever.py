"""Tests for ``CraftRetriever`` (Wave 2c)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.craft.library import CraftLibrary
from app.retrieval import CraftRetriever, Query, QueryFilters


# -- Fixtures -------------------------------------------------------------


def _write_tool(tools_dir: Path, tool_id: str, category: str) -> None:
    (tools_dir / f"{tool_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "id": tool_id,
                "name": tool_id.replace("_", " ").title(),
                "category": category,
                "description": f"test tool {tool_id}",
            },
            sort_keys=False,
        )
    )


def _write_examples(examples_dir: Path) -> None:
    """Two tools, two examples each — enough to exercise filter + k-limit."""
    data = {
        "examples": [
            {
                "id": "ex_alpha_scene",
                "tool_ids": ["tool_alpha"],
                "source": "original",
                "scale": "scene",
                "snippet": "Alpha scene snippet body.",
                "annotation": "alpha scene annotation",
            },
            {
                "id": "ex_alpha_chapter",
                "tool_ids": ["tool_alpha"],
                "source": "original",
                "scale": "chapter",
                "snippet": "Alpha chapter snippet body.",
                "annotation": "alpha chapter annotation",
            },
            {
                "id": "ex_beta_scene",
                "tool_ids": ["tool_beta"],
                "source": "original",
                "scale": "scene",
                "snippet": "Beta scene snippet body.",
                "annotation": "beta scene annotation",
            },
            {
                "id": "ex_beta_chapter",
                "tool_ids": ["tool_beta"],
                "source": "original",
                "scale": "chapter",
                "snippet": "Beta chapter snippet body.",
                "annotation": "beta chapter annotation",
            },
        ]
    }
    (examples_dir / "examples.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.fixture()
def library(tmp_path: Path) -> CraftLibrary:
    # CraftLibrary loads ``<root>/{structures,tools,styles,examples}``. We
    # only populate the subdirs we need; missing dirs are silently empty.
    root = tmp_path / "craft"
    (root / "tools").mkdir(parents=True)
    (root / "examples").mkdir(parents=True)
    _write_tool(root / "tools", "tool_alpha", "tension")
    _write_tool(root / "tools", "tool_beta", "reversal")
    _write_examples(root / "examples")
    return CraftLibrary(root)


@pytest.fixture()
def retriever(library: CraftLibrary) -> CraftRetriever:
    return CraftRetriever(library)


# -- Tests ----------------------------------------------------------------


def test_library_has_expected_examples(library: CraftLibrary):
    assert len(library.examples_for_tool("tool_alpha")) == 2
    assert len(library.examples_for_tool("tool_beta")) == 2


async def test_tool_id_filter_returns_only_that_tools_examples(
    retriever: CraftRetriever,
):
    q = Query(filters=QueryFilters(tool_id="tool_alpha").to_dict())
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 2
    for r in results:
        assert r.metadata["tool_id"] == "tool_alpha"
        assert r.source_id.startswith("craft/tool_alpha/")
        assert "tool_alpha" in r.metadata["tool_ids"]
    assert {r.metadata["example_id"] for r in results} == {
        "ex_alpha_scene",
        "ex_alpha_chapter",
    }


async def test_missing_tool_id_returns_empty(retriever: CraftRetriever):
    # ``Query.filters`` defaults to {} — no tool_id key at all.
    results = await retriever.retrieve(Query(), k=10)
    assert results == []


async def test_unknown_tool_id_returns_empty(retriever: CraftRetriever):
    q = Query(filters={"tool_id": "tool_does_not_exist"})
    results = await retriever.retrieve(q, k=10)
    assert results == []


async def test_k_limits_results(retriever: CraftRetriever):
    q = Query(filters=QueryFilters(tool_id="tool_alpha").to_dict())
    results = await retriever.retrieve(q, k=1)
    assert len(results) == 1
    assert results[0].metadata["tool_id"] == "tool_alpha"


async def test_metadata_includes_tool_id(retriever: CraftRetriever):
    q = Query(filters=QueryFilters(tool_id="tool_beta").to_dict())
    results = await retriever.retrieve(q, k=10)
    assert results
    for r in results:
        assert "tool_id" in r.metadata
        assert r.metadata["tool_id"] == "tool_beta"


async def test_result_text_is_example_snippet(retriever: CraftRetriever):
    q = Query(filters=QueryFilters(tool_id="tool_alpha", scale="scene").to_dict())
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 1
    assert results[0].text == "Alpha scene snippet body."
    assert results[0].metadata["example_id"] == "ex_alpha_scene"
    assert results[0].metadata["scale"] == "scene"


async def test_scale_filter_narrows(retriever: CraftRetriever):
    q = Query(filters=QueryFilters(tool_id="tool_beta", scale="chapter").to_dict())
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 1
    assert results[0].metadata["example_id"] == "ex_beta_chapter"
    assert results[0].metadata["scale"] == "chapter"


async def test_pov_and_register_filters_are_noop_on_current_schema(
    retriever: CraftRetriever,
):
    # ``Example`` has no ``pov`` / ``register`` fields — these filters are
    # accepted for forward-compat but ignored today. The spec explicitly
    # says to use what exists and ignore the rest, so the full tool list
    # should still come back.
    q = Query(
        filters=QueryFilters(
            tool_id="tool_alpha", pov="second", register="plain"
        ).to_dict()
    )
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 2


async def test_score_is_flat_one(retriever: CraftRetriever):
    q = Query(filters=QueryFilters(tool_id="tool_alpha").to_dict())
    results = await retriever.retrieve(q, k=10)
    for r in results:
        assert r.score == 1.0


async def test_source_id_shape(retriever: CraftRetriever):
    q = Query(filters=QueryFilters(tool_id="tool_alpha").to_dict())
    results = await retriever.retrieve(q, k=10)
    assert {r.source_id for r in results} == {
        "craft/tool_alpha/ex_alpha_scene",
        "craft/tool_alpha/ex_alpha_chapter",
    }
