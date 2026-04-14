"""Day 7 tests for :class:`app.optimization.ExampleCurator`.

Mining is a pure DB read; the writers produce well-formed YAML that
``CraftLibrary`` can deserialize, or sidecar anti-pattern files that are
independent of the ``Example`` schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.craft.schemas import Example
from app.optimization import ExampleCandidate, ExampleCurator


# ---------------------------------------------------------------------------
# mining
# ---------------------------------------------------------------------------


def test_mine_top_examples_returns_sorted_desc(seeded_world, tmp_path: Path):
    cur = ExampleCurator(
        seeded_world,
        craft_examples_dir=tmp_path / "examples",
        anti_patterns_dir=tmp_path / "anti",
    )
    top = cur.mine_top_examples(
        "free_indirect_quality", k=3, quest_id="qA", min_score=0.5,
    )
    assert top, "expected at least one top example"
    scores = [c.score for c in top]
    assert scores == sorted(scores, reverse=True)
    # All above the 0.5 threshold we set.
    assert all(s >= 0.5 for s in scores)


def test_mine_bottom_examples_returns_sorted_asc(seeded_world, tmp_path: Path):
    cur = ExampleCurator(
        seeded_world,
        craft_examples_dir=tmp_path / "examples",
        anti_patterns_dir=tmp_path / "anti",
    )
    bottom = cur.mine_bottom_examples(
        "free_indirect_quality", k=3, quest_id="qA", max_score=0.5,
    )
    assert bottom
    scores = [c.score for c in bottom]
    assert scores == sorted(scores)
    assert all(s <= 0.5 for s in scores)


def test_mine_top_examples_respects_min_score(seeded_world, tmp_path: Path):
    cur = ExampleCurator(
        seeded_world,
        craft_examples_dir=tmp_path / "examples",
        anti_patterns_dir=tmp_path / "anti",
    )
    # No example has free_indirect_quality >= 0.99 in the fixture.
    result = cur.mine_top_examples(
        "free_indirect_quality", k=5, quest_id="qA", min_score=0.99,
    )
    assert result == []


def test_example_candidate_stable_id_is_deterministic():
    c = ExampleCandidate(
        dimension="free_indirect_quality",
        score=0.3,
        snippet="x",
        scorecard_id=42,
        quest_id="qA",
        update_number=1,
        trace_id="tr-1",
    )
    c2 = ExampleCandidate(
        dimension="free_indirect_quality",
        score=0.9,  # different score shouldn't change id
        snippet="y",
        scorecard_id=42,
        quest_id="qA",
        update_number=1,
        trace_id="tr-1",
    )
    assert c.stable_id == c2.stable_id


# ---------------------------------------------------------------------------
# writers — craft library
# ---------------------------------------------------------------------------


def test_update_craft_library_writes_yaml_compatible_with_example_schema(
    seeded_world, tmp_path: Path,
):
    out = tmp_path / "examples"
    cur = ExampleCurator(seeded_world, craft_examples_dir=out,
                         anti_patterns_dir=tmp_path / "anti")
    top = cur.mine_top_examples(
        "free_indirect_quality", k=3, quest_id="qA", min_score=0.5,
    )
    assert top
    path = cur.update_craft_library(
        top,
        dim_to_tool_id_mapping={"free_indirect_quality": ["chekhovs_gun"]},
    )
    assert path is not None
    data = yaml.safe_load(path.read_text())
    assert "examples" in data
    assert data["examples"]
    # Every record deserializes as a valid Example.
    for item in data["examples"]:
        Example.model_validate(item)


def test_update_craft_library_skips_dims_without_mapping(
    seeded_world, tmp_path: Path,
):
    cur = ExampleCurator(seeded_world, craft_examples_dir=tmp_path / "examples",
                         anti_patterns_dir=tmp_path / "anti")
    top = cur.mine_top_examples(
        "free_indirect_quality", k=3, quest_id="qA", min_score=0.5,
    )
    # No mapping for this dim -> nothing written -> None returned.
    path = cur.update_craft_library(top, dim_to_tool_id_mapping={})
    assert path is None


def test_update_craft_library_is_idempotent_on_repeated_calls(
    seeded_world, tmp_path: Path,
):
    cur = ExampleCurator(seeded_world, craft_examples_dir=tmp_path / "examples",
                         anti_patterns_dir=tmp_path / "anti")
    top = cur.mine_top_examples(
        "free_indirect_quality", k=3, quest_id="qA", min_score=0.5,
    )
    path = cur.update_craft_library(
        top, dim_to_tool_id_mapping={"free_indirect_quality": ["chekhovs_gun"]},
    )
    first = yaml.safe_load(Path(path).read_text())
    path2 = cur.update_craft_library(
        top, dim_to_tool_id_mapping={"free_indirect_quality": ["chekhovs_gun"]},
    )
    second = yaml.safe_load(Path(path2).read_text())
    # Same number of records before and after the second call -- no dupes.
    assert len(first["examples"]) == len(second["examples"])


# ---------------------------------------------------------------------------
# writers — anti-patterns
# ---------------------------------------------------------------------------


def test_update_anti_patterns_writes_yaml_and_meta(
    seeded_world, tmp_path: Path,
):
    anti_dir = tmp_path / "anti"
    cur = ExampleCurator(seeded_world,
                         craft_examples_dir=tmp_path / "examples",
                         anti_patterns_dir=anti_dir)
    bottom = cur.mine_bottom_examples(
        "free_indirect_quality", k=3, quest_id="qA", max_score=0.5,
    )
    assert bottom
    written = cur.update_anti_patterns(bottom)
    assert any(p.suffix == ".yaml" for p in written)
    assert any(p.suffix == ".json" for p in written)

    yaml_path = anti_dir / "free_indirect_quality.yaml"
    meta_path = anti_dir / "free_indirect_quality.meta.json"
    assert yaml_path.is_file()
    assert meta_path.is_file()

    data = yaml.safe_load(yaml_path.read_text())
    assert "anti_patterns" in data
    assert data["anti_patterns"]
    for item in data["anti_patterns"]:
        assert {"id", "dimension", "score", "snippet"} <= set(item.keys())

    meta = json.loads(meta_path.read_text())
    assert meta
    # Every yaml entry has a matching metadata record.
    for item in data["anti_patterns"]:
        assert item["id"] in meta
        assert "reason" in meta[item["id"]]


def test_update_anti_patterns_uses_reason_provider(
    seeded_world, tmp_path: Path,
):
    cur = ExampleCurator(
        seeded_world,
        craft_examples_dir=tmp_path / "examples",
        anti_patterns_dir=tmp_path / "anti",
    )
    bottom = cur.mine_bottom_examples(
        "free_indirect_quality", k=1, quest_id="qA", max_score=0.5,
    )
    def reason(c):
        return f"custom-reason-{c.scorecard_id}"
    cur.update_anti_patterns(bottom, reason_provider=reason)
    meta_path = tmp_path / "anti" / "free_indirect_quality.meta.json"
    meta = json.loads(meta_path.read_text())
    assert any(v["reason"].startswith("custom-reason-") for v in meta.values())


def test_update_anti_patterns_creates_directory(seeded_world, tmp_path: Path):
    nested = tmp_path / "never_existed" / "anti"
    cur = ExampleCurator(
        seeded_world,
        craft_examples_dir=tmp_path / "examples",
        anti_patterns_dir=nested,
    )
    bottom = cur.mine_bottom_examples(
        "free_indirect_quality", k=1, quest_id="qA", max_score=0.5,
    )
    assert bottom
    written = cur.update_anti_patterns(bottom)
    assert nested.is_dir()
    assert all(p.exists() for p in written)
