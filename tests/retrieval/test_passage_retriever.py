"""Tests for ``PassageRetriever`` (Wave 1b, metadata-only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.retrieval import PassageRetriever, Query, QueryFilters


# -- Fixtures -------------------------------------------------------------


def _write_manifest(path: Path) -> None:
    data = {
        "version": 1,
        "scoring": {"critic_error_weight": 0.25, "critic_warning_weight": 0.10},
        "works": [
            {
                "id": "novel_a",
                "title": "Novel A",
                "author": "Anon",
                "year": 1920,
                "pov": "third_limited",
                "is_quest": False,
                "expected": {
                    "voice_distinctiveness": 0.9,
                    "subtext_presence": 0.8,
                    "clarity": 0.6,
                },
                "passages": [
                    {"id": "p01", "sha256": "PENDING",
                     "expected_high": ["voice_distinctiveness"],
                     "expected_low": ["clarity"]},
                    {"id": "p02", "sha256": "PENDING",
                     "expected_high": ["subtext_presence"],
                     "expected_low": ["clarity"]},
                ],
            },
            {
                "id": "novel_b",
                "title": "Novel B",
                "author": "Anon",
                "year": 1960,
                "pov": "first",
                "is_quest": False,
                "expected": {
                    "voice_distinctiveness": 0.4,
                    "subtext_presence": 0.3,
                    "clarity": 0.95,
                },
                "passages": [
                    {"id": "p01", "sha256": "PENDING",
                     "expected_high": ["clarity"],
                     "expected_low": ["subtext_presence"]},
                ],
            },
            {
                "id": "quest_c",
                "title": "Quest C",
                "author": "Anon",
                "year": 2020,
                "pov": "second",
                "is_quest": True,
                "expected": {
                    "voice_distinctiveness": 0.7,
                    "subtext_presence": 0.6,
                    "clarity": 0.8,
                },
                "passages": [
                    {"id": "p01", "sha256": "PENDING",
                     "expected_high": ["voice_distinctiveness"],
                     "expected_low": ["subtext_presence"]},
                    {"id": "p02", "sha256": "PENDING",
                     "expected_high": ["clarity"],
                     "expected_low": ["subtext_presence"]},
                ],
            },
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_labels(labels_dir: Path) -> None:
    """Emit two shards matching the ``labels_claude_part_*.json`` convention."""
    shard1 = {
        "rater": "claude",
        "model": "claude-opus-4-6",
        "passages": [
            {
                "work_id": "novel_a",
                "passage_id": "p01",
                "is_quest": False,
                "dimensions": {
                    "voice_distinctiveness": 0.95,
                    "subtext_presence": 0.85,
                    "clarity": 0.55,
                },
                "rationale": "novel_a p01",
            },
            {
                "work_id": "novel_a",
                "passage_id": "p02",
                "is_quest": False,
                "dimensions": {
                    "voice_distinctiveness": 0.80,
                    "subtext_presence": 0.90,
                    "clarity": 0.60,
                },
                "rationale": "novel_a p02",
            },
        ],
    }
    shard2 = {
        "passages": [
            {
                "work_id": "novel_b",
                "passage_id": "p01",
                "is_quest": False,
                "dimensions": {
                    "voice_distinctiveness": 0.35,
                    "subtext_presence": 0.25,
                    "clarity": 0.98,
                },
                "rationale": "novel_b p01",
            },
            {
                "work_id": "quest_c",
                "passage_id": "p01",
                "is_quest": True,
                "dimensions": {
                    "voice_distinctiveness": 0.72,
                    "subtext_presence": 0.55,
                    "clarity": 0.82,
                },
                "rationale": "quest_c p01",
            },
            {
                "work_id": "quest_c",
                "passage_id": "p02",
                "is_quest": True,
                "dimensions": {
                    "voice_distinctiveness": 0.65,
                    "subtext_presence": 0.60,
                    "clarity": 0.88,
                },
                "rationale": "quest_c p02",
            },
        ],
    }
    (labels_dir / "labels_claude_part_alpha.json").write_text(json.dumps(shard1))
    (labels_dir / "labels_claude_part_beta.json").write_text(json.dumps(shard2))


def _write_passages(passages_dir: Path) -> None:
    spec = {
        ("novel_a", "p01"): "She set the kettle down as if it weighed more than iron.",
        ("novel_a", "p02"): "The letter sat on the mantel, unopened for a week.",
        ("novel_b", "p01"): "I went to the bar. It was quiet. I ordered a whiskey.",
        ("quest_c", "p01"): "You push open the door; the hinges protest, and so do you.",
        ("quest_c", "p02"): "You count the coins again. The number has not changed.",
    }
    for (work_id, passage_id), body in spec.items():
        wdir = passages_dir / work_id
        wdir.mkdir(parents=True, exist_ok=True)
        # Use YAML frontmatter on one passage to exercise the stripping path.
        if (work_id, passage_id) == ("novel_a", "p01"):
            content = f"---\nwork: {work_id}\nid: {passage_id}\n---\n{body}\n"
        else:
            content = body + "\n"
        (wdir / f"{passage_id}.txt").write_text(content, encoding="utf-8")


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


@pytest.fixture()
def retriever(corpus: tuple[Path, Path, Path]) -> PassageRetriever:
    manifest, labels, passages = corpus
    return PassageRetriever(manifest, labels, passages, enable_semantic=False)


# -- Tests ----------------------------------------------------------------


def test_index_includes_all_labeled_passages(retriever: PassageRetriever):
    # 2 from novel_a + 1 from novel_b + 2 from quest_c = 5
    assert retriever.index_size == 5


async def test_filter_pov_returns_only_matching(retriever: PassageRetriever):
    q = Query(filters=QueryFilters(pov="second"))
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 2
    assert all(r.metadata["pov"] == "second" for r in results)
    assert {r.work_id for r in results} == {"quest_c"}


async def test_score_range_narrows_results(retriever: PassageRetriever):
    # Only novel_a's passages (0.80 / 0.95) and quest_c p01 (0.72) should clear
    # a [0.7, 1.0] voice_distinctiveness window. novel_b (0.35) and quest_c p02
    # (0.65) should drop.
    q = Query(filters=QueryFilters(
        score_ranges={"voice_distinctiveness": (0.7, 1.0)},
    ))
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 3
    keys = {(r.work_id, r.passage_id) for r in results}
    assert keys == {
        ("novel_a", "p01"),
        ("novel_a", "p02"),
        ("quest_c", "p01"),
    }
    for r in results:
        assert 0.7 <= r.metadata["actual_scores"]["voice_distinctiveness"] <= 1.0


async def test_exclude_works_drops_them(retriever: PassageRetriever):
    q = Query(filters=QueryFilters(exclude_works={"novel_a", "novel_b"}))
    results = await retriever.retrieve(q, k=10)
    assert {r.work_id for r in results} == {"quest_c"}


async def test_k_limits_result_count(retriever: PassageRetriever):
    results = await retriever.retrieve(Query(), k=2)
    assert len(results) == 2
    # Stable tie-break orders by (work_id, passage_id); novel_a/p01 then p02.
    assert (results[0].work_id, results[0].passage_id) == ("novel_a", "p01")
    assert (results[1].work_id, results[1].passage_id) == ("novel_a", "p02")


async def test_is_quest_filter(retriever: PassageRetriever):
    q = Query(filters=QueryFilters(is_quest=True))
    results = await retriever.retrieve(q, k=10)
    assert {r.work_id for r in results} == {"quest_c"}
    assert all(r.metadata["is_quest"] for r in results)


async def test_score_is_tiebreaker_by_midpoint(retriever: PassageRetriever):
    # Range midpoint 0.85 — novel_a p01 (0.95) is 0.10 off; novel_a p02 (0.80)
    # is 0.05 off. Both pass. With half-width 0.15 normalizing, p01 rel=0.67
    # (>0.5 → penalty); p02 rel=0.33 (≤0.5 → no penalty). p02 must rank first.
    q = Query(filters=QueryFilters(
        pov="third_limited",
        score_ranges={"voice_distinctiveness": (0.7, 1.0)},
    ))
    results = await retriever.retrieve(q, k=10)
    assert [(r.work_id, r.passage_id) for r in results] == [
        ("novel_a", "p02"),
        ("novel_a", "p01"),
    ]
    assert results[0].score > results[1].score


async def test_enable_semantic_raises(corpus: tuple[Path, Path, Path]):
    manifest, labels, passages = corpus
    with pytest.raises(NotImplementedError):
        PassageRetriever(manifest, labels, passages, enable_semantic=True)


async def test_yaml_frontmatter_stripped(retriever: PassageRetriever):
    q = Query(filters=QueryFilters(
        score_ranges={"voice_distinctiveness": (0.9, 1.0)},
    ))
    results = await retriever.retrieve(q, k=1)
    assert results
    # novel_a/p01 has frontmatter in the fixture; the stripped body should start
    # with the prose, not with YAML.
    assert (results[0].work_id, results[0].passage_id) == ("novel_a", "p01")
    assert not results[0].text.startswith("---")
    assert results[0].text.startswith("She set the kettle down")


async def test_missing_passage_text_skipped(tmp_path: Path):
    # Manifest + labels reference a passage whose file is absent — it should be
    # silently dropped from the index rather than crashing.
    manifest = tmp_path / "manifest.yaml"
    labels_dir = tmp_path / "labels"
    passages_dir = tmp_path / "passages"
    labels_dir.mkdir()
    passages_dir.mkdir()
    _write_manifest(manifest)
    _write_labels(labels_dir)
    # Only create one passage file.
    (passages_dir / "novel_a").mkdir()
    (passages_dir / "novel_a" / "p01.txt").write_text("only me")

    r = PassageRetriever(manifest, labels_dir, passages_dir)
    assert r.index_size == 1
    results = await r.retrieve(Query(), k=10)
    assert len(results) == 1
    assert (results[0].work_id, results[0].passage_id) == ("novel_a", "p01")


async def test_fallback_to_combined_labels(tmp_path: Path):
    # No shards present, but labels_claude_all.json is — retriever must read it.
    manifest = tmp_path / "manifest.yaml"
    labels_dir = tmp_path / "labels"
    passages_dir = tmp_path / "passages"
    labels_dir.mkdir()
    passages_dir.mkdir()
    _write_manifest(manifest)
    _write_passages(passages_dir)

    combined = {
        "passages": [
            {
                "work_id": "novel_b",
                "passage_id": "p01",
                "is_quest": False,
                "dimensions": {
                    "voice_distinctiveness": 0.35,
                    "clarity": 0.98,
                },
                "rationale": "combined",
            },
        ],
    }
    (labels_dir / "labels_claude_all.json").write_text(json.dumps(combined))

    r = PassageRetriever(manifest, labels_dir, passages_dir)
    assert r.index_size == 1
    results = await r.retrieve(Query(), k=1)
    assert (results[0].work_id, results[0].passage_id) == ("novel_b", "p01")
