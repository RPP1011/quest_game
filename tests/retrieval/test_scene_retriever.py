"""Tests for ``SceneShapeRetriever`` (Wave 3c, metadata mode)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.retrieval import Query, QueryFilters, SceneShapeRetriever


# -- Fixtures -------------------------------------------------------------


def _write_manifest(path: Path) -> None:
    data = {
        "version": 1,
        "scoring": {"critic_error_weight": 0.25, "critic_warning_weight": 0.10},
        "works": [
            {
                "id": "high_tension_novel",
                "title": "High-Tension Novel",
                "author": "Anon",
                "pov": "third_omniscient",
                "is_quest": False,
                "expected": {"tension_execution": 0.8},
                "passages": [
                    {"id": "s01", "sha256": "PENDING"},
                    {"id": "s02", "sha256": "PENDING"},
                ],
            },
            {
                "id": "low_tension_novel",
                "title": "Low-Tension Novel",
                "author": "Anon",
                "pov": "first",
                "is_quest": False,
                "expected": {"tension_execution": 0.25},
                "passages": [
                    {"id": "s01", "sha256": "PENDING"},
                ],
            },
            {
                "id": "quest_alpha",
                "title": "Quest Alpha",
                "author": "Anon",
                "pov": "second",
                "is_quest": True,
                "expected": {"tension_execution": 0.75},
                "passages": [
                    {"id": "s01", "sha256": "PENDING"},
                    {"id": "s02", "sha256": "PENDING"},
                ],
            },
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_labels(labels_dir: Path) -> None:
    """Two arc-label files; one is a bare list, one is wrapped in 'passages'."""
    novels = {
        "rater": "claude",
        "model": "claude-opus-4-6",
        "passages": [
            {
                "work_id": "high_tension_novel",
                "passage_id": "s01",
                "is_quest": False,
                "dimensions": {
                    "tension_execution": 0.85,
                    "scene_coherence": 0.78,
                    "emotional_arc": 0.7,
                    "thematic_development": 0.6,
                    "consequence_weight": 0.75,
                },
                "rationale": "high tension scene",
            },
            {
                "work_id": "high_tension_novel",
                "passage_id": "s02",
                "is_quest": False,
                "dimensions": {
                    "tension_execution": 0.35,
                    "scene_coherence": 0.6,
                    "emotional_arc": 0.4,
                    "thematic_development": 0.5,
                    "consequence_weight": 0.4,
                },
                # Label a dramatic_function on this one so we can test
                # the dramatic_function filter.
                "dramatic_function": "quiet_reflection",
                "rationale": "interlude",
            },
            {
                "work_id": "low_tension_novel",
                "passage_id": "s01",
                "is_quest": False,
                "dimensions": {
                    "tension_execution": 0.2,
                    "scene_coherence": 0.5,
                    "emotional_arc": 0.3,
                    "thematic_development": 0.4,
                    "consequence_weight": 0.25,
                },
                "rationale": "ambling scene",
            },
        ],
    }
    quest = [
        {
            "work_id": "quest_alpha",
            "passage_id": "s01",
            "is_quest": True,
            "dimensions": {
                "tension_execution": 0.78,
                "scene_coherence": 0.82,
                "emotional_arc": 0.75,
                "thematic_development": 0.6,
                "consequence_weight": 0.7,
                "choice_hook_quality": 0.8,
            },
            "dramatic_function": "escalation",
            "rationale": "escalation scene",
        },
        {
            "work_id": "quest_alpha",
            "passage_id": "s02",
            "is_quest": True,
            "dimensions": {
                "tension_execution": 0.9,
                "scene_coherence": 0.7,
                "emotional_arc": 0.85,
                "thematic_development": 0.5,
                "consequence_weight": 0.88,
            },
            "rationale": "climax scene",
        },
    ]
    (labels_dir / "labels_claude_arc_novels.json").write_text(json.dumps(novels))
    (labels_dir / "labels_claude_arc_quest.json").write_text(json.dumps(quest))


def _write_scenes(scenes_dir: Path) -> None:
    spec = {
        ("high_tension_novel", "s01"): (
            "The explosion tore through the hall; she dragged him by the collar "
            "toward the only door that still stood. Heat chased them. The world "
            "narrowed to the door, the door, the door."
        ),
        ("high_tension_novel", "s02"): (
            "Afterward, in the garden, neither spoke. The fountain filled the "
            "silence for them. She watched a bee crawl the rim of a cup."
        ),
        ("low_tension_novel", "s01"): (
            "We walked along the river for a time. There is no better way to "
            "spend an afternoon than walking, my aunt used to say, especially "
            "in June."
        ),
        ("quest_alpha", "s01"): (
            "You hear the second set of footsteps before you see them. Two of "
            "them. Behind, now. You count the coins in your pocket without "
            "looking — three, and the iron nail."
        ),
        ("quest_alpha", "s02"): (
            "The bridge gives beneath you. You catch the rope. Below, the river "
            "is a dark, wide mouth. The rope is fraying at the knot. You have "
            "a heartbeat to choose."
        ),
    }
    for (work_id, scene_id), body in spec.items():
        wdir = scenes_dir / work_id
        wdir.mkdir(parents=True, exist_ok=True)
        # Exercise the frontmatter stripping path on one scene.
        if (work_id, scene_id) == ("quest_alpha", "s01"):
            content = f"---\nwork: {work_id}\nid: {scene_id}\n---\n{body}\n"
        else:
            content = body + "\n"
        (wdir / f"{scene_id}.txt").write_text(content, encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, str, Path]:
    manifest = tmp_path / "scenes_manifest.yaml"
    labels = tmp_path / "labels"
    scenes = tmp_path / "scenes"
    labels.mkdir()
    scenes.mkdir()
    _write_manifest(manifest)
    _write_labels(labels)
    _write_scenes(scenes)
    return manifest, str(labels / "labels_claude_arc_*.json"), scenes


@pytest.fixture()
def retriever(corpus: tuple[Path, str, Path]) -> SceneShapeRetriever:
    manifest, labels_glob, scenes = corpus
    return SceneShapeRetriever(
        scenes_manifest_path=manifest,
        arc_labels_glob=labels_glob,
        scenes_dir=scenes,
        enable_semantic=False,
    )


# -- Tests ----------------------------------------------------------------


def test_index_includes_all_labeled_scenes(retriever: SceneShapeRetriever):
    # 2 from high_tension_novel + 1 from low_tension_novel + 2 from quest_alpha = 5
    assert retriever.index_size == 5


async def test_filter_pov_returns_only_matching(retriever: SceneShapeRetriever):
    q = Query(filters=QueryFilters(pov="second").to_dict())
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 2
    assert {r.metadata["work_id"] for r in results} == {"quest_alpha"}
    for r in results:
        assert r.metadata["pov"] == "second"
        assert r.metadata["scale"] == "scene"


async def test_score_range_narrows_results(retriever: SceneShapeRetriever):
    q = Query(filters=QueryFilters(
        score_ranges={"tension_execution": (0.7, 1.0)},
    ).to_dict())
    results = await retriever.retrieve(q, k=10)
    keys = {(r.metadata["work_id"], r.metadata["scene_id"]) for r in results}
    # high_tension_novel/s01 (0.85), quest_alpha/s01 (0.78), quest_alpha/s02 (0.90)
    # should all clear. low_tension_novel/s01 (0.2) and high_tension_novel/s02
    # (0.35) should drop.
    assert keys == {
        ("high_tension_novel", "s01"),
        ("quest_alpha", "s01"),
        ("quest_alpha", "s02"),
    }
    for r in results:
        assert 0.7 <= r.metadata["actual_scores"]["tension_execution"] <= 1.0


async def test_dramatic_function_filter(retriever: SceneShapeRetriever):
    q = Query(filters={"dramatic_function": "escalation"})
    results = await retriever.retrieve(q, k=10)
    assert len(results) == 1
    assert (results[0].metadata["work_id"], results[0].metadata["scene_id"]) == (
        "quest_alpha", "s01",
    )
    assert results[0].metadata["dramatic_function"] == "escalation"


async def test_exclude_works_drops_them(retriever: SceneShapeRetriever):
    q = Query(filters=QueryFilters(
        exclude_works={"high_tension_novel", "low_tension_novel"},
    ).to_dict())
    results = await retriever.retrieve(q, k=10)
    assert {r.metadata["work_id"] for r in results} == {"quest_alpha"}


async def test_is_quest_filter(retriever: SceneShapeRetriever):
    q = Query(filters=QueryFilters(is_quest=True).to_dict())
    results = await retriever.retrieve(q, k=10)
    assert {r.metadata["work_id"] for r in results} == {"quest_alpha"}
    assert all(r.metadata["is_quest"] for r in results)


async def test_score_midpoint_ranking(retriever: SceneShapeRetriever):
    # Range 0.7-1.0 → midpoint 0.85, half-width 0.15. rel_scaled = |v-0.85|/0.15.
    # rel>0.5 (i.e. outside the inner half of the window) → -0.2 penalty.
    # Scores of surviving scenes:
    #   high_tension_novel/s01: 0.85  → rel=0    → 1.0
    #   quest_alpha/s02:        0.90  → rel=0.33 → 1.0
    #   quest_alpha/s01:        0.78  → rel=0.47 → 1.0 (under 0.5 threshold)
    # To force a tiebreak into play, squeeze the window tighter.
    q = Query(filters=QueryFilters(
        score_ranges={"tension_execution": (0.80, 0.90)},
    ).to_dict())
    results = await retriever.retrieve(q, k=10)
    # high_tension_novel/s01 (0.85) is centered; quest_alpha/s02 (0.90) is at the
    # edge. s02 should take a penalty; s01 should rank first.
    keys = [(r.metadata["work_id"], r.metadata["scene_id"]) for r in results]
    assert keys[0] == ("high_tension_novel", "s01")
    assert results[0].score > results[-1].score


async def test_k_limits_result_count(retriever: SceneShapeRetriever):
    results = await retriever.retrieve(Query(), k=2)
    assert len(results) == 2


async def test_source_id_and_text_shape(retriever: SceneShapeRetriever):
    q = Query(filters=QueryFilters(
        score_ranges={"tension_execution": (0.83, 0.87)},
    ).to_dict())
    results = await retriever.retrieve(q, k=1)
    assert results
    r = results[0]
    assert r.source_id == "high_tension_novel/s01"
    assert "explosion" in r.text
    assert r.metadata["scale"] == "scene"


async def test_yaml_frontmatter_stripped(retriever: SceneShapeRetriever):
    q = Query(filters={"dramatic_function": "escalation"})
    results = await retriever.retrieve(q, k=1)
    assert results
    assert not results[0].text.startswith("---")
    assert results[0].text.startswith("You hear the second set")


async def test_missing_scene_text_skipped(tmp_path: Path):
    manifest = tmp_path / "scenes_manifest.yaml"
    labels_dir = tmp_path / "labels"
    scenes_dir = tmp_path / "scenes"
    labels_dir.mkdir()
    scenes_dir.mkdir()
    _write_manifest(manifest)
    _write_labels(labels_dir)
    # Only create one scene file out of the full fixture set.
    (scenes_dir / "high_tension_novel").mkdir()
    (scenes_dir / "high_tension_novel" / "s01.txt").write_text("only me")

    r = SceneShapeRetriever(
        scenes_manifest_path=manifest,
        arc_labels_glob=str(labels_dir / "labels_claude_arc_*.json"),
        scenes_dir=scenes_dir,
    )
    assert r.index_size == 1
    results = await r.retrieve(Query(), k=10)
    assert len(results) == 1
    assert (results[0].metadata["work_id"], results[0].metadata["scene_id"]) == (
        "high_tension_novel", "s01",
    )


async def test_unknown_work_dropped(tmp_path: Path):
    # Label references a work the manifest doesn't know; it must not crash.
    manifest = tmp_path / "scenes_manifest.yaml"
    labels_dir = tmp_path / "labels"
    scenes_dir = tmp_path / "scenes"
    labels_dir.mkdir()
    scenes_dir.mkdir()
    _write_manifest(manifest)
    (labels_dir / "labels_claude_arc_extra.json").write_text(json.dumps([
        {
            "work_id": "unknown_work",
            "passage_id": "s01",
            "is_quest": False,
            "dimensions": {"tension_execution": 0.9},
        },
    ]))
    _write_scenes(scenes_dir)
    # Add a stray file for unknown_work; it must still be dropped for lack of
    # manifest metadata.
    (scenes_dir / "unknown_work").mkdir()
    (scenes_dir / "unknown_work" / "s01.txt").write_text("ghost scene")

    r = SceneShapeRetriever(
        scenes_manifest_path=manifest,
        arc_labels_glob=str(labels_dir / "labels_claude_arc_*.json"),
        scenes_dir=scenes_dir,
    )
    results = await r.retrieve(Query(), k=50)
    assert all(r_.metadata["work_id"] != "unknown_work" for r_ in results)
