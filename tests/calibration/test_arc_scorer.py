"""Tests for app/calibration/arc_scorer.py and the arc-dim prompts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.calibration.arc_scorer import (
    ARC_COMMON_DIMS,
    ARC_QUEST_DIMS,
    ArcBatchJudge,
    arc_dims_for,
    parse_response,
    score_scenes,
    strip_frontmatter,
)


PROMPTS = Path("prompts")
ALL_ARC_DIMS = list(ARC_COMMON_DIMS) + list(ARC_QUEST_DIMS)


class _StubClient:
    def __init__(self, canned: dict) -> None:
        self.canned = canned
        self.prompts: list[str] = []

    async def chat_structured(self, *, messages, json_schema, schema_name,
                              temperature, max_tokens, thinking) -> str:
        self.prompts.append(messages[0].content)
        return json.dumps(self.canned)

    async def chat(self, *, messages, **kwargs) -> str:  # pragma: no cover
        self.prompts.append(messages[0].content)
        return json.dumps(self.canned)


def test_arc_dims_gated_by_is_quest():
    assert set(arc_dims_for(False)) == set(ARC_COMMON_DIMS)
    assert set(arc_dims_for(True)) == set(ARC_COMMON_DIMS) | set(ARC_QUEST_DIMS)


@pytest.mark.parametrize("dim", ALL_ARC_DIMS)
def test_each_arc_prompt_renders(dim):
    judge = ArcBatchJudge(PROMPTS)
    rubric = judge._load_rubric(dim)
    assert dim in rubric
    # Each prompt mentions a full-scene input constraint.
    assert "scene" in rubric.lower() or "SCENE" in rubric
    # Anchors present.
    assert "HIGH" in rubric or "1.0" in rubric
    assert "LOW" in rubric or "0.0" in rubric


def test_render_prompt_quest_includes_all_arc_dims():
    judge = ArcBatchJudge(PROMPTS)
    out = judge.render_prompt(
        scene="Scene body.",
        work_id="demo",
        pov="second",
        is_quest=True,
    )
    for d in ALL_ARC_DIMS:
        assert d in out
    assert "Scene body." in out


def test_render_prompt_novel_excludes_quest_dims():
    judge = ArcBatchJudge(PROMPTS)
    out = judge.render_prompt(
        scene="x",
        work_id="demo",
        pov="first",
        is_quest=False,
    )
    for d in ARC_QUEST_DIMS:
        assert d not in out
    for d in ARC_COMMON_DIMS:
        assert d in out


def test_parse_response_happy_path():
    names = ["tension_execution"]
    raw = '{"tension_execution": {"score": 0.6, "rationale": "ok"}}'
    out = parse_response(raw, names)
    assert out["tension_execution"].score == 0.6


def test_parse_response_missing_raises():
    with pytest.raises(ValueError):
        parse_response(
            '{"tension_execution": {"score": 0.5, "rationale": "x"}}',
            ["tension_execution", "choice_hook_quality"],
        )


def test_strip_frontmatter():
    text = "---\nkey: val\n---\n\nbody text"
    assert strip_frontmatter(text) == "body text"
    assert strip_frontmatter("no fm here") == "no fm here"


async def test_arc_batch_judge_end_to_end_stub():
    judge = ArcBatchJudge(PROMPTS)
    canned = {d: {"score": 0.5, "rationale": "stub"} for d in ALL_ARC_DIMS}
    client = _StubClient(canned)
    scored = await judge.score(
        client=client,
        scene="Scene text.",
        work_id="marked_for_death",
        pov="second",
        is_quest=True,
    )
    assert set(scored) == set(ALL_ARC_DIMS)
    assert "Scene text." in client.prompts[0]


async def test_score_scenes_writes_rater_json(tmp_path: Path):
    # Build a tiny scenes dir + manifest.
    scenes_dir = tmp_path / "scenes"
    (scenes_dir / "marked_for_death").mkdir(parents=True)
    (scenes_dir / "marked_for_death" / "s01.txt").write_text(
        "---\nwork: demo\n---\n\nThis is a tiny scene body.",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "m.yaml"
    manifest_path.write_text(
        "version: 1\nscoring: {critic_error_weight: 0.25, critic_warning_weight: 0.1}\n"
        "works:\n"
        "- id: marked_for_death\n"
        "  title: Marked for Death\n"
        "  author: x\n"
        "  year: 2015\n"
        "  pov: second\n"
        "  is_quest: true\n"
        "  expected: {tension_execution: 0.75}\n"
        "  passages:\n"
        "  - {id: s01, sha256: PENDING, expected_high: [], expected_low: []}\n",
        encoding="utf-8",
    )

    canned = {d: {"score": 0.7, "rationale": "r"} for d in ALL_ARC_DIMS}
    client = _StubClient(canned)

    out_path = tmp_path / "rater_arc_stub.json"
    report = await score_scenes(
        manifest_path=manifest_path,
        scenes_dir=scenes_dir,
        client=client,
        model_tag="stub",
        out_path=out_path,
    )

    assert report["kind"] == "arc"
    assert report["model"] == "stub"
    assert len(report["scenes"]) == 1
    entry = report["scenes"][0]
    assert entry["work_id"] == "marked_for_death"
    assert entry["scene_id"] == "s01"
    assert entry["is_quest"] is True
    assert set(entry["scores"]) == set(ALL_ARC_DIMS)
    # File persisted.
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["scenes"][0]["scene_id"] == "s01"
