"""Day 4: SFT collection auto-save.

When ``quest_config["sft_collection"]["enabled"]`` is True AND
``n_candidates > 1`` AND a ``quest_id`` is present, the write stage must
persist a per-scene JSON record to ``data/sft/<quest_id>/`` containing the
craft brief, every candidate's prose, its 12-dim scorer breakdown + composite
score, and the winner index. Default off: pipelines without the flag write
nothing.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.engine.trace import PipelineTrace
from app.planning.schemas import CraftBrief, CraftPlan, CraftScenePlan
from app.scoring import DIMENSION_NAMES, Scorer
from app.world import StateDelta, WorldStateManager
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class VariantClient:
    def __init__(self, prose_per_call: list[str]) -> None:
        self._prose = list(prose_per_call)
        self.calls: list[dict[str, Any]] = []

    async def chat_structured(self, **_: Any) -> str:
        raise AssertionError("chat_structured should not be called")

    async def chat(self, *, messages, **kw) -> str:
        self.calls.append({"temperature": kw.get("temperature"),
                           "seed": kw.get("seed")})
        return self._prose.pop(0)


def _make_world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(), update_number=1)
    return sm, conn


def _make_pipeline(world, client, **kw):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    return Pipeline(world, cb, client, **kw)


def _trace() -> PipelineTrace:
    return PipelineTrace(trace_id=uuid.uuid4().hex, trigger="test")


def _one_scene_plan(brief: str = "Write the protagonist entering the hall.") -> CraftPlan:
    return CraftPlan(
        scenes=[CraftScenePlan(scene_id=7)],
        briefs=[CraftBrief(scene_id=7, brief=brief)],
    )


# ---------------------------------------------------------------------------


async def test_sft_default_off_writes_nothing(tmp_path):
    """Default path: flag off ⇒ no SFT dir, no sft_collection trace stage."""
    world, conn = _make_world(tmp_path)
    try:
        sft_dir = tmp_path / "sft"
        client = VariantClient(["you walk in.", "you pause."])
        # Flag absent — default off. Also try flag explicitly False.
        pipeline = _make_pipeline(
            world, client,
            n_candidates=2,
            scorer=Scorer(),
            quest_id="q-demo",
            quest_config={"sft_collection": {"dir": str(sft_dir)}},
        )
        trace = _trace()
        await pipeline._run_write(
            trace, _one_scene_plan(),
            player_action="enter",
            update_number=3,
        )
        assert not sft_dir.exists()
        assert "sft_collection" not in {s.stage_name for s in trace.stages}
    finally:
        conn.close()


async def test_sft_flag_on_writes_per_scene_record(tmp_path):
    """Flag on + n>1 + quest_id ⇒ exactly one SFT JSON per scene, with the
    craft brief, every candidate, per-dim scorer breakdown, and winner index."""
    world, conn = _make_world(tmp_path)
    try:
        sft_dir = tmp_path / "sft"
        prose = [
            "you step into the cold hall and listen.",
            "you linger by the doorway, watching.",
            "you drift forward through the shadows.",
        ]
        client = VariantClient(prose)
        pipeline = _make_pipeline(
            world, client,
            n_candidates=3,
            scorer=Scorer(),
            quest_id="q-demo",
            quest_config={
                "sft_collection": {"enabled": True, "dir": str(sft_dir)},
            },
        )
        trace = _trace()
        brief_text = "Write the protagonist entering the hall."
        winner = await pipeline._run_write(
            trace, _one_scene_plan(brief=brief_text),
            player_action="enter",
            update_number=5,
        )

        assert winner in prose

        # Exactly one file in data/sft/q-demo/
        quest_dir = sft_dir / "q-demo"
        assert quest_dir.exists()
        files = sorted(quest_dir.glob("*.json"))
        assert len(files) == 1, f"expected 1 file, got {files}"
        fname = files[0].name
        # Name convention: u<update>_s<scene>_<trace>.json
        assert fname.startswith("u5_s7_")
        assert fname.endswith(f"_{trace.trace_id}.json")

        rec = json.loads(files[0].read_text())
        assert rec["quest_id"] == "q-demo"
        assert rec["update_number"] == 5
        assert rec["scene_index"] == 7
        assert rec["pipeline_trace_id"] == trace.trace_id
        assert rec["craft_brief"] == brief_text
        assert isinstance(rec["winner_index"], int)
        assert rec["winner_index"] in (0, 1, 2)
        assert len(rec["candidates"]) == 3
        for cand in rec["candidates"]:
            assert set(cand.keys()) >= {
                "index", "prose", "weighted_score",
                "overall_score", "dimension_scores",
            }
            assert cand["prose"] in prose
            assert 0.0 <= cand["weighted_score"]
            assert set(cand["dimension_scores"].keys()) == set(DIMENSION_NAMES)

        # Winner's prose equals the prose actually returned.
        winner_entry = next(
            c for c in rec["candidates"] if c["index"] == rec["winner_index"]
        )
        assert winner_entry["prose"] == winner

        # sft_collection trace stage recorded.
        sft_stages = [s for s in trace.stages if s.stage_name == "sft_collection"]
        assert len(sft_stages) == 1
        assert sft_stages[0].detail["path"].endswith(fname)
    finally:
        conn.close()


async def test_sft_skipped_when_quest_id_missing(tmp_path):
    """SFT records are quest-scoped; no quest_id ⇒ skip silently."""
    world, conn = _make_world(tmp_path)
    try:
        sft_dir = tmp_path / "sft"
        client = VariantClient(["you a.", "you b."])
        pipeline = _make_pipeline(
            world, client,
            n_candidates=2,
            scorer=Scorer(),
            # no quest_id
            quest_config={
                "sft_collection": {"enabled": True, "dir": str(sft_dir)},
            },
        )
        trace = _trace()
        await pipeline._run_write(
            trace, _one_scene_plan(),
            player_action="x", update_number=1,
        )
        assert not sft_dir.exists()
        assert "sft_collection" not in {s.stage_name for s in trace.stages}
    finally:
        conn.close()


async def test_sft_skipped_when_n_is_one(tmp_path):
    """N=1 ⇒ no rerank, no candidates to save, no SFT file."""
    world, conn = _make_world(tmp_path)
    try:
        sft_dir = tmp_path / "sft"
        client = VariantClient(["you stand still."])
        pipeline = _make_pipeline(
            world, client,
            n_candidates=1,
            scorer=Scorer(),
            quest_id="q-demo",
            quest_config={
                "sft_collection": {"enabled": True, "dir": str(sft_dir)},
            },
        )
        trace = _trace()
        await pipeline._run_write(
            trace, _one_scene_plan(),
            player_action="x", update_number=1,
        )
        assert not sft_dir.exists()
        assert "sft_collection" not in {s.stage_name for s in trace.stages}
    finally:
        conn.close()


async def test_sft_one_file_per_scene_on_multi_scene_plan(tmp_path):
    """A 2-scene plan with N=2 produces exactly 2 SFT files, each carrying
    the correct scene_index and the matching craft brief."""
    world, conn = _make_world(tmp_path)
    try:
        sft_dir = tmp_path / "sft"
        prose = [
            "you enter scene one, first try.",
            "you enter scene one, second try.",
            "you move to scene two, first try.",
            "you move to scene two, second try.",
        ]
        client = VariantClient(prose)
        pipeline = _make_pipeline(
            world, client,
            n_candidates=2,
            scorer=Scorer(),
            quest_id="q-demo",
            quest_config={
                "sft_collection": {"enabled": True, "dir": str(sft_dir)},
            },
        )
        trace = _trace()
        plan = CraftPlan(
            scenes=[CraftScenePlan(scene_id=1), CraftScenePlan(scene_id=2)],
            briefs=[
                CraftBrief(scene_id=1, brief="scene one brief"),
                CraftBrief(scene_id=2, brief="scene two brief"),
            ],
        )
        await pipeline._run_write(
            trace, plan, player_action="x", update_number=9,
        )

        files = sorted((sft_dir / "q-demo").glob("*.json"))
        assert len(files) == 2
        recs = [json.loads(f.read_text()) for f in files]
        by_scene = {r["scene_index"]: r for r in recs}
        assert set(by_scene) == {1, 2}
        assert by_scene[1]["craft_brief"] == "scene one brief"
        assert by_scene[2]["craft_brief"] == "scene two brief"
        for r in recs:
            assert r["update_number"] == 9
            assert r["pipeline_trace_id"] == trace.trace_id
            assert len(r["candidates"]) == 2
    finally:
        conn.close()
