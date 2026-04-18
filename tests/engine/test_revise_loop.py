"""Tests for the post-write check→revise loop in the hierarchical flow.

The previous implementation gated revise on `not has_critical`, which
meant world-rule violations got flagged but committed unchanged. The
loop should now run revise on any non-trivial issues (critical OR
fixable) up to MAX_REVISE_ATTEMPTS, and only mark `flagged_qm` if
critical issues remain *after* the revise budget is exhausted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.world import Entity, EntityType, StateDelta, WorldStateManager
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


_DRAMATIC_JSON = json.dumps({
    "action_resolution": {"kind": "success", "narrative": "Done."},
    "scenes": [
        {
            "scene_id": 1, "dramatic_question": "Q?", "outcome": "yes.",
            "beats": ["beat1", "beat2"],
            "dramatic_function": "escalation",
        }
    ],
    "ending_hook": "next.",
    "suggested_choices": [{"title": "go on", "description": "", "tags": []}],
})

_EMOTIONAL_JSON = json.dumps({
    "scenes": [
        {
            "scene_id": 1, "primary_emotion": "tension", "intensity": 0.5,
            "entry_state": "wary", "exit_state": "resolved",
            "transition_type": "escalation", "emotional_source": "the door",
        }
    ],
    "update_emotional_arc": "tension → resolve",
    "contrast_strategy": "quiet before storm",
})

_CRAFT_JSON = json.dumps({
    "scenes": [{
        "scene_id": 1, "temporal": {"description": "linear"},
        "register": {
            "sentence_variance": "medium", "concrete_abstract_ratio": 0.6,
            "interiority_depth": "medium", "sensory_density": "moderate",
            "dialogue_ratio": 0.3, "pace": "measured",
        },
    }],
    "briefs": [{"scene_id": 1, "brief": "Write."}],
})

_PROSE_BAD = "She physically touched the goddess. The wood beneath them creaked."
_PROSE_FIXED = "Warmth bloomed where the goddess passed near. The wood beneath her creaked."

_CHECK_CRITICAL = json.dumps({
    "issues": [
        {
            "severity": "critical", "category": "world_rule",
            "message": "Goddess described touching matter, violates rule",
            "suggested_fix": "Render her presence as warmth or pressure, not touch.",
        }
    ]
})

_CHECK_CLEAN = json.dumps({"issues": []})

# Both the metaphor-critic pass and the typed-edit pass call client.chat().
# Tests must include one response per call so the scripted client stays in sync.
_METAPHOR_CRITIC_EMPTY = '{"families": {}, "total_figurative": 0, "dominant_family": null, "dominant_percentage": 0}'
_TYPED_EDIT_EMPTY = '{"edits": []}'

_EMPTY_EXTRACT = json.dumps({
    "entity_updates": [], "new_relationships": [],
    "removed_relationships": [], "timeline_events": [],
    "foreshadowing_updates": [],
})


class ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.log: list[str] = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        r = self._responses.pop(0)
        assert r["kind"] == "structured", f"Expected structured for {schema_name}, got {r}"
        self.log.append(f"structured:{schema_name}")
        return r["content"]

    async def chat(self, *, messages, **kw):
        r = self._responses.pop(0)
        assert r["kind"] == "chat", f"Expected chat, got {r}"
        self.log.append("chat")
        return r["content"]


class FakeArcPlanner:
    async def plan(self, **kw):
        from app.planning.schemas import ArcDirective
        return ArcDirective(
            current_phase="setup", phase_assessment="ok",
            theme_priorities=[], plot_objectives=[], character_arcs=[],
            tension_range=(0.3, 0.7), hooks_to_plant=[], hooks_to_pay_off=[],
            parallels_to_schedule=[],
        )


class FakeDramaticPlanner:
    def __init__(self, client):
        self._client = client

    async def plan(self, **kw):
        from app.planning.schemas import DramaticPlan
        from app.world.output_parser import OutputParser
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="DramaticPlan",
        )
        return OutputParser.parse_json(raw, schema=DramaticPlan)


class FakeEmotionalPlanner:
    def __init__(self, client):
        self._client = client

    async def plan(self, **kw):
        from app.planning.schemas import EmotionalPlan
        from app.world.output_parser import OutputParser
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="EmotionalPlan",
        )
        return OutputParser.parse_json(raw, schema=EmotionalPlan)


class FakeCraftPlanner:
    def __init__(self, client):
        self._client = client

    async def plan(self, **kw):
        from app.planning.schemas import CraftPlan
        from app.world.output_parser import OutputParser
        raw = await self._client.chat_structured(
            messages=[], json_schema={}, schema_name="CraftPlan",
        )
        return OutputParser.parse_json(raw, schema=CraftPlan)


class FakeCraftLibrary:
    def tools(self, category=None): return []
    def recommend_tools(self, *a, **kw): return []
    def examples_for_tool(self, tool_id): return []
    def style(self, sid): return None


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(
            id="hero", entity_type=EntityType.CHARACTER, name="Hero",
        )),
    ]), update_number=1)
    yield sm
    conn.close()


def _cb(world):
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


def _make_pipeline(world, client):
    return Pipeline(
        world, _cb(world), client,
        arc_planner=FakeArcPlanner(),
        dramatic_planner=FakeDramaticPlanner(client),
        emotional_planner=FakeEmotionalPlanner(client),
        craft_planner=FakeCraftPlanner(client),
        craft_library=FakeCraftLibrary(),
    )


@pytest.mark.asyncio
async def test_critical_issue_triggers_revise_then_clears(world):
    """Critical issue → revise → recheck → clean → committed."""
    client = ScriptedClient([
        {"kind": "structured", "content": _DRAMATIC_JSON},
        {"kind": "structured", "content": _EMOTIONAL_JSON},
        {"kind": "structured", "content": _CRAFT_JSON},
        {"kind": "chat", "content": _PROSE_BAD},
        {"kind": "chat", "content": _PROSE_BAD},  # 2 beats
        {"kind": "structured", "content": _CHECK_CRITICAL},  # initial check finds critical
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm
        {"kind": "chat", "content": _PROSE_FIXED},  # revise output
        {"kind": "structured", "content": _CHECK_CLEAN},   # recheck clean
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm
        {"kind": "chat", "content": _TYPED_EDIT_EMPTY},    # detect_edits
        {"kind": "structured", "content": _EMPTY_EXTRACT},
    ])
    pipeline = _make_pipeline(world, client)
    out = await pipeline.run(player_action="x", update_number=2)
    assert out.trace.outcome == "committed"
    stage_names = [s.stage_name for s in out.trace.stages]
    # check → revise → check sequence present
    check_idxs = [i for i, n in enumerate(stage_names) if n == "check"]
    revise_idxs = [i for i, n in enumerate(stage_names) if n == "revise"]
    assert len(check_idxs) >= 2, f"expected ≥2 check stages, got {stage_names}"
    assert len(revise_idxs) >= 1
    assert check_idxs[0] < revise_idxs[0] < check_idxs[1]
    # Final committed prose is the fixed version
    assert _PROSE_FIXED in out.prose


@pytest.mark.asyncio
async def test_critical_persists_through_two_revises_then_flags(world):
    """All MAX_REVISE_ATTEMPTS (4) fail to clear critical → outcome=flagged_qm."""
    client = ScriptedClient([
        {"kind": "structured", "content": _DRAMATIC_JSON},
        {"kind": "structured", "content": _EMOTIONAL_JSON},
        {"kind": "structured", "content": _CRAFT_JSON},
        {"kind": "chat", "content": _PROSE_BAD},
        {"kind": "chat", "content": _PROSE_BAD},
        {"kind": "structured", "content": _CHECK_CRITICAL},  # check 1
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm after check 1
        {"kind": "chat", "content": _PROSE_BAD},             # revise 1 (still bad)
        {"kind": "structured", "content": _CHECK_CRITICAL},  # check 2
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm after check 2
        {"kind": "chat", "content": _PROSE_BAD},             # revise 2 (still bad)
        {"kind": "structured", "content": _CHECK_CRITICAL},  # check 3
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm after check 3
        {"kind": "chat", "content": _PROSE_BAD},             # revise 3 (still bad)
        {"kind": "structured", "content": _CHECK_CRITICAL},  # check 4
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm after check 4
        {"kind": "chat", "content": _PROSE_BAD},             # revise 4 (still bad)
        {"kind": "structured", "content": _CHECK_CRITICAL},  # check 5 (final — budget exhausted)
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm after check 5
        {"kind": "chat", "content": _TYPED_EDIT_EMPTY},    # detect_edits (best-effort)
        # No extract — gated on not has_critical
    ])
    pipeline = _make_pipeline(world, client)
    out = await pipeline.run(player_action="x", update_number=3)
    assert out.trace.outcome == "flagged_qm"
    stage_names = [s.stage_name for s in out.trace.stages]
    assert stage_names.count("revise") == 4   # MAX_REVISE_ATTEMPTS
    assert stage_names.count("check") == 5


@pytest.mark.asyncio
async def test_clean_first_check_skips_revise(world):
    """No issues → no revise stage at all."""
    client = ScriptedClient([
        {"kind": "structured", "content": _DRAMATIC_JSON},
        {"kind": "structured", "content": _EMOTIONAL_JSON},
        {"kind": "structured", "content": _CRAFT_JSON},
        {"kind": "chat", "content": _PROSE_BAD},
        {"kind": "chat", "content": _PROSE_BAD},
        {"kind": "structured", "content": _CHECK_CLEAN},  # clean on first check
        {"kind": "chat", "content": _METAPHOR_CRITIC_EMPTY},  # classify_metaphors_llm
        {"kind": "chat", "content": _TYPED_EDIT_EMPTY},    # detect_edits
        {"kind": "structured", "content": _EMPTY_EXTRACT},
    ])
    pipeline = _make_pipeline(world, client)
    out = await pipeline.run(player_action="x", update_number=4)
    assert out.trace.outcome == "committed"
    stage_names = [s.stage_name for s in out.trace.stages]
    assert "revise" not in stage_names
    assert stage_names.count("check") == 1
