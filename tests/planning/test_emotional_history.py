"""Tests for emotional-trajectory history + monotony heuristic."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.engine.prompt_renderer import PromptRenderer
from app.planning.emotional_planner import EmotionalPlanner, detect_monotony
from app.planning.schemas import (
    ActionResolution,
    CharacterEmotionalState,
    DramaticPlan,
    DramaticScene,
    EmotionalPlan,
    EmotionalScenePlan,
)
from app.world.db import open_db
from app.world.schema import EmotionalBeat, Entity, EntityType
from app.world.state_manager import WorldStateManager

PROMPTS = Path(__file__).parent.parent.parent / "prompts"


def _beat(u: int, emotion: str) -> EmotionalBeat:
    return EmotionalBeat(
        quest_id="q1",
        update_number=u,
        scene_index=0,
        primary_emotion=emotion,
        intensity=0.5,
        source="t",
    )


def test_detect_monotony_triggers_when_all_same():
    beats = [_beat(1, "dread"), _beat(2, "dread"), _beat(3, "dread")]
    assert detect_monotony(beats, window=3) is True


def test_detect_monotony_false_when_mixed():
    beats = [_beat(1, "dread"), _beat(2, "grief"), _beat(3, "dread")]
    assert detect_monotony(beats, window=3) is False


def test_detect_monotony_false_when_too_few_beats():
    beats = [_beat(1, "dread"), _beat(2, "dread")]
    assert detect_monotony(beats, window=3) is False


def test_detect_monotony_uses_only_window_tail():
    beats = [_beat(1, "joy"), _beat(2, "dread"),
             _beat(3, "dread"), _beat(4, "dread")]
    # Only the last 3 are examined; those are all dread.
    assert detect_monotony(beats, window=3) is True


def test_detect_monotony_configurable_window():
    beats = [_beat(1, "dread"), _beat(2, "dread"),
             _beat(3, "dread"), _beat(4, "dread")]
    assert detect_monotony(beats, window=4) is True
    assert detect_monotony(beats, window=5) is False  # only 4 beats


_DRAMATIC = DramaticPlan(
    action_resolution=ActionResolution(kind="partial", narrative="x"),
    scenes=[DramaticScene(
        scene_id=1, dramatic_question="q", outcome="o",
        beats=["a"], dramatic_function="rising",
    )],
    update_tension_target=0.5, ending_hook="h", suggested_choices=[],
)

_VALID_PLAN = EmotionalPlan(
    scenes=[EmotionalScenePlan(
        scene_id=1, primary_emotion="dread", intensity=0.5,
        entry_state="a", exit_state="b", transition_type="escalation",
        emotional_source="s",
    )],
    update_emotional_arc="a", contrast_strategy="c",
)


class CapturingClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_user: str | None = None

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        # messages: [system, user]
        self.last_user = messages[1].content
        return self._response


@pytest.mark.asyncio
async def test_planner_injects_recent_beats_into_prompt(tmp_path):
    conn = open_db(tmp_path / "w.db")
    world = WorldStateManager(conn)
    world.create_entity(Entity(id="hero", entity_type=EntityType.CHARACTER, name="Aldric"))

    client = CapturingClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    planner = EmotionalPlanner(client, renderer)

    beats = [_beat(1, "dread"), _beat(2, "dread"), _beat(3, "dread")]
    await planner.plan(
        dramatic=_DRAMATIC,
        world=world,
        recent_prose=[],
        recent_beats=beats,
        monotony_flag=True,
    )

    assert client.last_user is not None
    assert "Recent Emotional Trajectory" in client.last_user
    assert "dread" in client.last_user
    assert "Monotony flag" in client.last_user


@pytest.mark.asyncio
async def test_planner_omits_monotony_when_flag_false(tmp_path):
    conn = open_db(tmp_path / "w.db")
    world = WorldStateManager(conn)
    world.create_entity(Entity(id="hero", entity_type=EntityType.CHARACTER, name="Aldric"))

    client = CapturingClient(_VALID_PLAN.model_dump_json())
    renderer = PromptRenderer(PROMPTS)
    planner = EmotionalPlanner(client, renderer)

    await planner.plan(
        dramatic=_DRAMATIC, world=world, recent_prose=[],
        recent_beats=[], monotony_flag=False,
    )

    assert "Monotony flag" not in (client.last_user or "")
    assert "No prior emotional beats recorded." in (client.last_user or "")
