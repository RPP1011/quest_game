"""Integration tests for the EXTRACT stage inside Pipeline.run()."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.world import Entity, EntityType, StateDelta, WorldStateManager
from app.world.delta import EntityCreate
from app.world.db import open_db
from app.world.schema import ForeshadowingHook


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Scripted client (same pattern as test_pipeline_branching.py)
# ---------------------------------------------------------------------------


class ScriptedClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.log: list[str] = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str:
        r = self._responses.pop(0)
        assert r["kind"] == "structured", f"unexpected structured call; next was {r}"
        self.log.append(f"structured:{schema_name}")
        return r["content"]

    async def chat(self, *, messages, **kw) -> str:
        r = self._responses.pop(0)
        assert r["kind"] == "chat", f"unexpected chat call; next was {r}"
        self.log.append("chat")
        return r["content"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="alice", entity_type=EntityType.CHARACTER, name="Alice")),
        EntityCreate(entity=Entity(id="bob", entity_type=EntityType.CHARACTER, name="Bob")),
    ]), update_number=1)
    sm.add_foreshadowing(ForeshadowingHook(
        id="hook:1",
        description="A mysterious shadow",
        planted_at_update=1,
        payoff_target="shadow revealed",
    ))
    yield sm
    conn.close()


def _cb(world: WorldStateManager) -> ContextBuilder:
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


_PLAN = '{"beats": ["Alice meets Bob"], "suggested_choices": ["greet", "ignore"]}'
_PROSE = "Alice smiled at Bob across the tavern."
_CHECK_CLEAN = '{"issues": []}'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_extract_applies_entity_update(world: WorldStateManager):
    extract_json = json.dumps({
        "entity_updates": [{"id": "alice", "patch": {"status": "dormant"}}],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="greet", update_number=2)
    assert out.trace.outcome == "committed"
    # Extract stage must appear
    stage_names = [s.stage_name for s in out.trace.stages]
    assert "extract" in stage_names
    # Entity update must have been applied
    alice = world.get_entity("alice")
    assert alice.status.value == "dormant"


async def test_extract_adds_relationship(world: WorldStateManager):
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [{"source_id": "alice", "target_id": "bob", "rel_type": "ally"}],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    await Pipeline(world, _cb(world), client).run(player_action="greet", update_number=2)
    rels = world.list_relationships(source_id="alice")
    assert any(r.rel_type == "ally" and r.target_id == "bob" for r in rels)


async def test_extract_adds_timeline_event(world: WorldStateManager):
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [
            {"description": "Alice smiled at Bob", "involved_entities": ["alice", "bob"]},
        ],
        "foreshadowing_updates": [],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    await Pipeline(world, _cb(world), client).run(player_action="greet", update_number=2)
    events = world.list_timeline(update_number=2)
    assert len(events) == 1
    assert events[0].description == "Alice smiled at Bob"
    assert "alice" in events[0].involved_entities


async def test_extract_with_invalid_entity_id_does_not_apply_and_records_error(
    world: WorldStateManager,
):
    extract_json = json.dumps({
        "entity_updates": [{"id": "nonexistent", "patch": {"status": "dormant"}}],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="greet", update_number=2)
    # Pipeline should still succeed (not fail the chapter)
    assert out.trace.outcome == "committed"
    # Extract stage recorded with errors
    extract_stage = next(s for s in out.trace.stages if s.stage_name == "extract")
    assert len(extract_stage.errors) >= 1
    assert any("nonexistent" in e.message for e in extract_stage.errors)
    # Entity alice should NOT be dormant (only alice and bob exist)
    alice = world.get_entity("alice")
    assert alice.status.value == "active"


async def test_extract_skipped_when_critical(world: WorldStateManager):
    """When check returns critical, extract is NOT called."""
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "bad"}]}'},
        # REPLAN
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "still bad"}]}'},
        # No extract call expected
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="greet", update_number=2)
    assert out.trace.outcome == "flagged_qm"
    stage_names = [s.stage_name for s in out.trace.stages]
    assert "extract" not in stage_names


async def test_extract_user_prompt_renders_themes_and_motives(
    world: WorldStateManager, tmp_path,
):
    """G4/G13: themes and active unconscious motives surface in the user prompt."""
    from app.planning.world_extensions import Theme
    world.add_theme("q1", Theme(
        id="t:loyalty", proposition="loyalty demands self-erasure", stance="exploring",
    ))
    # Attach an active unconscious motive to alice
    alice = world.get_entity("alice")
    world.update_entity(alice.id, {"data": {"unconscious_motives": [
        {"id": "m:1", "motive": "needs approval to feel real",
         "active_since_update": 0, "resolved_at_update": None},
    ]}})
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    out = await Pipeline(
        world, _cb(world), client, quest_id="q1",
    ).run(player_action="greet", update_number=2)
    extract_stage = next(s for s in out.trace.stages if s.stage_name == "extract")
    prompt = extract_stage.input_prompt
    assert "t:loyalty" in prompt
    assert "loyalty demands self-erasure" in prompt
    assert "exploring" in prompt
    assert "m:1" in prompt
    assert "needs approval" in prompt
    # And the new output-field guidance is advertised
    assert "theme_stance_updates" in prompt
    assert "motive_resolutions" in prompt


async def test_extract_user_prompt_without_themes_or_motives(world: WorldStateManager):
    """Backward compat: when no themes/motives provided, extract still works."""
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    # no quest_id -> themes stay empty; no motives on entities either
    out = await Pipeline(world, _cb(world), client).run(
        player_action="greet", update_number=2,
    )
    assert out.trace.outcome == "committed"
    extract_stage = next(s for s in out.trace.stages if s.stage_name == "extract")
    # Prompt still renders; the section headings exist but show "(none)"
    assert "Current themes" in extract_stage.input_prompt
    assert "Active unconscious motives" in extract_stage.input_prompt


async def test_extract_persists_theme_stance_update(world: WorldStateManager):
    """G4: scripted theme_stance_updates actually persist via update_theme_stance."""
    from app.planning.world_extensions import Theme
    world.add_theme("q1", Theme(
        id="t:loyalty", proposition="loyalty demands self-erasure", stance="exploring",
    ))
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
        "theme_stance_updates": [{"id": "t:loyalty", "new_stance": "questioning"}],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    out = await Pipeline(
        world, _cb(world), client, quest_id="q1",
    ).run(player_action="greet", update_number=2)
    assert out.trace.outcome == "committed"
    assert world.get_theme("q1", "t:loyalty").stance == "questioning"


async def test_extract_persists_motive_resolution(world: WorldStateManager):
    """G13: scripted motive_resolutions mark motive resolved_at_update in entity data."""
    alice = world.get_entity("alice")
    world.update_entity(alice.id, {"data": {"unconscious_motives": [
        {"id": "m:1", "motive": "needs approval to feel real",
         "active_since_update": 0, "resolved_at_update": None},
        {"id": "m:2", "motive": "fears being seen",
         "active_since_update": 0, "resolved_at_update": None},
    ]}})
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [],
        "motive_resolutions": [
            {"character_id": "alice", "motive_id": "m:1", "resolved_at_update": 2},
        ],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    out = await Pipeline(
        world, _cb(world), client, quest_id="q1",
    ).run(player_action="greet", update_number=2)
    assert out.trace.outcome == "committed"
    alice = world.get_entity("alice")
    motives = alice.data.get("unconscious_motives", [])
    by_id = {m["id"]: m for m in motives}
    assert by_id["m:1"]["resolved_at_update"] == 2
    assert by_id["m:2"]["resolved_at_update"] is None


async def test_extract_updates_foreshadowing(world: WorldStateManager):
    extract_json = json.dumps({
        "entity_updates": [],
        "new_relationships": [],
        "removed_relationships": [],
        "timeline_events": [],
        "foreshadowing_updates": [
            {"id": "hook:1", "new_status": "referenced", "add_reference": 2},
        ],
    })
    client = ScriptedClient([
        {"kind": "structured", "content": _PLAN},
        {"kind": "chat", "content": _PROSE},
        {"kind": "structured", "content": _CHECK_CLEAN},
        {"kind": "structured", "content": extract_json},
    ])
    await Pipeline(world, _cb(world), client).run(player_action="greet", update_number=2)
    hook = world.get_foreshadowing("hook:1")
    assert hook.status.value == "referenced"
    assert 2 in hook.references
