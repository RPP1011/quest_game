from __future__ import annotations
import pytest

from app.rollout.kb_extractor import (
    extract_entity_introductions,
    extract_hook_events,
    extract_thread_advances,
    find_entity_mentions,
    persist_chapter_kb,
)
from app.world.db import open_db
from app.world.schema import (
    Entity, EntityType, RolloutRun, StoryCandidate,
)
from app.world.state_manager import WorldStateManager


def _trace_with_extract(extract_output: dict) -> dict:
    return {
        "stages": [
            {"stage_name": "dramatic", "parsed_output": {}},
            {"stage_name": "extract", "parsed_output": extract_output},
        ],
    }


def _trace_with_dramatic(dramatic_output: dict) -> dict:
    return {"stages": [{"stage_name": "dramatic", "parsed_output": dramatic_output}]}


def test_extract_hook_events():
    trace = _trace_with_extract({
        "foreshadowing_updates": [
            {"id": "fs:1", "new_status": "planted"},
            {"id": "fs:2", "new_status": "paid_off"},
            {"id": "fs:3"},  # missing new_status — skipped
        ],
    })
    rows = extract_hook_events(trace)
    assert {"hook_id": "fs:1", "new_status": "planted"} in rows
    assert {"hook_id": "fs:2", "new_status": "paid_off"} in rows
    assert len(rows) == 2


def test_extract_entity_introductions():
    trace = _trace_with_extract({
        "entity_updates": [
            {"id": "char:abuela", "patch": {"status": "active"}},
            {"id": "char:hero", "patch": {"status": "active",
                                          "last_referenced_update": 5}},
            {"id": "char:rival", "patch": {"last_referenced_update": 5}},  # no status
            {"id": "char:dead", "patch": {"status": "deceased"}},  # not active
        ],
    })
    intro = extract_entity_introductions(trace)
    assert sorted(intro) == ["char:abuela", "char:hero"]


def test_extract_thread_advances():
    trace = _trace_with_dramatic({
        "thread_advances": [
            {"thread_id": "pt:main", "target_arc_position": "rising"},
            {"thread_id": "pt:sub", "new_arc_position": "climax"},
            {"id": "pt:other", "target": "denouement"},
        ],
    })
    adv = extract_thread_advances(trace)
    assert adv["pt:main"] == "rising"
    assert adv["pt:sub"] == "climax"
    assert adv["pt:other"] == "denouement"


def test_find_entity_mentions_word_boundary():
    entities = [
        Entity(id="char:lan", entity_type=EntityType.CHARACTER, name="Lan"),
        Entity(id="char:ju", entity_type=EntityType.CHARACTER, name="Ju"),
        Entity(id="loc:cantica", entity_type=EntityType.LOCATION, name="Cantica"),
    ]
    prose = "Lan looked at her sister Ju across the deck. Cantica was distant."
    found = find_entity_mentions(prose, entities)
    assert sorted(found) == ["char:ju", "char:lan", "loc:cantica"]


def test_find_entity_mentions_avoids_substrings():
    """\b boundary should prevent 'Lan' matching 'Lanier' or 'land'."""
    entities = [Entity(id="char:lan", entity_type=EntityType.CHARACTER, name="Lan")]
    prose = "He landed on the Lanier estate, which was lavish."
    found = find_entity_mentions(prose, entities)
    assert found == []


def test_extract_handles_missing_extract_stage():
    trace = {"stages": [{"stage_name": "dramatic", "parsed_output": {}}]}
    assert extract_hook_events(trace) == []
    assert extract_entity_introductions(trace) == []


@pytest.fixture
def sm(tmp_path):
    conn = open_db(tmp_path / "w.db")
    wsm = WorldStateManager(conn)
    wsm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=2,
    ))
    wsm.create_rollout(RolloutRun(
        id="r1", quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=2,
    ))
    yield wsm
    conn.close()


def test_persist_chapter_kb_writes_hooks_and_entities(sm):
    trace = _trace_with_extract({
        "foreshadowing_updates": [
            {"id": "fs:abuela", "new_status": "planted"},
            {"id": "fs:pistol", "new_status": "paid_off"},
        ],
        "entity_updates": [
            {"id": "char:cozme", "patch": {"status": "active"}},
        ],
    })
    summary = persist_chapter_kb(
        world=sm, quest_id="q1", rollout_id="r1",
        chapter_index=3, prose="Cozme stood there.",
        trace=trace,
        all_entities=[Entity(id="char:cozme", entity_type=EntityType.CHARACTER, name="Cozme")],
    )
    assert summary["hooks_planted"] == ["fs:abuela"]
    assert summary["hooks_paid_off"] == ["fs:pistol"]
    assert summary["entities_introduced"] == ["char:cozme"]
    assert summary["entities_mentioned"] == ["char:cozme"]

    hooks = sm.list_hook_payoffs("q1")
    by_hook = {h["hook_id"]: h for h in hooks}
    assert by_hook["fs:abuela"]["planted_at_chapter"] == 3
    assert by_hook["fs:pistol"]["paid_off_at_chapter"] == 3

    eu = sm.list_entity_usage("q1")
    assert len(eu) == 1
    assert eu[0]["entity_id"] == "char:cozme"
    assert eu[0]["introduced_at_chapter"] == 3
    assert eu[0]["mention_chapters"] == [3]


def test_persist_handles_no_trace(sm):
    """No trace = no error, empty summary."""
    summary = persist_chapter_kb(
        world=sm, quest_id="q1", rollout_id="r1",
        chapter_index=1, prose="anything", trace=None,
    )
    assert summary["hooks_planted"] == []
    assert summary["hooks_paid_off"] == []
