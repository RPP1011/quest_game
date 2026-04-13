# tests/server/test_api.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.server import create_app


# ---------------------------------------------------------------------------
# Canned JSON responses for each schema name used by the hierarchical pipeline
# ---------------------------------------------------------------------------

_CANNED = {
    "ArcDirective": json.dumps({
        "current_phase": "act_1",
        "phase_assessment": "Early stage.",
        "theme_priorities": [],
        "plot_objectives": [],
        "character_arcs": [],
        "tension_range": [0.3, 0.7],
        "hooks_to_plant": [],
        "hooks_to_pay_off": [],
        "parallels_to_schedule": [],
    }),
    "DramaticPlan": json.dumps({
        "action_resolution": {"kind": "success", "narrative": "The action succeeds."},
        "scenes": [
            {
                "scene_id": 1,
                "dramatic_question": "Will the hero prevail?",
                "outcome": "The hero prevails.",
                "beats": ["beat1"],
                "dramatic_function": "inciting_incident",
            }
        ],
        "update_tension_target": 0.5,
        "ending_hook": "A door creaks open.",
        "suggested_choices": [{"title": "Go", "description": "Depart.", "tags": []}],
        "tools_selected": [],
        "thread_advances": [],
        "questions_opened": [],
        "questions_closed": [],
    }),
    "EmotionalPlan": json.dumps({
        "scenes": [
            {
                "scene_id": 1,
                "primary_emotion": "anticipation",
                "intensity": 0.5,
                "entry_state": "calm",
                "exit_state": "resolved",
                "transition_type": "escalation",
                "emotional_source": "player_action",
            }
        ],
        "update_emotional_arc": "rising",
        "contrast_strategy": "juxtapose calm with tension",
    }),
    "CraftPlan": json.dumps({
        "scenes": [
            {
                "scene_id": 1,
                "temporal": {"description": "linear present-scene"},
                "register": {
                    "sentence_variance": "medium",
                    "concrete_abstract_ratio": 0.6,
                    "interiority_depth": "medium",
                    "sensory_density": "moderate",
                    "dialogue_ratio": 0.3,
                    "pace": "measured",
                },
                "passage_register_overrides": [],
                "motif_instructions": [],
                "narrator_focus": [],
                "narrator_withholding": [],
                "sensory_palette": {},
                "voice_notes": [],
                "parallel_instruction": None,
                "negative_space": [],
                "opening_instruction": None,
                "closing_instruction": None,
            }
        ],
        "briefs": [{"scene_id": 1, "brief": "Write the scene with clarity."}],
    }),
    "CheckOutput": json.dumps({"issues": []}),
    "StateDelta": json.dumps({
        "entity_creates": [],
        "entity_updates": [],
        "relationship_changes": [],
        "foreshadowing_updates": [],
        "plot_thread_updates": [],
        "timeline_events": [],
    }),
    # Legacy BeatSheet schema (flat pipeline)
    "BeatSheet": json.dumps({
        "beats": ["beat1"],
        "suggested_choices": [{"title": "Go", "description": "Depart.", "tags": []}],
    }),
}


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch):
    # No real llama-server — we inject a fake client.
    from app.runtime.client import InferenceClient

    class FakeClient:
        async def chat_structured(self, messages=None, *, json_schema=None, schema_name="", **kw):
            return _CANNED.get(
                schema_name,
                '{"beats": ["beat1"], "suggested_choices": [{"title": "Go", "description": "Depart.", "tags": []}]}',
            )

        async def chat(self, messages=None, **kw):
            return "Prose."

    def _make_app(url):
        return FakeClient()

    monkeypatch.setattr("app.server._make_client", _make_app)
    return create_app(quests_dir=tmp_path / "quests", server_url="http://fake")


@pytest.fixture
def client(app_factory):
    return TestClient(app_factory)


def _seed_json():
    return {
        "entities": [{"id": "a", "entity_type": "character", "name": "Alice"}],
    }


def test_list_quests_empty(client):
    r = client.get("/api/quests")
    assert r.status_code == 200
    assert r.json() == []


def test_create_and_list_quest(client, tmp_path: Path):
    r = client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    assert r.status_code == 201, r.text
    assert r.json()["id"] == "q1"
    r = client.get("/api/quests")
    assert len(r.json()) == 1
    assert r.json()[0]["id"] == "q1"


def test_create_duplicate_rejected(client):
    client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    r = client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    assert r.status_code == 409


def test_get_quest_chapters_empty(client):
    client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    r = client.get("/api/quests/q1/chapters")
    assert r.status_code == 200
    assert r.json() == []


def test_advance_writes_chapter_and_trace(client):
    client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    r = client.post("/api/quests/q1/advance", json={"action": "look around"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Prose" in body["prose"]
    assert body["choices"] == [{"title": "Go", "description": "Depart.", "tags": []}]
    assert body["outcome"] == "committed"
    # Chapter appears in list
    r2 = client.get("/api/quests/q1/chapters")
    assert len(r2.json()) == 1
    # Trace was persisted and fetchable
    r3 = client.get(f"/api/quests/q1/traces/{body['trace_id']}")
    assert r3.status_code == 200
    assert r3.json()["outcome"] == "committed"


def test_list_traces(client):
    client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    client.post("/api/quests/q1/advance", json={"action": "x"})
    client.post("/api/quests/q1/advance", json={"action": "y"})
    r = client.get("/api/quests/q1/traces")
    assert len(r.json()) == 2


def test_404_for_unknown_quest(client):
    r = client.get("/api/quests/nope/chapters")
    assert r.status_code == 404


def test_chapter_summary_includes_choices_from_trace(client):
    client.post("/api/quests", json={"id": "q1", "seed": _seed_json()})
    r = client.post("/api/quests/q1/advance", json={"action": "look around"})
    assert r.status_code == 200, r.text
    r2 = client.get("/api/quests/q1/chapters")
    assert r2.status_code == 200
    chapters = r2.json()
    assert len(chapters) == 1
    assert chapters[0]["choices"] == [{"title": "Go", "description": "Depart.", "tags": []}]


def test_scene_endpoint_returns_current_state(tmp_path: Path, monkeypatch):
    """Scene endpoint reflects location, characters, and plot threads."""
    from app.runtime.client import InferenceClient
    from app.world.schema import ArcPosition, PlotThread

    class FakeClient:
        async def chat_structured(self, messages=None, *, json_schema=None, schema_name="", **kw):
            return _CANNED.get(schema_name, '{"beats": ["b1"], "suggested_choices": []}')

        async def chat(self, messages=None, **kw):
            return "It was a quiet evening."

    def _make_app(url):
        return FakeClient()

    monkeypatch.setattr("app.server._make_client", _make_app)
    app = create_app(quests_dir=tmp_path / "quests", server_url="http://fake")
    c = TestClient(app)

    seed = {
        "entities": [
            {"id": "tavern", "entity_type": "location", "name": "The Rusty Flagon"},
            {"id": "bob", "entity_type": "character", "name": "Bob"},
            {"id": "eva", "entity_type": "character", "name": "Eva"},
        ],
        "plot_threads": [
            {
                "id": "pt:main", "name": "The Lost Relic", "description": "Find it.",
                "arc_position": "rising", "priority": 8,
            }
        ],
    }
    c.post("/api/quests", json={"id": "sc1", "seed": seed})
    c.post("/api/quests/sc1/advance", json={"action": "look around"})

    r = c.get("/api/quests/sc1/scene")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["location"] == "The Rusty Flagon"
    assert set(body["present_characters"]) == {"Bob", "Eva"}
    assert "The Lost Relic" in body["plot_threads"]
    assert "It was a quiet evening." in body["recent_prose_tail"]


def test_create_quest_bootstraps_arc(tmp_path: Path, monkeypatch):
    """POST /api/quests bootstraps a QuestArcState and writes config.json."""
    class FakeClient:
        async def chat_structured(self, messages=None, *, json_schema=None, schema_name="", **kw):
            return _CANNED.get(schema_name, "{}")

        async def chat(self, messages=None, **kw):
            return "Prose."

    def _make_app(url):
        return FakeClient()

    monkeypatch.setattr("app.server._make_client", _make_app)
    quests_dir = tmp_path / "quests"
    app = create_app(quests_dir=quests_dir, server_url="http://fake")
    c = TestClient(app)

    seed = {
        "structure_id": "three_act",
        "genre": "fantasy",
        "premise": "A hero's journey.",
        "themes": ["redemption"],
        "protagonist": "Aria",
        "entities": [{"id": "hero", "entity_type": "character", "name": "Aria"}],
    }
    r = c.post("/api/quests", json={"id": "arc_test", "seed": seed})
    assert r.status_code == 201, r.text

    # Verify arc row was created
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    db_path = quests_dir / "arc_test" / "quest.db"
    sm = WorldStateManager(open_db(db_path))
    arc = sm.get_arc("arc_test", "main")
    assert arc.structure_id == "three_act"
    assert arc.arc_id == "main"
    assert arc.quest_id == "arc_test"
    assert arc.current_phase_index == 0

    # Verify config.json was written
    config_path = quests_dir / "arc_test" / "config.json"
    assert config_path.is_file()
    config = json.loads(config_path.read_text())
    assert config["genre"] == "fantasy"
    assert config["premise"] == "A hero's journey."
    assert config["themes"] == ["redemption"]
    assert config["protagonist"] == "Aria"
