# tests/server/test_api.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.server import create_app


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch):
    # No real llama-server — we inject a fake client.
    from app.runtime.client import InferenceClient

    class FakeClient:
        async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
            return '{"beats": ["beat1"], "suggested_choices": [{"title": "Go", "description": "Depart."}]}'

        async def chat(self, *, messages, **kw):
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
        async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
            return '{"beats": ["b1"], "suggested_choices": []}'

        async def chat(self, *, messages, **kw):
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
