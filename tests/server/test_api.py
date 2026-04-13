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
            return '{"beats": ["beat1"], "suggested_choices": ["Go"]}'

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
    assert body["choices"] == ["Go"]
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
    assert chapters[0]["choices"] == ["Go"]
