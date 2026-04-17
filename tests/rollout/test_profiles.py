from __future__ import annotations
import json

import pytest

from app.rollout.action_selector import select_action
from app.rollout.profiles import (
    VirtualPlayerProfile, list_profiles, load_profile,
)


def test_list_profiles_bundled():
    ps = list_profiles()
    ids = {p.id for p in ps}
    assert "impulsive" in ids
    assert "cautious" in ids
    assert "honor_bound" in ids


def test_load_profile_impulsive():
    p = load_profile("impulsive")
    assert p.id == "impulsive"
    assert "impulsive" in p.description.lower()
    assert p.action_selection_rubric


def test_load_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("no_such_profile")


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list = []

    async def chat_structured(self, messages, *, json_schema, schema_name="Output", **kw):
        self.calls.append((messages, json_schema, schema_name))
        return self._response


@pytest.mark.asyncio
async def test_select_action_returns_chosen_index():
    client = FakeClient(json.dumps({
        "chosen_index": 2,
        "rationale": "highest stakes per impulsive rubric",
    }))
    profile = load_profile("impulsive")
    choices = [
        {"title": "Wait and watch", "description": "Gather info"},
        {"title": "Retreat", "description": "Preserve resources"},
        {"title": "Charge the door", "description": "Commit to confrontation"},
    ]
    idx, rationale = await select_action(
        client=client, profile=profile, choices=choices,
    )
    assert idx == 2
    assert "stakes" in rationale.lower()


@pytest.mark.asyncio
async def test_select_action_clamps_out_of_range():
    client = FakeClient(json.dumps({"chosen_index": 99, "rationale": "oops"}))
    profile = load_profile("cautious")
    choices = [{"title": "A"}, {"title": "B"}]
    idx, _ = await select_action(
        client=client, profile=profile, choices=choices,
    )
    assert idx == 0  # clamped to valid range


@pytest.mark.asyncio
async def test_select_action_empty_choices():
    client = FakeClient("{}")  # won't be called
    profile = load_profile("cautious")
    idx, reason = await select_action(
        client=client, profile=profile, choices=[],
    )
    assert idx == 0
    assert "no choices" in reason.lower()


@pytest.mark.asyncio
async def test_select_action_falls_back_on_error():
    class BrokenClient:
        async def chat_structured(self, *a, **kw):
            raise RuntimeError("server down")
    profile = load_profile("impulsive")
    idx, reason = await select_action(
        client=BrokenClient(), profile=profile,
        choices=[{"title": "A"}, {"title": "B"}],
    )
    assert idx == 0
    assert "fallback" in reason.lower()


@pytest.mark.asyncio
async def test_select_action_schema_bounds(tmp_path):
    client = FakeClient(json.dumps({"chosen_index": 1, "rationale": "x"}))
    profile = VirtualPlayerProfile(
        id="test", description="test", action_selection_rubric="pick anything",
    )
    choices = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    await select_action(client=client, profile=profile, choices=choices)
    _, schema, _ = client.calls[0]
    assert schema["properties"]["chosen_index"]["maximum"] == 2
