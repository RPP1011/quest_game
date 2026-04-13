from pathlib import Path
import pytest
from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.world import Entity, EntityType, StateDelta, WorldStateManager
from app.world.delta import EntityCreate
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class ScriptedClient:
    """Replay a scripted sequence of responses, one per call in order."""
    def __init__(self, responses: list[dict]) -> None:
        # Each response: {"kind": "structured"|"chat", "content": str}
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


@pytest.fixture
def world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(entity_creates=[
        EntityCreate(entity=Entity(id="a", entity_type=EntityType.CHARACTER, name="A")),
    ]), update_number=1)
    yield sm
    conn.close()


def _cb(world):
    return ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())


async def test_clean_check_commits_directly(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["beat1"], "suggested_choices": ["x"]}'},
        {"kind": "chat", "content": "Prose v1."},
        {"kind": "structured", "content": '{"issues": []}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.prose == "Prose v1."
    assert out.trace.outcome == "committed"
    assert [s.stage_name for s in out.trace.stages] == ["plan", "write", "check"]


async def test_warning_triggers_revise_and_recheck(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["beat1"], "suggested_choices": []}'},
        {"kind": "chat", "content": "Prose v1 with purple prose."},
        {"kind": "structured", "content": '{"issues": [{"severity": "warning", "category": "prose_quality", "message": "purple prose", "suggested_fix": "simplify"}]}'},
        {"kind": "chat", "content": "Prose v2 simpler."},
        {"kind": "structured", "content": '{"issues": []}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.prose == "Prose v2 simpler."
    assert out.trace.outcome == "committed"
    stage_names = [s.stage_name for s in out.trace.stages]
    assert stage_names == ["plan", "write", "check", "revise", "check"]


async def test_critical_triggers_replan_once(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["bad beat"], "suggested_choices": []}'},
        {"kind": "chat", "content": "Prose v1."},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "magic banned"}]}'},
        # REPLAN
        {"kind": "structured", "content": '{"beats": ["better beat"], "suggested_choices": []}'},
        {"kind": "chat", "content": "Prose v2."},
        {"kind": "structured", "content": '{"issues": []}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.prose == "Prose v2."
    stage_names = [s.stage_name for s in out.trace.stages]
    assert stage_names == ["plan", "write", "check", "plan", "write", "check"]
    assert out.trace.outcome == "committed"


async def test_critical_after_replan_flags_qm(world):
    client = ScriptedClient([
        {"kind": "structured", "content": '{"beats": ["b1"], "suggested_choices": []}'},
        {"kind": "chat", "content": "v1"},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "bad"}]}'},
        # REPLAN
        {"kind": "structured", "content": '{"beats": ["b2"], "suggested_choices": []}'},
        {"kind": "chat", "content": "v2"},
        {"kind": "structured", "content": '{"issues": [{"severity": "critical", "category": "world_rule", "message": "still bad"}]}'},
    ])
    out = await Pipeline(world, _cb(world), client).run(player_action="act", update_number=2)
    assert out.trace.outcome == "flagged_qm"
    # Narrative is still written (v2), flagged
    assert world.list_narrative()[0].raw_text == "v2"
