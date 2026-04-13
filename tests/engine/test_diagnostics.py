from pathlib import Path
import pytest
from app.engine.diagnostics import DiagnosticsManager
from app.engine.stages import StageResult
from app.engine.trace import PipelineTrace
from app.engine.trace_store import TraceStore


class RecClient:
    def __init__(self, structured="{}", chat="text"):
        self.structured_response = structured
        self.chat_response = chat
        self.last = None

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw):
        self.last = ("structured", messages, schema_name)
        return self.structured_response

    async def chat(self, *, messages, **kw):
        self.last = ("chat", messages)
        return self.chat_response


def _seed_trace(store: TraceStore) -> str:
    t = PipelineTrace(trace_id="t1", trigger="greet")
    t.add_stage(StageResult(stage_name="plan", input_prompt="orig plan prompt",
                             raw_output="{}", parsed_output={"beats": []}))
    t.add_stage(StageResult(stage_name="write", input_prompt="orig write prompt",
                             raw_output="orig prose", parsed_output="orig prose"))
    store.save(t)
    return "t1"


async def test_replay_plan_stage(tmp_path: Path):
    store = TraceStore(tmp_path)
    tid = _seed_trace(store)
    client = RecClient(structured='{"beats":["new"],"suggested_choices":[]}')
    dm = DiagnosticsManager(client=client, store=store)
    r = await dm.replay(tid, "plan")
    assert r.stage_name == "plan"
    assert client.last[0] == "structured"
    # original prompt was reused
    assert any("orig plan prompt" in m.content for m in client.last[1])
    assert r.raw_output == '{"beats":["new"],"suggested_choices":[]}'


async def test_replay_write_uses_override(tmp_path: Path):
    store = TraceStore(tmp_path)
    tid = _seed_trace(store)
    client = RecClient(chat="new prose")
    dm = DiagnosticsManager(client=client, store=store)
    r = await dm.replay(tid, "write", prompt_override="CHANGED PROMPT")
    assert client.last[0] == "chat"
    assert any(m.content == "CHANGED PROMPT" for m in client.last[1])
    assert r.parsed_output == "new prose"


async def test_replay_missing_stage_raises(tmp_path: Path):
    store = TraceStore(tmp_path)
    tid = _seed_trace(store)
    dm = DiagnosticsManager(client=RecClient(), store=store)
    with pytest.raises(ValueError):
        await dm.replay(tid, "nonexistent")
