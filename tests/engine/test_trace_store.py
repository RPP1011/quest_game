# tests/engine/test_trace_store.py
from pathlib import Path
from app.engine.stages import StageResult
from app.engine.trace import PipelineTrace
from app.engine.trace_store import TraceStore


def test_save_and_load_roundtrip(tmp_path: Path):
    store = TraceStore(tmp_path)
    t = PipelineTrace(trace_id="abc", trigger="greet")
    t.add_stage(StageResult(stage_name="plan", input_prompt="p", raw_output="r", parsed_output={"beats": ["b"]}))
    t.outcome = "committed"
    store.save(t)
    loaded = store.load("abc")
    assert loaded.trace_id == "abc"
    assert loaded.outcome == "committed"
    assert loaded.stages[0].parsed_output == {"beats": ["b"]}


def test_list_ids(tmp_path: Path):
    store = TraceStore(tmp_path)
    for tid in ("a", "b", "c"):
        store.save(PipelineTrace(trace_id=tid, trigger="x"))
    assert sorted(store.list_ids()) == ["a", "b", "c"]


def test_load_missing_raises(tmp_path: Path):
    import pytest
    store = TraceStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")
