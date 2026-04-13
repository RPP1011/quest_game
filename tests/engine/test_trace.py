from app.engine.stages import StageResult, TokenUsage
from app.engine.trace import PipelineTrace


def test_trace_accumulates_stages():
    t = PipelineTrace(trace_id="abc", trigger="greet")
    t.add_stage(StageResult(stage_name="plan", input_prompt="p", raw_output="r",
                             parsed_output={"beats": []},
                             token_usage=TokenUsage(prompt=10, completion=5)))
    t.add_stage(StageResult(stage_name="write", input_prompt="p", raw_output="prose",
                             parsed_output="prose",
                             token_usage=TokenUsage(prompt=20, completion=40)))
    assert t.total_tokens.total == 75
    assert [s.stage_name for s in t.stages] == ["plan", "write"]


def test_trace_serializable():
    t = PipelineTrace(trace_id="abc", trigger="x")
    dumped = t.model_dump()
    assert dumped["trace_id"] == "abc"
    assert dumped["stages"] == []
