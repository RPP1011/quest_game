from app.engine.inference_params import InferenceParams, TokenUsage
from app.engine.stages import StageConfig, StageError, StageResult


def test_inference_params_defaults():
    p = InferenceParams()
    assert p.temperature == 0.7
    assert p.thinking is True
    assert p.max_tokens is None
    assert p.top_p == 1.0


def test_token_usage_total():
    u = TokenUsage(prompt=100, completion=50, thinking=20)
    assert u.total == 170


def test_stage_config_roundtrip():
    c = StageConfig(
        name="plan",
        system_prompt_template="stages/plan/system.j2",
        user_prompt_template="stages/plan/user.j2",
        output_schema={"type": "object"},
        inference_params=InferenceParams(temperature=0.4, thinking=True),
    )
    assert c.name == "plan"
    assert c.max_retries == 2


def test_stage_result_records_everything():
    r = StageResult(
        stage_name="plan",
        input_prompt="hello",
        raw_output="world",
        parsed_output={"beats": []},
        token_usage=TokenUsage(prompt=10, completion=5),
        latency_ms=120,
        retries=0,
        errors=[],
    )
    assert r.parsed_output == {"beats": []}
    assert r.token_usage.total == 15


def test_stage_error_carries_context():
    e = StageError(kind="parse_error", message="bad json", detail={"snippet": "{"})
    assert e.kind == "parse_error"
