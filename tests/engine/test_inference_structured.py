import json
import pytest
from pytest_httpx import HTTPXMock
from app.runtime.client import ChatMessage, InferenceClient


BASE = "http://127.0.0.1:8090"


async def test_chat_structured_includes_schema(httpx_mock: HTTPXMock):
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions", method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": '{"x": 7}'}}]},
    )
    client = InferenceClient(base_url=BASE)
    result = await client.chat_structured(
        messages=[ChatMessage(role="user", content="give me x")],
        json_schema=schema,
        schema_name="Thing",
    )
    assert json.loads(result) == {"x": 7}
    body = json.loads(httpx_mock.get_requests()[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "Thing"
    assert body["response_format"]["json_schema"]["schema"] == schema
