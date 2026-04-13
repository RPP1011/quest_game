import pytest
from pytest_httpx import HTTPXMock
from app.runtime.client import InferenceClient, ChatMessage
from app.runtime.errors import InferenceError


BASE = "http://127.0.0.1:8090"


async def test_chat_returns_assistant_content(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions",
        method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": "hello there"}}]},
    )
    client = InferenceClient(base_url=BASE)
    result = await client.chat(messages=[ChatMessage(role="user", content="hi")])
    assert result == "hello there"


async def test_chat_passes_sampling_params(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions",
        method="POST",
        json={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
    )
    client = InferenceClient(base_url=BASE)
    await client.chat(
        messages=[ChatMessage(role="user", content="hi")],
        temperature=0.8,
        max_tokens=64,
    )
    req = httpx_mock.get_requests()[0]
    import json
    body = json.loads(req.content)
    assert body["temperature"] == 0.8
    assert body["max_tokens"] == 64
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is False


async def test_chat_raises_on_http_error(httpx_mock: HTTPXMock):
    httpx_mock.add_response(url=f"{BASE}/v1/chat/completions", method="POST", status_code=500)
    client = InferenceClient(base_url=BASE)
    with pytest.raises(InferenceError):
        await client.chat(messages=[ChatMessage(role="user", content="hi")])


async def test_stream_chat_yields_tokens(httpx_mock: HTTPXMock):
    sse = (
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        b'data: {"choices":[{"delta":{}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    httpx_mock.add_response(
        url=f"{BASE}/v1/chat/completions",
        method="POST",
        content=sse,
        headers={"content-type": "text/event-stream"},
    )
    client = InferenceClient(base_url=BASE)
    tokens = []
    async for t in client.stream_chat(messages=[ChatMessage(role="user", content="hi")]):
        tokens.append(t)
    assert tokens == ["hel", "lo"]
