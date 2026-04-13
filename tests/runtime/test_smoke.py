import os
import shutil
import socket
from pathlib import Path
import pytest

from app.runtime.backend import LlamaServerBackend, RuntimeParams
from app.runtime.client import ChatMessage, InferenceClient


pytestmark = pytest.mark.integration


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def smoke_model_path() -> Path:
    env = os.environ.get("QUEST_SMOKE_MODEL")
    if not env:
        pytest.skip("QUEST_SMOKE_MODEL not set")
    p = Path(env)
    if not p.is_file():
        pytest.skip(f"model not found: {p}")
    return p


@pytest.fixture
def llama_server_binary() -> str:
    binary = shutil.which("llama-server")
    if not binary:
        pytest.skip("llama-server not on PATH")
    return binary


async def test_real_model_end_to_end(smoke_model_path: Path, llama_server_binary: str):
    backend = LlamaServerBackend(binary=llama_server_binary, ready_timeout=120.0)
    port = _free_port()
    await backend.start(RuntimeParams(model_path=smoke_model_path, port=port, ctx_size=2048))
    try:
        assert await backend.health()
        client = InferenceClient(base_url=backend.base_url, timeout=120.0)
        reply = await client.chat(
            messages=[ChatMessage(role="user", content="Say the single word: pong")],
            temperature=0.0,
            max_tokens=16,
        )
        assert isinstance(reply, str) and len(reply) > 0
    finally:
        await backend.stop()
