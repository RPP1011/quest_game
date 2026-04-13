from __future__ import annotations
import asyncio
import sys
import textwrap
from pathlib import Path
import pytest
import httpx

from app.runtime.backend import LlamaServerBackend, RuntimeParams
from app.runtime.errors import BackendStartError


@pytest.fixture
def fake_server_script(tmp_path: Path) -> Path:
    """A tiny python 'llama-server' that serves /health on the requested port."""
    script = tmp_path / "fake-llama-server"
    script.write_text(textwrap.dedent(r"""
        #!/usr/bin/env python3
        import argparse, http.server, sys
        p = argparse.ArgumentParser()
        p.add_argument("--model"); p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, required=True); p.add_argument("--ctx-size", type=int)
        p.add_argument("--n-gpu-layers", type=int, default=0)
        p.add_argument("--threads", type=int, default=0)
        args, _ = p.parse_known_args()
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_response(404); self.end_headers()
            def log_message(self, *a, **kw): pass
        http.server.HTTPServer((args.host, args.port), H).serve_forever()
    """))
    script.chmod(0o755)
    return script


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"\x00")
    return p


def _free_port() -> int:
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


async def test_start_launches_subprocess_and_becomes_healthy(fake_server_script, model_path):
    port = _free_port()
    backend = LlamaServerBackend(binary=str(fake_server_script), python_exe=sys.executable)
    try:
        await backend.start(RuntimeParams(model_path=model_path, port=port))
        assert backend.is_running
        assert await backend.health()
        assert backend.base_url == f"http://127.0.0.1:{port}"
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{backend.base_url}/health")
            assert r.status_code == 200
    finally:
        await backend.stop()


async def test_stop_terminates_process(fake_server_script, model_path):
    port = _free_port()
    backend = LlamaServerBackend(binary=str(fake_server_script), python_exe=sys.executable)
    await backend.start(RuntimeParams(model_path=model_path, port=port))
    await backend.stop()
    assert not backend.is_running
    async with httpx.AsyncClient(timeout=0.5) as c:
        with pytest.raises(httpx.HTTPError):
            await c.get(f"http://127.0.0.1:{port}/health")


async def test_start_raises_if_binary_missing(model_path):
    backend = LlamaServerBackend(binary="/nonexistent/llama-server-xyz")
    with pytest.raises(BackendStartError):
        await backend.start(RuntimeParams(model_path=model_path, port=_free_port()))


async def test_start_raises_if_health_never_ready(tmp_path, model_path):
    # Fake binary that exits immediately
    script = tmp_path / "broken-server"
    script.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(1)\n")
    script.chmod(0o755)
    backend = LlamaServerBackend(binary=str(script), python_exe=sys.executable, ready_timeout=1.0)
    with pytest.raises(BackendStartError):
        await backend.start(RuntimeParams(model_path=model_path, port=_free_port()))
