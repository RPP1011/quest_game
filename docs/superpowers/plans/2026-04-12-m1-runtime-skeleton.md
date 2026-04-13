# M1 — Runtime Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the model runtime layer: scan the HF cache for GGUF models, manage a `llama-server` subprocess, and talk to it via an OpenAI-compatible async client.

**Architecture:** Three small, independently testable modules under `app/runtime/`. A `ModelCatalog` that walks the HF cache and returns metadata. A `RuntimeBackend` protocol with a `LlamaServerBackend` implementation that owns the subprocess lifecycle and exposes a `base_url` + `health()`. An `InferenceClient` thin async wrapper over the OpenAI-compatible endpoints using `httpx`. All I/O is async. One integration smoke test runs against a real tiny GGUF model; all other tests use fakes and fixture directories.

**Tech Stack:** Python 3.11, `uv`, `pytest`, `pytest-asyncio`, `httpx`, `pydantic` v2, external `llama-server` binary from `llama.cpp`

---

## File Structure

**Created in this plan:**
- `pyproject.toml` — project metadata, deps, tool config
- `app/__init__.py`
- `app/runtime/__init__.py`
- `app/runtime/catalog.py` — `ModelCatalog`, `ModelInfo`
- `app/runtime/backend.py` — `RuntimeBackend` protocol, `LlamaServerBackend`, `RuntimeParams`
- `app/runtime/client.py` — `InferenceClient`, message/response types
- `app/runtime/errors.py` — runtime-layer exceptions
- `tests/__init__.py`
- `tests/runtime/__init__.py`
- `tests/runtime/conftest.py` — shared fixtures (fake HF cache dir)
- `tests/runtime/test_catalog.py`
- `tests/runtime/test_backend.py`
- `tests/runtime/test_client.py`
- `tests/runtime/test_smoke.py` — opt-in, marked `integration`, runs a real tiny model
- `.gitignore`
- `README.md` — one-page quickstart

Each file has a single responsibility. The `ModelCatalog` never touches subprocesses; `LlamaServerBackend` never touches HTTP; `InferenceClient` never touches subprocesses.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `app/__init__.py`, `app/runtime/__init__.py`
- Create: `tests/__init__.py`, `tests/runtime/__init__.py`
- Create: `README.md`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "quest-game"
version = "0.1.0"
description = "Locally-hosted AI-generated forum-style quest game"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "pydantic>=2.7",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "pytest-httpx>=0.30",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: runs against a real llama-server + real model (slow, opt-in)",
]
addopts = "-m 'not integration'"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
*.egg-info/
.env
dist/
build/
node_modules/
.DS_Store
```

- [ ] **Step 3: Create empty package init files**

Each of `app/__init__.py`, `app/runtime/__init__.py`, `tests/__init__.py`, `tests/runtime/__init__.py` should be an empty file (zero bytes is fine).

- [ ] **Step 4: Write `README.md`**

```markdown
# Quest Game

Local web app for playing AI-generated forum-style quests.

## Dev setup

```
uv sync
uv run pytest
```

Integration tests (require a real `llama-server` binary and a tiny GGUF model on disk):

```
uv run pytest -m integration
```

See `docs/superpowers/specs/` for design, `docs/superpowers/plans/` for milestone plans.
```

- [ ] **Step 5: Install and verify**

Run: `uv sync && uv run pytest`
Expected: `0 tests collected` (no errors, clean exit).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore app tests README.md
git commit -m "chore: scaffold python project"
```

---

## Task 2: Runtime errors module

**Files:**
- Create: `app/runtime/errors.py`

- [ ] **Step 1: Write the module**

```python
class RuntimeError_(Exception):
    """Base for runtime-layer errors."""


class ModelNotFoundError(RuntimeError_):
    pass


class BackendStartError(RuntimeError_):
    pass


class BackendNotReadyError(RuntimeError_):
    pass


class InferenceError(RuntimeError_):
    pass
```

- [ ] **Step 2: Commit**

```bash
git add app/runtime/errors.py
git commit -m "feat(runtime): add runtime error types"
```

---

## Task 3: ModelCatalog — data types and fixture

**Files:**
- Create: `tests/runtime/conftest.py`
- Create: `tests/runtime/test_catalog.py`
- Create: `app/runtime/catalog.py`

Background for the engineer: Hugging Face's local cache lives at `~/.cache/huggingface/hub` by default. Each repo lives under `models--<org>--<name>/` with a `snapshots/<hash>/` directory that contains symlinks to blobs. GGUF files typically live inside the snapshot directory; their filenames encode quantization (e.g. `Llama-3.1-8B-Instruct-Q4_K_M.gguf`). We want: given a cache root, return a list of `ModelInfo(repo_id, filename, path, size_bytes, quant)` for every `.gguf` file we can find.

- [ ] **Step 1: Write shared conftest with a fake HF cache**

```python
# tests/runtime/conftest.py
from pathlib import Path
import pytest


@pytest.fixture
def fake_hf_cache(tmp_path: Path) -> Path:
    """Build a minimal fake HF cache layout with two GGUF files and one non-GGUF."""
    hub = tmp_path / "hub"
    repo_a = hub / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots" / "abc123"
    repo_b = hub / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "def456"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    (repo_a / "Llama-3.1-8B-Instruct-Q4_K_M.gguf").write_bytes(b"\x00" * 1024)
    (repo_a / "config.json").write_text("{}")
    (repo_b / "qwen2.5-0.5b-instruct-q8_0.gguf").write_bytes(b"\x00" * 512)
    return hub
```

- [ ] **Step 2: Write failing tests for ModelCatalog**

```python
# tests/runtime/test_catalog.py
from pathlib import Path
from app.runtime.catalog import ModelCatalog, ModelInfo


def test_scan_finds_all_gguf_files(fake_hf_cache: Path):
    catalog = ModelCatalog(cache_root=fake_hf_cache)
    models = catalog.scan()
    assert len(models) == 2
    filenames = {m.filename for m in models}
    assert filenames == {
        "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "qwen2.5-0.5b-instruct-q8_0.gguf",
    }


def test_scan_extracts_repo_id(fake_hf_cache: Path):
    catalog = ModelCatalog(cache_root=fake_hf_cache)
    models = catalog.scan()
    repo_ids = {m.repo_id for m in models}
    assert repo_ids == {
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
    }


def test_scan_extracts_quant_label(fake_hf_cache: Path):
    catalog = ModelCatalog(cache_root=fake_hf_cache)
    models = {m.filename: m for m in catalog.scan()}
    assert models["Llama-3.1-8B-Instruct-Q4_K_M.gguf"].quant == "Q4_K_M"
    assert models["qwen2.5-0.5b-instruct-q8_0.gguf"].quant == "Q8_0"


def test_scan_records_size(fake_hf_cache: Path):
    catalog = ModelCatalog(cache_root=fake_hf_cache)
    models = {m.filename: m for m in catalog.scan()}
    assert models["Llama-3.1-8B-Instruct-Q4_K_M.gguf"].size_bytes == 1024
    assert models["qwen2.5-0.5b-instruct-q8_0.gguf"].size_bytes == 512


def test_scan_returns_empty_for_missing_cache(tmp_path: Path):
    catalog = ModelCatalog(cache_root=tmp_path / "does-not-exist")
    assert catalog.scan() == []


def test_scan_is_cached_until_refresh(fake_hf_cache: Path):
    catalog = ModelCatalog(cache_root=fake_hf_cache)
    first = catalog.scan()
    (fake_hf_cache / "models--new--Repo" / "snapshots" / "xyz").mkdir(parents=True)
    (fake_hf_cache / "models--new--Repo" / "snapshots" / "xyz" / "new-Q4_0.gguf").write_bytes(b"")
    assert catalog.scan() == first  # cached
    refreshed = catalog.scan(refresh=True)
    assert len(refreshed) == len(first) + 1
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest tests/runtime/test_catalog.py -v`
Expected: ImportError / ModuleNotFoundError for `app.runtime.catalog`.

- [ ] **Step 4: Implement `app/runtime/catalog.py`**

```python
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


_QUANT_RE = re.compile(r"-(Q\d+[_A-Z0-9]*|F16|F32|BF16)\.gguf$", re.IGNORECASE)
_REPO_DIR_RE = re.compile(r"^models--(?P<org>[^-]+(?:-[^-]+)*?)--(?P<name>.+)$")


@dataclass(frozen=True)
class ModelInfo:
    repo_id: str
    filename: str
    path: Path
    size_bytes: int
    quant: str | None


class ModelCatalog:
    def __init__(self, cache_root: Path) -> None:
        self._cache_root = Path(cache_root)
        self._cache: list[ModelInfo] | None = None

    def scan(self, refresh: bool = False) -> list[ModelInfo]:
        if self._cache is not None and not refresh:
            return self._cache
        if not self._cache_root.is_dir():
            self._cache = []
            return self._cache
        results: list[ModelInfo] = []
        for repo_dir in self._cache_root.iterdir():
            if not repo_dir.is_dir():
                continue
            m = _REPO_DIR_RE.match(repo_dir.name)
            if not m:
                continue
            repo_id = f"{m['org']}/{m['name']}"
            snapshots = repo_dir / "snapshots"
            if not snapshots.is_dir():
                continue
            for gguf in snapshots.rglob("*.gguf"):
                results.append(
                    ModelInfo(
                        repo_id=repo_id,
                        filename=gguf.name,
                        path=gguf.resolve(),
                        size_bytes=gguf.stat().st_size,
                        quant=_extract_quant(gguf.name),
                    )
                )
        self._cache = sorted(results, key=lambda m: (m.repo_id, m.filename))
        return self._cache


def _extract_quant(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/runtime/test_catalog.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/runtime/catalog.py tests/runtime/conftest.py tests/runtime/test_catalog.py
git commit -m "feat(runtime): scan HF cache for GGUF models"
```

---

## Task 4: RuntimeBackend protocol + RuntimeParams

**Files:**
- Create: `app/runtime/backend.py` (protocol + params only; real backend added in Task 5)

- [ ] **Step 1: Write the module**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RuntimeParams:
    model_path: Path
    ctx_size: int = 32_768
    n_threads: int | None = None
    n_gpu_layers: int = 0
    host: str = "127.0.0.1"
    port: int = 8090
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class RuntimeBackend(Protocol):
    @property
    def base_url(self) -> str: ...
    async def start(self, params: RuntimeParams) -> None: ...
    async def stop(self) -> None: ...
    async def health(self) -> bool: ...
    @property
    def is_running(self) -> bool: ...
```

- [ ] **Step 2: Commit**

```bash
git add app/runtime/backend.py
git commit -m "feat(runtime): define RuntimeBackend protocol"
```

---

## Task 5: LlamaServerBackend implementation

**Files:**
- Modify: `app/runtime/backend.py` (append `LlamaServerBackend`)
- Create: `tests/runtime/test_backend.py`

Design: the backend shells out to `llama-server` (path configurable, default `llama-server` on PATH) with the GGUF path and params, polls `http://host:port/health` until ready (timeout 60s by default), and kills the process on `stop()`. The subprocess is owned by the backend; we never reuse an external server in this class.

- [ ] **Step 1: Write failing tests using a fake binary**

```python
# tests/runtime/test_backend.py
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
```

Note to engineer: we invoke the fake script via `python_exe <script>` rather than directly to avoid relying on shebangs working in the test env. The real backend invokes the binary directly.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/runtime/test_backend.py -v`
Expected: ImportError for `LlamaServerBackend`.

- [ ] **Step 3: Implement `LlamaServerBackend` in `app/runtime/backend.py`**

Append this to the file (keep existing `RuntimeParams` and `RuntimeBackend`):

```python
import asyncio
import shutil
import httpx
from .errors import BackendStartError


class LlamaServerBackend:
    def __init__(
        self,
        binary: str = "llama-server",
        ready_timeout: float = 60.0,
        python_exe: str | None = None,
    ) -> None:
        self._binary = binary
        self._ready_timeout = ready_timeout
        self._python_exe = python_exe  # test-only: wrap binary with interpreter
        self._proc: asyncio.subprocess.Process | None = None
        self._params: RuntimeParams | None = None

    @property
    def base_url(self) -> str:
        if self._params is None:
            raise RuntimeError("backend not started")
        return f"http://{self._params.host}:{self._params.port}"

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self, params: RuntimeParams) -> None:
        if self.is_running:
            raise BackendStartError("backend already running")
        if self._python_exe is None and shutil.which(self._binary) is None and not Path(self._binary).is_file():
            raise BackendStartError(f"binary not found: {self._binary}")
        argv = self._build_argv(params)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as e:
            raise BackendStartError(str(e)) from e
        self._params = params
        try:
            await self._wait_ready()
        except Exception:
            await self.stop()
            raise

    def _build_argv(self, params: RuntimeParams) -> list[str]:
        argv: list[str] = []
        if self._python_exe is not None:
            argv.append(self._python_exe)
        argv.append(self._binary)
        argv += [
            "--model", str(params.model_path),
            "--host", params.host,
            "--port", str(params.port),
            "--ctx-size", str(params.ctx_size),
            "--n-gpu-layers", str(params.n_gpu_layers),
        ]
        if params.n_threads is not None:
            argv += ["--threads", str(params.n_threads)]
        argv += list(params.extra_args)
        return argv

    async def _wait_ready(self) -> None:
        assert self._params is not None
        deadline = asyncio.get_event_loop().time() + self._ready_timeout
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient(timeout=1.0) as client:
            while True:
                if self._proc is None or self._proc.returncode is not None:
                    raise BackendStartError(f"llama-server exited (rc={self._proc.returncode if self._proc else None})")
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_event_loop().time() > deadline:
                    raise BackendStartError(f"llama-server not ready after {self._ready_timeout}s")
                await asyncio.sleep(0.2)

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        self._params = None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def health(self) -> bool:
        if not self.is_running or self._params is None:
            return False
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/runtime/test_backend.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/runtime/backend.py tests/runtime/test_backend.py
git commit -m "feat(runtime): manage llama-server subprocess lifecycle"
```

---

## Task 6: InferenceClient — chat completions

**Files:**
- Create: `app/runtime/client.py`
- Create: `tests/runtime/test_client.py`

Design: an async client for the OpenAI-compatible `/v1/chat/completions` endpoint served by `llama-server`. Two methods: `chat(messages, **opts)` for full responses, `stream_chat(messages, **opts)` for SSE-style token streaming. Messages use a simple pydantic model.

- [ ] **Step 1: Write failing tests with `pytest-httpx`**

```python
# tests/runtime/test_client.py
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/runtime/test_client.py -v`
Expected: ImportError for `app.runtime.client`.

- [ ] **Step 3: Implement `app/runtime/client.py`**

```python
from __future__ import annotations
import json
from typing import AsyncIterator, Literal
import httpx
from pydantic import BaseModel
from .errors import InferenceError


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InferenceClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **extra: object,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, stream=False, extra=extra)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                r = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise InferenceError(str(e)) from e
            data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise InferenceError(f"malformed response: {data!r}") from e

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **extra: object,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, temperature, max_tokens, stream=True, extra=extra)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", f"{self._base_url}/v1/chat/completions", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            return
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token
            except httpx.HTTPError as e:
                raise InferenceError(str(e)) from e

    def _build_payload(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        *,
        stream: bool,
        extra: dict[str, object],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)
        return payload
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/runtime/test_client.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/runtime/client.py tests/runtime/test_client.py
git commit -m "feat(runtime): async OpenAI-compatible inference client"
```

---

## Task 7: Integration smoke test (opt-in)

**Files:**
- Create: `tests/runtime/test_smoke.py`

Purpose: end-to-end proof that `ModelCatalog → LlamaServerBackend → InferenceClient` wire up correctly against real `llama-server` + a real GGUF. Marked `integration`, excluded from default `pytest` run (see `pyproject.toml`).

Engineer note: this test requires (a) `llama-server` on PATH, and (b) env var `QUEST_SMOKE_MODEL` pointing to a small GGUF on disk (e.g. Qwen2.5-0.5B-Instruct-Q4_K_M). Skips otherwise. This is the only test in the suite that loads a real model; it should run in < 60s on CPU for a 0.5B model.

- [ ] **Step 1: Write the test**

```python
# tests/runtime/test_smoke.py
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
```

- [ ] **Step 2: Verify it's excluded from default run**

Run: `uv run pytest -v`
Expected: the smoke test is deselected (not run), all other runtime tests PASS.

- [ ] **Step 3: (Optional, locally) run the integration test**

If the engineer has `llama-server` and a small GGUF:

```
QUEST_SMOKE_MODEL=/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  uv run pytest -m integration -v
```

Expected: one test PASSES.

- [ ] **Step 4: Commit**

```bash
git add tests/runtime/test_smoke.py
git commit -m "test(runtime): end-to-end integration smoke test"
```

---

## Task 8: Public package surface

**Files:**
- Modify: `app/runtime/__init__.py`

- [ ] **Step 1: Export the public API**

```python
from .backend import LlamaServerBackend, RuntimeBackend, RuntimeParams
from .catalog import ModelCatalog, ModelInfo
from .client import ChatMessage, InferenceClient
from .errors import (
    BackendNotReadyError,
    BackendStartError,
    InferenceError,
    ModelNotFoundError,
)

__all__ = [
    "BackendNotReadyError",
    "BackendStartError",
    "ChatMessage",
    "InferenceClient",
    "InferenceError",
    "LlamaServerBackend",
    "ModelCatalog",
    "ModelInfo",
    "ModelNotFoundError",
    "RuntimeBackend",
    "RuntimeParams",
]
```

- [ ] **Step 2: Verify imports resolve**

Run: `uv run python -c "from app.runtime import ModelCatalog, LlamaServerBackend, InferenceClient, ChatMessage; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: all non-integration tests PASS (catalog: 6, backend: 4, client: 4 = 14 tests).

- [ ] **Step 4: Commit**

```bash
git add app/runtime/__init__.py
git commit -m "feat(runtime): expose public api"
```

---

## Done criteria

- `uv run pytest` passes with 14 tests
- `uv run pytest -m integration` passes locally when `llama-server` + `QUEST_SMOKE_MODEL` are available
- `from app.runtime import ...` resolves the documented public API
- No open TODOs in the M1 code

M2 (Engine skeleton) planning starts once M1 is merged and you can import real `ModelCatalog`, `LlamaServerBackend`, `InferenceClient`.
