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
