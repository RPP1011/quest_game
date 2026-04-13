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
