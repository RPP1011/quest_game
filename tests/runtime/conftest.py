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
