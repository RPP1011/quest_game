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
