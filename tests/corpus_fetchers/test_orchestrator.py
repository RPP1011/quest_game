from __future__ import annotations

from pathlib import Path

import yaml

from tools import build_corpus


def _minimal_manifest(tmp_path: Path) -> Path:
    manifest = {
        "version": 1,
        "scoring": {"critic_error_weight": 0.25, "critic_warning_weight": 0.1},
        "works": [
            {
                "id": "pride_and_prejudice",
                "title": "P&P",
                "author": "JA",
                "year": 1813,
                "pov": "third_limited_fis",
                "is_quest": False,
                "expected": {},
                "passages": [
                    {"id": "p01", "sha256": "PENDING",
                     "expected_high": [], "expected_low": []},
                    {"id": "p02", "sha256": "PENDING",
                     "expected_high": [], "expected_low": []},
                ],
            }
        ],
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return p


def test_orchestrator_end_to_end(tmp_path: Path, monkeypatch) -> None:
    manifest = _minimal_manifest(tmp_path)
    raw_root = tmp_path / "raw"
    passages_root = tmp_path / "passages"

    # Stub gutenberg.fetch to write a plausible full.txt
    def fake_fetch(work_id, *, raw_root, **kw):
        d = raw_root / work_id
        d.mkdir(parents=True, exist_ok=True)
        # 15 chapters of ~2000 words each
        parts = []
        for i in range(1, 16):
            parts.append(
                f"CHAPTER {i}\n\n"
                + " ".join(f"w{j}" for j in range(2000))
            )
        (d / "full.txt").write_text("\n\n".join(parts))
        (d / "meta.json").write_text("{}")
        return d / "full.txt"

    monkeypatch.setitem(build_corpus.FETCHERS, "gutenberg", fake_fetch)

    # Also write a minimal sources.yaml in tmp
    sources = {
        "works": {
            "pride_and_prejudice": {
                "type": "gutenberg",
                "gutenberg_id": 1342,
                "license": "public_domain_us",
            }
        }
    }
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(yaml.safe_dump(sources))

    rc = build_corpus.run(
        ["pride_and_prejudice"],
        manifest_path=manifest,
        raw_root=raw_root,
        passages_root=passages_root,
        sources_path=sources_path,
        run_init=True,
    )
    assert rc == 0
    assert (passages_root / "pride_and_prejudice" / "p01.txt").is_file()
    assert (passages_root / "pride_and_prejudice" / "p02.txt").is_file()
    # init should have updated hashes
    updated = yaml.safe_load(manifest.read_text())
    sha = updated["works"][0]["passages"][0]["sha256"]
    assert sha != "PENDING" and len(sha) == 64
