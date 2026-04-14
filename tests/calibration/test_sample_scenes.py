"""Tests for tools/sample_scenes.py."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools import sample_scenes


def _fixture_chapter(n: int, words_per_scene: int = 1200, scenes: int = 3) -> str:
    """Build a chapter with ``scenes`` scenes separated by '* * *'."""
    blocks = []
    for s in range(scenes):
        paras = []
        per_para = 30
        n_paras = max(1, words_per_scene // per_para)
        for p in range(n_paras):
            paras.append(
                " ".join(f"w{n}_{s}_{p}_{k}" for k in range(per_para))
            )
        blocks.append("\n\n".join(paras))
    body = "\n\n* * *\n\n".join(blocks)
    return f"CHAPTER {n}\n\n{body}"


def _build_corpus(root: Path, work_id: str, n_chapters: int = 8) -> Path:
    d = root / work_id
    d.mkdir(parents=True)
    full = "\n\n".join(_fixture_chapter(i + 1) for i in range(n_chapters))
    (d / "full.txt").write_text(full, encoding="utf-8")
    return d


def test_find_scene_breaks_detects_dividers():
    text = "alpha beta\n\n* * *\n\ngamma delta\n\n---\n\nepsilon"
    offs = sample_scenes.find_scene_breaks(text)
    assert 0 in offs
    assert len(text) in offs
    # Two explicit dividers plus start/end.
    assert len(offs) >= 4


def test_find_scene_breaks_detects_paragraph_gap():
    text = "one two three\n\n\n\nfour five six"
    offs = sample_scenes.find_scene_breaks(text)
    # Start + end + the paragraph-gap break.
    assert len(offs) == 3


def test_sample_work_deterministic_and_word_count(tmp_path: Path):
    raw = tmp_path / "raw"
    scenes = tmp_path / "scenes"
    _build_corpus(raw, "demo")
    work = {"id": "demo", "title": "Demo", "author": "T"}

    out1 = sample_scenes.sample_work(
        work, raw_root=raw, scenes_root=scenes, n_scenes=3,
    )
    texts1 = [p.read_text(encoding="utf-8") for p in out1]

    # Rerun in a fresh dir → identical bytes.
    scenes2 = tmp_path / "scenes2"
    out2 = sample_scenes.sample_work(
        work, raw_root=raw, scenes_root=scenes2, n_scenes=3,
    )
    texts2 = [p.read_text(encoding="utf-8") for p in out2]
    assert texts1 == texts2

    for path in out1:
        body = path.read_text(encoding="utf-8").split("---", 2)[2]
        wc = len(body.split())
        assert sample_scenes.TARGET_MIN_WORDS <= wc <= sample_scenes.TARGET_MAX_WORDS, (
            f"{path.name} had {wc} words"
        )

    # Frontmatter is valid YAML.
    fm = out1[0].read_text(encoding="utf-8").split("---", 2)[1]
    data = yaml.safe_load(fm)
    assert data["kind"] == "scene"
    assert "word_count" in data


def test_build_scene_respects_window(tmp_path: Path):
    raw = tmp_path / "raw"
    _build_corpus(raw, "demo")
    chapters = sample_scenes.load_chapters(raw / "demo")
    scene, sources = sample_scenes.build_scene(chapters, 0)
    wc = len(scene.split())
    assert sample_scenes.TARGET_MIN_WORDS <= wc <= sample_scenes.TARGET_MAX_WORDS
    assert sources[0] == chapters[0].index
