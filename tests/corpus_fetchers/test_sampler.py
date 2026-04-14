from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools import sample_passages


def _make_chapter(i: int, words: int = 3000, dialogue: bool = False) -> str:
    base = f"ChapterStart{i} " + " ".join(f"word{j}" for j in range(words))
    if dialogue:
        # Inject lots of quoted speech.
        speech = ' '.join(['"Hello there, my friend."'] * 60)
        mid = words // 2
        tokens = base.split()
        tokens = tokens[:mid] + [speech] + tokens[mid:]
        return " ".join(tokens)
    return base


def _write_raw(tmp_path: Path, work_id: str, n_chapters: int = 10) -> Path:
    d = tmp_path / "raw" / work_id
    d.mkdir(parents=True)
    for i in range(1, n_chapters + 1):
        dialogue = (i % 2 == 0)
        (d / f"chap_{i:04d}.txt").write_text(_make_chapter(i, 3000, dialogue))
    return d


def test_sample_work_determinism_and_constraints(tmp_path: Path) -> None:
    _write_raw(tmp_path, "wid", n_chapters=10)
    work = {
        "id": "wid",
        "title": "Widget",
        "author": "Anon",
        "passages": [
            {"id": "p01", "expected_high": [], "expected_low": []},
            {"id": "p02", "expected_high": [], "expected_low": []},
        ],
    }
    raw_root = tmp_path / "raw"
    p1 = tmp_path / "p1"
    p2 = tmp_path / "p2"
    sample_passages.sample_work(work, raw_root=raw_root, passages_root=p1)
    sample_passages.sample_work(work, raw_root=raw_root, passages_root=p2)

    t1a = (p1 / "wid" / "p01.txt").read_text()
    t2a = (p2 / "wid" / "p01.txt").read_text()
    t1b = (p1 / "wid" / "p02.txt").read_text()
    t2b = (p2 / "wid" / "p02.txt").read_text()
    assert t1a == t2a  # deterministic
    assert t1b == t2b

    # frontmatter present
    assert t1a.startswith("---\n")
    fm_end = t1a.index("---\n", 4)
    fm = yaml.safe_load(t1a[4:fm_end])
    assert fm["work"] == "Widget"
    assert fm["passage_id"] == "p01"
    body = t1a[fm_end + 4:].strip()
    wc = len(body.split())
    assert 500 <= wc <= 1000


def test_sample_dialogue_balance(tmp_path: Path) -> None:
    _write_raw(tmp_path, "wid", n_chapters=12)
    work = {
        "id": "wid",
        "title": "Widget",
        "author": "Anon",
        "passages": [
            {"id": "p01", "expected_high": [], "expected_low": []},
            {"id": "p02", "expected_high": [], "expected_low": []},
        ],
    }
    raw_root = tmp_path / "raw"
    pr = tmp_path / "out"
    sample_passages.sample_work(work, raw_root=raw_root, passages_root=pr)
    from app.calibration.heuristics import dialogue_ratio

    b1 = (pr / "wid" / "p01.txt").read_text().split("---\n", 2)[-1]
    b2 = (pr / "wid" / "p02.txt").read_text().split("---\n", 2)[-1]
    # p01 is the dialogue-lighter slot
    assert dialogue_ratio(b1) <= dialogue_ratio(b2)


def test_sample_mid_chapter_window(tmp_path: Path) -> None:
    # Put a unique token only in the forbidden first-20% / last-15% region and
    # confirm we never sample it.
    d = tmp_path / "raw" / "wid"
    d.mkdir(parents=True)
    for i in range(1, 8):
        words = [f"w{j}" for j in range(3000)]
        # Sentinel in first 10%
        words[50] = "EARLYSENTINEL"
        # Sentinel in last 10%
        words[2950] = "LATESENTINEL"
        (d / f"chap_{i:04d}.txt").write_text(" ".join(words))
    work = {
        "id": "wid",
        "title": "W",
        "author": "A",
        "passages": [{"id": "p01", "expected_high": [], "expected_low": []}],
    }
    sample_passages.sample_work(
        work, raw_root=tmp_path / "raw", passages_root=tmp_path / "out"
    )
    body = (tmp_path / "out" / "wid" / "p01.txt").read_text()
    assert "EARLYSENTINEL" not in body
    assert "LATESENTINEL" not in body


def test_sample_from_gutenberg_full_text(tmp_path: Path) -> None:
    d = tmp_path / "raw" / "wid"
    d.mkdir(parents=True)
    parts = []
    for i in range(1, 16):
        parts.append(f"CHAPTER {i}\n\n" + " ".join(f"w{j}" for j in range(2000)))
    (d / "full.txt").write_text("\n\n".join(parts))
    work = {
        "id": "wid",
        "title": "W",
        "author": "A",
        "passages": [
            {"id": "p01", "expected_high": [], "expected_low": []},
            {"id": "p02", "expected_high": [], "expected_low": []},
        ],
    }
    sample_passages.sample_work(
        work, raw_root=tmp_path / "raw", passages_root=tmp_path / "out"
    )
    assert (tmp_path / "out" / "wid" / "p01.txt").is_file()
    assert (tmp_path / "out" / "wid" / "p02.txt").is_file()
