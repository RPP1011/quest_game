from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.corpus_fetchers import gutenberg
from tools.corpus_fetchers._http import FetchError


GUTEN_SAMPLE = """\
Some preamble metadata here.

*** START OF THE PROJECT GUTENBERG EBOOK TITLE ***

Real body line one.
Chapter I content.

More body.

*** END OF THE PROJECT GUTENBERG EBOOK TITLE ***

License blurb here.
"""


def test_strip_banner() -> None:
    out = gutenberg.strip_banner(GUTEN_SAMPLE)
    assert "preamble metadata" not in out
    assert "License blurb" not in out
    assert "Real body line one." in out
    assert "Chapter I content." in out


def test_fetch_happy_path(tmp_path: Path) -> None:
    calls: list[str] = []

    def http_get(url: str, *, allow_404: bool = False):
        calls.append(url)
        return GUTEN_SAMPLE

    out = gutenberg.fetch(
        "pride_and_prejudice",
        gutenberg_id=1342,
        raw_root=tmp_path,
        title="P&P",
        author="JA",
        http_get=http_get,
    )
    assert out.is_file()
    body = out.read_text()
    assert "Real body" in body and "START OF" not in body
    meta = json.loads((tmp_path / "pride_and_prejudice" / "meta.json").read_text())
    assert meta["gutenberg_id"] == 1342
    assert meta["work_id"] == "pride_and_prejudice"
    assert calls[0].endswith("pg1342.txt")


def test_fetch_falls_back_on_404(tmp_path: Path) -> None:
    calls: list[str] = []

    def http_get(url: str, *, allow_404: bool = False):
        calls.append(url)
        if "cache/epub" in url:
            return None  # 404
        return GUTEN_SAMPLE

    gutenberg.fetch(
        "ulysses",
        gutenberg_id=4300,
        raw_root=tmp_path,
        http_get=http_get,
    )
    assert len(calls) >= 2


def test_fetch_idempotent(tmp_path: Path) -> None:
    (tmp_path / "ulysses").mkdir(parents=True)
    full = tmp_path / "ulysses" / "full.txt"
    full.write_text("X" * 5000)

    def http_get(url: str, *, allow_404: bool = False):  # pragma: no cover
        raise AssertionError("should not be called")

    out = gutenberg.fetch(
        "ulysses", gutenberg_id=4300, raw_root=tmp_path, http_get=http_get
    )
    assert out == full


def test_fetch_all_urls_fail(tmp_path: Path) -> None:
    def http_get(url: str, *, allow_404: bool = False):
        return None

    with pytest.raises(FetchError):
        gutenberg.fetch(
            "ulysses", gutenberg_id=4300, raw_root=tmp_path, http_get=http_get
        )
