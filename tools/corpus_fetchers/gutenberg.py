"""Project Gutenberg plain-text fetcher.

Usage: ``python -m tools.corpus_fetchers.gutenberg <work_id>``.

Looks up ``work_id`` in ``tools/corpus_fetchers/sources.yaml`` to find the
numeric Gutenberg id, downloads plain text, strips the banner, writes
``data/calibration/raw/<work_id>/full.txt`` and ``meta.json``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

from ._http import FetchError, get_text


RAW_ROOT = Path("data/calibration/raw")
SOURCES = Path(__file__).parent / "sources.yaml"

START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
    re.IGNORECASE,
)
END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
    re.IGNORECASE,
)

log = logging.getLogger("corpus_fetchers.gutenberg")


def strip_banner(text: str) -> str:
    """Return body between PG START/END markers (or full text if absent)."""
    start = START_RE.search(text)
    body = text[start.end():] if start else text
    end = END_RE.search(body)
    if end:
        body = body[: end.start()]
    return body.strip() + "\n"


def _urls_for(gid: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]


def fetch(
    work_id: str,
    *,
    gutenberg_id: Optional[int] = None,
    raw_root: Path = RAW_ROOT,
    title: Optional[str] = None,
    author: Optional[str] = None,
    sources_path: Path = SOURCES,
    http_get=get_text,
) -> Path:
    """Download PG text for ``work_id``. Idempotent if ``full.txt`` exists."""
    if gutenberg_id is None:
        cfg = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
        entry = cfg["works"].get(work_id)
        if not entry:
            raise FetchError(f"no source entry for {work_id}")
        gutenberg_id = entry.get("gutenberg_id")
        title = title or entry.get("title")
        author = author or entry.get("author")
        if gutenberg_id is None:
            raise FetchError(
                f"{work_id}: gutenberg_id unknown; supply manually or use another fetcher"
            )

    out_dir = raw_root / work_id
    full_path = out_dir / "full.txt"
    meta_path = out_dir / "meta.json"
    if full_path.is_file() and full_path.stat().st_size > 1000:
        log.info("already fetched: %s", full_path)
        return full_path
    out_dir.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    for url in _urls_for(int(gutenberg_id)):
        log.info("fetching %s", url)
        try:
            text = http_get(url, allow_404=True)
        except FetchError as exc:
            last_err = exc
            continue
        if text is None:
            continue
        body = strip_banner(text)
        full_path.write_text(body, encoding="utf-8")
        meta = {
            "work_id": work_id,
            "title": title,
            "author": author,
            "gutenberg_id": int(gutenberg_id),
            "source_url": url,
            "license": "public_domain_us",
            "fetched_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00","Z"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("wrote %s (%d chars)", full_path, len(body))
        return full_path
    raise FetchError(f"all Gutenberg URLs failed for {work_id}: {last_err}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="gutenberg")
    ap.add_argument("work_id")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    args = ap.parse_args(argv)
    try:
        fetch(args.work_id, raw_root=Path(args.raw_root))
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
