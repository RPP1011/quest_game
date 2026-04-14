"""WordPress Table-of-Contents walker.

Given a TOC page URL, extract chapter links (anchors to posts on the same
host), fetch each, and dump the article body text. Idempotent; respects
the shared rate limit in :mod:`tools.corpus_fetchers._http`.
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
from urllib.parse import urljoin, urlparse

import yaml

from ._http import FetchError, get_text


RAW_ROOT = Path("data/calibration/raw")
SOURCES = Path(__file__).parent / "sources.yaml"

log = logging.getLogger("corpus_fetchers.wordpress")

ANCHOR_RE = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
ENTRY_RE = re.compile(
    r'<div[^>]*class="[^"]*entry-content[^"]*"[^>]*>(.*?)</div>\s*<!--\s*\.entry-content',
    re.IGNORECASE | re.DOTALL,
)
ARTICLE_RE = re.compile(r"<article[^>]*>(.*?)</article>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SKIP_LINK_WORDS = re.compile(
    r"(about|subscribe|donate|patreon|twitter|discord|index|home|table of contents|next|previous)",
    re.IGNORECASE,
)


def _strip_html(html: str) -> str:
    html = html.replace("<br />", "\n").replace("<br>", "\n").replace("</p>", "\n\n")
    text = TAG_RE.sub("", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&#8217;", "'")
        .replace("&#8220;", '"')
        .replace("&#8221;", '"')
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()) + "\n"


def _extract_chapter_links(toc_html: str, toc_url: str) -> list[str]:
    host = urlparse(toc_url).netloc
    seen: dict[str, None] = {}
    for m in ANCHOR_RE.finditer(toc_html):
        href = m.group(1)
        label = TAG_RE.sub("", m.group(2)).strip()
        if not href or href.startswith("#"):
            continue
        if SKIP_LINK_WORDS.search(label) and len(label) < 40:
            continue
        url = urljoin(toc_url, href)
        p = urlparse(url)
        if p.netloc and p.netloc != host:
            continue
        # Heuristic: WordPress chapter permalinks have a path with slashes and
        # are not the TOC itself.
        if url.rstrip("/") == toc_url.rstrip("/"):
            continue
        if not p.path or p.path == "/":
            continue
        seen.setdefault(url, None)
    return list(seen.keys())


def _extract_body(html: str) -> str:
    m = ENTRY_RE.search(html)
    if m:
        return _strip_html(m.group(1))
    m = ARTICLE_RE.search(html)
    if m:
        return _strip_html(m.group(1))
    return _strip_html(html)


def fetch(
    work_id: str,
    *,
    toc_url: Optional[str] = None,
    raw_root: Path = RAW_ROOT,
    sources_path: Path = SOURCES,
    http_get=get_text,
    max_chapters: Optional[int] = None,
) -> Path:
    if toc_url is None:
        cfg = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
        entry = cfg["works"].get(work_id) or {}
        toc_url = entry.get("toc_url")
        if not toc_url:
            raise FetchError(f"no wordpress toc_url for {work_id}")

    out_dir = raw_root / work_id
    meta_path = out_dir / "meta.json"
    if meta_path.is_file() and any(out_dir.glob("chap_*.txt")):
        log.info("already fetched: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching TOC: %s", toc_url)
    toc_html = http_get(toc_url)
    if not toc_html:
        raise FetchError(f"empty TOC for {work_id}")
    links = _extract_chapter_links(toc_html, toc_url)
    if not links:
        raise FetchError(f"no chapter links on TOC for {work_id}")
    if max_chapters:
        links = links[:max_chapters]
    log.info("found %d chapter links", len(links))

    chapters = []
    for i, url in enumerate(links, start=1):
        page = http_get(url)
        if not page:
            continue
        text = _extract_body(page)
        fname = f"chap_{i:04d}.txt"
        (out_dir / fname).write_text(text, encoding="utf-8")
        chapters.append({"index": i, "source_url": url, "file": fname})

    meta = {
        "work_id": work_id,
        "source": "wordpress",
        "toc_url": toc_url,
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00","Z"),
        "license": "author_public_web",
        "chapters": chapters,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="wordpress")
    ap.add_argument("work_id")
    ap.add_argument("--raw-root", default=str(RAW_ROOT))
    ap.add_argument("--max-chapters", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        fetch(
            args.work_id,
            raw_root=Path(args.raw_root),
            max_chapters=args.max_chapters,
        )
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
