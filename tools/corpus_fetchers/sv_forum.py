"""Sufficient Velocity quest-thread fetcher.

SV renders an HTML ``threadmarks`` page per thread listing chapter posts.
Each post has a permalink anchor like ``/threads/foo.1234/post-98765``. We
parse anchors under the threadmarks container, dedupe, fetch each post page,
and extract the post body text.

Because SV's exact HTML can drift and some threads disable public threadmark
views, the parser is best-effort and logs how many posts it found.
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
from urllib.parse import urljoin

import yaml

from ._http import FetchError, get_text


RAW_ROOT = Path("data/calibration/raw")
SOURCES = Path(__file__).parent / "sources.yaml"

log = logging.getLogger("corpus_fetchers.sv_forum")

# Very lax: match href tokens that look like threadmark post links.
POST_HREF_RE = re.compile(
    r'href="(/threads/[^"#]+/threadmarks[^"]*)"|'
    r'href="(/threads/[^"]+/post-\d+)"',
    re.IGNORECASE,
)
POST_BODY_RE = re.compile(
    r'<article[^>]*class="[^"]*message-body[^"]*"[^>]*>(.*?)</article>',
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    # Decode a few common entities and strip tags.
    html = html.replace("<br />", "\n").replace("<br>", "\n").replace("</p>", "\n\n")
    text = TAG_RE.sub("", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()) + "\n"


def _threadmarks_url(thread_url: str) -> str:
    base = thread_url.rstrip("/")
    return base + "/threadmarks"


def _extract_post_links(html: str, base_url: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in POST_HREF_RE.finditer(html):
        path = m.group(1) or m.group(2)
        if not path or "/threadmarks" in path:
            continue
        url = urljoin(base_url, path)
        seen.setdefault(url, None)
    return list(seen.keys())


def fetch(
    work_id: str,
    *,
    thread_url: Optional[str] = None,
    raw_root: Path = RAW_ROOT,
    sources_path: Path = SOURCES,
    http_get=get_text,
    max_chapters: Optional[int] = None,
) -> Path:
    if thread_url is None:
        cfg = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
        entry = cfg["works"].get(work_id) or {}
        thread_url = entry.get("thread_url")
        if not thread_url:
            raise FetchError(f"no sv_forum thread_url for {work_id}")

    out_dir = raw_root / work_id
    meta_path = out_dir / "meta.json"
    if meta_path.is_file() and any(out_dir.glob("chap_*.txt")):
        log.info("already fetched: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tm_url = _threadmarks_url(thread_url)
    log.info("fetching threadmarks: %s", tm_url)
    tm_html = http_get(tm_url)
    if tm_html is None:
        raise FetchError(f"threadmarks page empty for {work_id}")
    links = _extract_post_links(tm_html, thread_url)
    if not links:
        raise FetchError(
            f"no threadmark post links found for {work_id}; "
            "SV may have disabled public threadmarks or changed markup"
        )
    if max_chapters:
        links = links[:max_chapters]
    log.info("found %d chapter posts", len(links))

    chapters = []
    for i, url in enumerate(links, start=1):
        page = http_get(url)
        if not page:
            continue
        body_match = POST_BODY_RE.search(page)
        body = body_match.group(1) if body_match else page
        text = _strip_html(body)
        fname = f"chap_{i:04d}.txt"
        (out_dir / fname).write_text(text, encoding="utf-8")
        chapters.append({"index": i, "source_url": url, "file": fname})

    meta = {
        "work_id": work_id,
        "source": "sv_forum",
        "thread_url": thread_url,
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00","Z"),
        "license": "fan_work_public_web",
        "chapters": chapters,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="sv_forum")
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
