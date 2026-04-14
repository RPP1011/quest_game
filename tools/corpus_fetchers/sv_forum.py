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
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import yaml
from bs4 import BeautifulSoup

from ._http import FetchError, get_text


RAW_ROOT = Path("data/calibration/raw")
SOURCES = Path(__file__).parent / "sources.yaml"

log = logging.getLogger("corpus_fetchers.sv_forum")

def _strip_html(html_fragment: str) -> str:
    """Extract text from an HTML fragment, preserving paragraph breaks."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p in soup.find_all(["p", "div"]):
        p.append("\n\n")
    text = soup.get_text()
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip()) + "\n"


def _threadmarks_url(thread_url: str) -> str:
    base = thread_url.rstrip("/")
    return base + "/threadmarks"


def _extract_post_links(html: str, base_url: str) -> list[str]:
    """Return ordered, deduped (page_url, post_anchor) pairs of threadmarked posts.

    SV threadmark entries are anchors inside `.structItem-title` with hrefs
    like ``/threads/slug.id/page-N#post-M``. We return full URLs including
    the ``#post-M`` fragment so the body extractor can find the specific
    post on a multi-post page.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: dict[str, None] = {}
    # Prefer threadmark entries in the structured list.
    anchors = soup.select(".structItem-title a")
    if not anchors:
        anchors = [a for a in soup.find_all("a", href=True)
                   if "/threads/" in a["href"] and "#post-" in a["href"]]
    for a in anchors:
        href = a.get("href", "")
        if "#post-" not in href or "/threads/" not in href:
            continue
        url = urljoin(base_url, href)
        seen.setdefault(url, None)
    return list(seen.keys())


def _extract_post_body(page_html: str, post_anchor: str | None = None) -> str:
    """Extract a specific post body from an SV post page.

    ``post_anchor`` is the fragment id without '#', e.g. "post-4925222".
    If given, scope to the <article> that houses that post; otherwise
    return the first message-body article on the page.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    target = None
    if post_anchor:
        # XF posts are wrapped in <article class="message" data-content="post-M">
        # or <article id="post-M"> depending on version.
        target = soup.find("article", attrs={"data-content": post_anchor})
        if target is None:
            target = soup.find(id=post_anchor)
        if target is not None:
            body = target.find(class_=lambda c: c and "message-body" in c)
            if body is None:
                body = target.find(class_=lambda c: c and "bbWrapper" in c)
            if body is not None:
                return str(body)
    article = soup.find("article", class_=lambda c: c and "message-body" in c)
    if article is None:
        article = soup.find("div", class_=lambda c: c and "bbWrapper" in c)
    return str(article) if article else page_html


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
        anchor = url.split("#", 1)[1] if "#" in url else None
        page_url = url.split("#", 1)[0]
        page = http_get(page_url)
        if not page:
            continue
        text = _strip_html(_extract_post_body(page, post_anchor=anchor))
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
