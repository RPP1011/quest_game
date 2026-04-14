from __future__ import annotations

import json
from pathlib import Path

from tools.corpus_fetchers import wordpress


TOC = """
<html><body>
<h1>Table of Contents</h1>
<a href="/2015/03/25/prologue/">Prologue</a>
<a href="/2015/04/01/chapter-1/">Chapter 1</a>
<a href="/2015/04/08/chapter-2/">Chapter 2</a>
<a href="https://twitter.com/me">Twitter</a>
<a href="#top">Back to top</a>
</body></html>
"""

CHAPTER_HTML = """
<html><body>
<article>
<div class="entry-content">
<p>Chapter body paragraph one.</p>
<p>Dialogue: "Hello," she said.</p>
</div><!-- .entry-content -->
</article>
</body></html>
"""


def test_extract_links() -> None:
    links = wordpress._extract_chapter_links(
        TOC, "https://practicalguidetoevil.wordpress.com/table-of-contents/"
    )
    assert len(links) == 3
    assert all("wordpress.com" in u for u in links)


def test_fetch_end_to_end(tmp_path: Path) -> None:
    base = "https://practicalguidetoevil.wordpress.com/table-of-contents/"
    responses: dict[str, str] = {base: TOC}
    for url in wordpress._extract_chapter_links(TOC, base):
        responses[url] = CHAPTER_HTML

    calls: list[str] = []

    def http_get(url: str, *, allow_404: bool = False):
        calls.append(url)
        return responses[url]

    out_dir = wordpress.fetch(
        "practical_guide_evil",
        toc_url=base,
        raw_root=tmp_path,
        http_get=http_get,
    )
    assert (out_dir / "chap_0001.txt").is_file()
    assert (out_dir / "chap_0003.txt").is_file()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert len(meta["chapters"]) == 3
    body = (out_dir / "chap_0001.txt").read_text()
    assert "Chapter body paragraph" in body
    assert "<p>" not in body  # stripped
