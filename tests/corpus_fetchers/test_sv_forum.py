from __future__ import annotations

from pathlib import Path

from tools.corpus_fetchers import sv_forum


THREADMARKS_HTML = """
<html><body>
<ul class="structItemContainer">
<li><a href="/threads/foo.1/post-100">Chapter 1</a></li>
<li><a href="/threads/foo.1/post-200">Chapter 2</a></li>
<li><a href="/threads/foo.1/post-100">(dup)</a></li>
<li><a href="/threads/foo.1/threadmarks?category_id=2">Informational</a></li>
</ul>
</body></html>
"""

POST_HTML = """
<html><body>
<article class="message-body js-selectToQuote">
<p>Post body line one.</p>
<p>"Hi," she said.</p>
</article>
</body></html>
"""


def test_extract_post_links() -> None:
    links = sv_forum._extract_post_links(
        THREADMARKS_HTML, "https://forums.sufficientvelocity.com/threads/foo.1/"
    )
    assert len(links) == 2
    assert all("/post-" in u for u in links)


def test_fetch(tmp_path: Path) -> None:
    thread = "https://forums.sufficientvelocity.com/threads/foo.1/"
    tm_url = thread + "threadmarks"
    responses = {tm_url: THREADMARKS_HTML}
    for u in sv_forum._extract_post_links(THREADMARKS_HTML, thread):
        responses[u] = POST_HTML

    def http_get(url: str, *, allow_404: bool = False):
        return responses[url]

    out = sv_forum.fetch(
        "marked_for_death",
        thread_url=thread,
        raw_root=tmp_path,
        http_get=http_get,
    )
    assert (out / "chap_0001.txt").is_file()
    body = (out / "chap_0001.txt").read_text()
    assert "Post body line one" in body
    assert "<article" not in body
