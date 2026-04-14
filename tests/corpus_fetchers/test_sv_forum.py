from __future__ import annotations

from pathlib import Path

from tools.corpus_fetchers import sv_forum


THREADMARKS_HTML = """
<html><body>
<div class="structItemContainer">
<div class="structItem-title"><a href="/threads/foo.1/#post-100">Chapter 1</a></div>
<div class="structItem-title"><a href="/threads/foo.1/page-3#post-200">Chapter 2</a></div>
<div class="structItem-title"><a href="/threads/foo.1/#post-100">(dup)</a></div>
<div class="structItem-title"><a href="/threads/foo.1/threadmarks?category_id=2">Informational</a></div>
</div>
</body></html>
"""

POST_HTML_PAGE1 = """
<html><body>
<article class="message" data-content="post-100">
<div class="message-body">
<div class="bbWrapper"><p>Post body line one.</p><p>"Hi," she said.</p></div>
</div>
</article>
<article class="message" data-content="post-101">
<div class="message-body"><div class="bbWrapper"><p>Different reply.</p></div></div>
</article>
</body></html>
"""

POST_HTML_PAGE3 = """
<html><body>
<article class="message" data-content="post-200">
<div class="message-body">
<div class="bbWrapper"><p>Second chapter body.</p></div>
</div>
</article>
</body></html>
"""


def test_extract_post_links() -> None:
    links = sv_forum._extract_post_links(
        THREADMARKS_HTML, "https://forums.sufficientvelocity.com/threads/foo.1/"
    )
    assert len(links) == 2
    assert all("#post-" in u for u in links)


def test_fetch(tmp_path: Path) -> None:
    thread = "https://forums.sufficientvelocity.com/threads/foo.1/"
    tm_url = thread + "threadmarks"
    responses = {
        tm_url: THREADMARKS_HTML,
        "https://forums.sufficientvelocity.com/threads/foo.1/": POST_HTML_PAGE1,
        "https://forums.sufficientvelocity.com/threads/foo.1/page-3": POST_HTML_PAGE3,
    }

    def http_get(url: str, *, allow_404: bool = False):
        return responses[url]

    out = sv_forum.fetch(
        "marked_for_death",
        thread_url=thread,
        raw_root=tmp_path,
        http_get=http_get,
    )
    assert (out / "chap_0001.txt").is_file()
    assert (out / "chap_0002.txt").is_file()
    body1 = (out / "chap_0001.txt").read_text()
    body2 = (out / "chap_0002.txt").read_text()
    assert "Post body line one" in body1
    assert "Different reply" not in body1
    assert "Second chapter body" in body2
    assert "<article" not in body1
