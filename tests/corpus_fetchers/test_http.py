from __future__ import annotations

import httpx
import pytest

from tools.corpus_fetchers import _http
from tools.corpus_fetchers._http import FetchError, get_text


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, d: float) -> None:
        self.calls.append(d)


def _client(handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, headers={"User-Agent": _http.USER_AGENT})


def test_get_text_happy() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        assert req.headers["user-agent"].startswith("quest_game-calibration-fetcher")
        return httpx.Response(200, text="hello")

    sleep = FakeSleep()
    out = get_text("https://x/y", client=_client(h), sleep=sleep, min_delay=0.0)
    assert out == "hello"
    assert sleep.calls == [0.0]


def test_get_text_404_allowed() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    out = get_text(
        "https://x/y",
        client=_client(h),
        sleep=FakeSleep(),
        min_delay=0.0,
        allow_404=True,
    )
    assert out is None


def test_get_text_429_retry_then_ok() -> None:
    calls = {"n": 0}

    def h(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow")
        return httpx.Response(200, text="ok")

    out = get_text(
        "https://x/y", client=_client(h), sleep=FakeSleep(), min_delay=0.0
    )
    assert out == "ok"
    assert calls["n"] == 2


def test_get_text_5xx_exhausts() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(FetchError):
        get_text(
            "https://x/y",
            client=_client(h),
            sleep=FakeSleep(),
            min_delay=0.0,
            max_retries=2,
        )
