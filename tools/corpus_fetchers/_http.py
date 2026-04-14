"""Shared HTTP helpers for corpus fetchers.

Conservative defaults: 1 req/sec, 3 retries with exponential backoff on 5xx
and 429 (honoring Retry-After), clear User-Agent string. All fetchers route
through :func:`get_text` so tests can monkey-patch a single seam.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import httpx


USER_AGENT = "quest_game-calibration-fetcher/1.0 (local research)"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MIN_DELAY = 1.0
MAX_RETRIES = 3

log = logging.getLogger("corpus_fetchers")


class FetchError(RuntimeError):
    """Non-retryable or retries-exhausted fetch failure."""


SleepFn = Callable[[float], None]


def _backoff(attempt: int) -> float:
    # 1, 2, 4 seconds
    return float(2 ** attempt)


def get_text(
    url: str,
    *,
    client: Optional[httpx.Client] = None,
    sleep: SleepFn = time.sleep,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_retries: int = MAX_RETRIES,
    allow_404: bool = False,
) -> Optional[str]:
    """GET ``url`` as text with retry/backoff. Returns None if 404 & allow_404."""
    owned = client is None
    cli = client or httpx.Client(
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )
    try:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = _backoff(attempt)
                log.info("retry %s in %.1fs: %s", attempt, delay, url)
                sleep(delay)
            else:
                sleep(min_delay)
            try:
                resp = cli.get(url)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning("transport error for %s: %s", url, exc)
                continue
            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        sleep(float(ra))
                    except ValueError:
                        pass
                last_exc = FetchError(f"{resp.status_code} for {url}")
                continue
            if resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code} for {url}")
            return resp.text
        raise FetchError(f"retries exhausted for {url}: {last_exc}")
    finally:
        if owned:
            cli.close()
