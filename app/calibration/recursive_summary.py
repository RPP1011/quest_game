"""Recursive MapReduce summarization for over-long scenes.

If a scene exceeds the model's context budget, chunk it into overlapping
windows, summarize each with the judge client, concatenate the summaries,
and recurse until the concatenated text is under ``target_chars``.

Algorithm:
    1. Split the text into fixed-char windows with a small overlap so that
       beats on window boundaries survive.
    2. For each window, request a compact summary via the chat client.
    3. Concatenate the summaries in original order.
    4. If the result is still over budget, recurse.

This is forward-compat for LFM-at-32k. Wire it into ``arc_scorer.py`` only
when the raw scene exceeds the configured character budget — most calibration
scenes sit well under 32k tokens and never trigger summarization.
"""
from __future__ import annotations

import logging
from typing import Protocol


log = logging.getLogger("recursive_summary")


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


DEFAULT_TARGET_CHARS = 8_000
DEFAULT_WINDOW_CHARS = 16_000
DEFAULT_OVERLAP_CHARS = 800
DEFAULT_SUMMARY_CHARS = 1_600  # per-window compression target


_SUMMARY_PROMPT = """\
Summarize the passage below into at most {max_chars} characters of dense
prose. Preserve: named characters and their states, location, active
conflicts and threads, any promises/threats, and the final beat of the
passage. Omit: scenic filler, restated backstory, meta commentary. Do not
introduce facts not in the passage.

PASSAGE:
<<<
{chunk}
>>>

SUMMARY:
"""


def _chunk(text: str, window: int, overlap: int) -> list[str]:
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must be in [0, window)")
    if len(text) <= window:
        return [text]
    chunks: list[str] = []
    step = window - overlap
    i = 0
    while i < len(text):
        chunks.append(text[i : i + window])
        if i + window >= len(text):
            break
        i += step
    return chunks


async def _summarize_chunk(
    chunk: str, *, client: ChatLike, max_chars: int,
) -> str:
    prompt = _SUMMARY_PROMPT.format(max_chars=max_chars, chunk=chunk)
    try:
        from app.runtime.client import ChatMessage  # type: ignore
        msg = ChatMessage(role="user", content=prompt)
    except Exception:  # pragma: no cover
        msg = {"role": "user", "content": prompt}
    raw = await client.chat(
        messages=[msg],
        temperature=0.2,
        max_tokens=max(256, max_chars // 3),
        thinking=False,
    )
    return raw.strip()


async def recursive_summarize(
    text: str,
    *,
    client: ChatLike,
    target_chars: int = DEFAULT_TARGET_CHARS,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    per_chunk_chars: int = DEFAULT_SUMMARY_CHARS,
    max_depth: int = 4,
) -> str:
    """Compress ``text`` to under ``target_chars``.

    Recurses at most ``max_depth`` times. If the shrink ratio stalls, the
    final pass truncates to ``target_chars`` to guarantee termination.
    """
    depth = 0
    current = text
    while len(current) > target_chars and depth < max_depth:
        chunks = _chunk(current, window_chars, overlap_chars)
        summaries: list[str] = []
        for i, c in enumerate(chunks):
            s = await _summarize_chunk(
                c, client=client, max_chars=per_chunk_chars,
            )
            summaries.append(s)
            log.debug(
                "depth=%d chunk=%d/%d in=%d out=%d",
                depth, i + 1, len(chunks), len(c), len(s),
            )
        nxt = "\n\n".join(summaries).strip()
        if len(nxt) >= len(current):
            # No progress — hard truncate.
            log.warning("recursive_summary stalled at depth=%d; truncating", depth)
            return nxt[:target_chars]
        current = nxt
        depth += 1
    if len(current) > target_chars:
        return current[:target_chars]
    return current
