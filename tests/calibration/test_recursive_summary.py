"""Tests for app/calibration/recursive_summary.py."""
from __future__ import annotations

import pytest

from app.calibration.recursive_summary import recursive_summarize, _chunk


class _ShrinkingClient:
    """Returns a summary that is ~1/4 the length of the input chunk."""

    async def chat(self, *, messages, **kwargs) -> str:
        content = messages[0].content if hasattr(messages[0], "content") else messages[0]["content"]
        marker = "PASSAGE:\n<<<\n"
        start = content.find(marker) + len(marker)
        end = content.rfind(">>>")
        chunk = content[start:end].strip()
        # Compress by keeping the first quarter of the chunk.
        return chunk[: max(50, len(chunk) // 4)]


def test_chunk_overlap_ok():
    text = "x" * 1000
    chunks = _chunk(text, window=400, overlap=100)
    assert chunks[0] == "x" * 400
    assert len(chunks) >= 3
    # Every chunk but the last is full-window-sized.
    for c in chunks[:-1]:
        assert len(c) == 400


def test_chunk_single_window():
    assert _chunk("abc", window=1000, overlap=100) == ["abc"]


async def test_recursive_summary_compresses_fixture():
    # 50k-char fixture — repeat a dense paragraph.
    para = (
        "Anna walked into the ruined hall. The clock on the east wall had "
        "stopped at seven. Her brother, still unaccounted for, had been seen "
        "here two nights prior. Outside, the rain was picking up."
    )
    fixture = (para + "\n\n") * (50_000 // (len(para) + 2) + 1)
    fixture = fixture[:50_000]
    assert len(fixture) >= 50_000

    client = _ShrinkingClient()
    out = await recursive_summarize(
        fixture,
        client=client,
        target_chars=8_000,
        window_chars=16_000,
        overlap_chars=800,
        per_chunk_chars=1_600,
    )
    assert len(out) < 8_000


async def test_recursive_summary_passthrough_when_under_target():
    client = _ShrinkingClient()
    short = "already small"
    assert await recursive_summarize(short, client=client, target_chars=1000) == short
