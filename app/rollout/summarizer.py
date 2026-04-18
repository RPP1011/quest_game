"""Rollout chapter summarizer.

Generates extractive summaries from rollout prose for quick comparison
of divergence between profiles. Summaries are hierarchical:
per-chapter → per-rollout arc.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.world.schema import RolloutChapter, RolloutRun
from app.world.state_manager import WorldStateManager


@dataclass
class ChapterSummary:
    rollout_id: str
    chapter_index: int
    action: str
    opening_line: str
    closing_line: str
    word_count: int


@dataclass
class RolloutSummary:
    rollout_id: str
    profile_id: str
    chapters: list[ChapterSummary]


def _first_sentence(text: str, max_len: int = 200) -> str:
    for punct in (". ", "! ", "? "):
        idx = text.find(punct)
        if 0 < idx < max_len:
            return text[: idx + 1]
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def _last_sentence(text: str, max_len: int = 200) -> str:
    tail = text.rstrip()
    for punct in (". ", "! ", "? "):
        idx = tail.rfind(punct, max(0, len(tail) - max_len * 2))
        if idx > 0:
            start = tail.rfind(punct, max(0, idx - max_len * 2), idx)
            start = start + 2 if start > 0 else max(0, idx - max_len)
            return tail[start : idx + 1]
    return "…" + tail[-max_len:].split(" ", 1)[-1]


def summarize_rollout(
    sm: WorldStateManager, rollout_id: str,
) -> RolloutSummary:
    run = sm.get_rollout(rollout_id)
    chapters = sm.list_rollout_chapters(rollout_id)
    chapter_summaries = []
    for ch in sorted(chapters, key=lambda c: c.chapter_index):
        prose = ch.prose or ""
        chapter_summaries.append(ChapterSummary(
            rollout_id=rollout_id,
            chapter_index=ch.chapter_index,
            action=ch.player_action or "",
            opening_line=_first_sentence(prose),
            closing_line=_last_sentence(prose),
            word_count=len(prose.split()),
        ))
    return RolloutSummary(
        rollout_id=rollout_id,
        profile_id=run.profile_id,
        chapters=chapter_summaries,
    )


def format_comparison(
    summaries: list[RolloutSummary],
) -> str:
    """Format multiple rollout summaries side-by-side for comparison."""
    lines: list[str] = []
    max_chapters = max(len(s.chapters) for s in summaries)

    for ch_idx in range(1, max_chapters + 1):
        lines.append(f"\n{'='*60}")
        lines.append(f"CHAPTER {ch_idx}")
        lines.append(f"{'='*60}")
        for s in summaries:
            ch = next(
                (c for c in s.chapters if c.chapter_index == ch_idx), None,
            )
            lines.append(f"\n  [{s.profile_id.upper()}] (rollout {s.rollout_id})")
            if ch:
                lines.append(f"  Action:  {ch.action}")
                lines.append(f"  Opens:   {ch.opening_line}")
                lines.append(f"  Closes:  {ch.closing_line}")
                lines.append(f"  Words:   {ch.word_count}")
            else:
                lines.append("  (no chapter)")

    lines.append(f"\n{'='*60}")
    lines.append("ARC COMPARISON")
    lines.append(f"{'='*60}")
    for s in summaries:
        total_words = sum(c.word_count for c in s.chapters)
        actions = [c.action for c in s.chapters]
        lines.append(f"\n  [{s.profile_id.upper()}] {len(s.chapters)} chapters, {total_words} total words")
        lines.append(f"  Action arc: {' → '.join(a[:50] for a in actions)}")

    return "\n".join(lines)
