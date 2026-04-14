"""Per-character voice-continuity retriever (Wave 4c — light variant).

Pools quoted dialogue attributed to a specific POV character so the writer
has concrete "past utterances" it can honor for voice continuity.

Attribution is heuristic: we look at committed ``NarrativeRecord`` rows
whose ``pov_character_id`` equals the target character and extract any
straight- or curly-quoted spans from the prose. This is noisy (some
quoted spans belong to other speakers) but cheap, and good enough to
remind the writer what that character has sounded like in the past.

The "heavy" path — an LLM post-pass that attributes every utterance to a
speaker — is deferred. See §3.6 of
``docs/superpowers/specs/2026-04-14-retrieval-layer-design.md``.

Cold start
----------
When the target character has not spoken yet (no committed records, or
records whose prose contains no matched quotes), the retriever falls
back to the character's stored ``Entity.data["voice"]["voice_samples"]``
(Gap G3 seed data) if present. Those rows carry ``source_id =
"voice/seed/<character_id>/<idx>"`` to distinguish them from
from-gameplay hits.
"""

from __future__ import annotations

import re
from typing import Any

from app.world.state_manager import WorldStateError, WorldStateManager

from .interface import Query, Result


# ---------------------------------------------------------------------------
# Dialogue extraction
# ---------------------------------------------------------------------------
#
# We match the dialogue body (non-greedy, no embedded matching quote) for
# each of the common quote-pair styles English prose uses:
#
#     "..."    straight double quotes
#     '...'    straight single quotes (rarer, noisy)
#     "..."    curly double quotes (U+201C / U+201D)
#     '...'    curly single quotes (U+2018 / U+2019)
#
# The length filter (5..400 chars) drops punctuation-only fragments and
# anything that looks like a block quote or a whole paragraph crammed
# between apostrophes.

_DIALOGUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'"([^"\n]{1,800})"'),
    re.compile(r"\u201C([^\u201C\u201D\n]{1,800})\u201D"),
    re.compile(r"\u2018([^\u2018\u2019\n]{1,800})\u2019"),
    # Straight single-quote apostrophes are ambiguous with possessives
    # and contractions; we match only sequences that look like a spoken
    # line (at least two words).
    re.compile(r"'([^'\n]{5,800})'"),
)


_MIN_QUOTE_LEN = 5
_MAX_QUOTE_LEN = 400


class VoiceRetriever:
    """Serve past utterances by a single POV character.

    Parameters
    ----------
    world:
        The :class:`WorldStateManager` backing the current quest. Used
        read-only via ``list_narrative`` and ``get_entity``.
    quest_id:
        The quest scope — purely for ``source_id`` namespacing. The
        retriever does not filter narrative rows by quest because
        ``WorldStateManager`` is already quest-scoped at construction.
    """

    def __init__(self, world: WorldStateManager, quest_id: str) -> None:
        self._world = world
        self._quest_id = quest_id

    # -- Quote extraction ----------------------------------------------

    @staticmethod
    def _extract_quoted_lines(text: str) -> list[str]:
        """Return every quoted span in ``text`` that fits the length window.

        Deduplicated case-sensitively, order-preserving (first occurrence
        wins). Handles straight/curly double- and single-quote pairs.
        """
        if not text:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for pat in _DIALOGUE_PATTERNS:
            for m in pat.finditer(text):
                body = m.group(1).strip()
                length = len(body)
                if length < _MIN_QUOTE_LEN or length > _MAX_QUOTE_LEN:
                    continue
                if body in seen:
                    continue
                seen.add(body)
                out.append(body)
        return out

    # -- Cold-start fallback -------------------------------------------

    def _seed_voice_samples(self, character_id: str) -> list[str]:
        """Read ``Entity.data["voice"]["voice_samples"]`` for ``character_id``.

        Returns ``[]`` if the entity is missing, not a character, or
        carries no voice samples.
        """
        try:
            entity = self._world.get_entity(character_id)
        except WorldStateError:
            return []
        voice = entity.data.get("voice") if isinstance(entity.data, dict) else None
        if not isinstance(voice, dict):
            return []
        samples = voice.get("voice_samples")
        if not isinstance(samples, list):
            return []
        return [s for s in samples if isinstance(s, str) and s.strip()]

    # -- Retrieval -----------------------------------------------------

    async def retrieve(self, query: Query, *, k: int = 3) -> list[Result]:
        filters: dict[str, Any] = dict(query.filters or {})
        character_id = filters.get("character_id")
        if not isinstance(character_id, str) or not character_id:
            return []

        last_n = filters.get("last_n_records")
        try:
            last_n_int = int(last_n) if last_n is not None else 30
        except (TypeError, ValueError):
            last_n_int = 30

        # ``list_narrative`` returns oldest-first; flip for recency and
        # cap to the last N records per spec (default 30).
        all_records = self._world.list_narrative(limit=10_000)
        records = list(reversed(all_records))[: max(last_n_int, 0)]

        # Collect quoted lines from records POV'd by this character.
        # Newest records first so caller's ``most-recent`` ordering is
        # implicit; within a record we preserve the in-text position.
        pooled: list[tuple[str, dict[str, Any]]] = []
        for rec in records:
            if rec.pov_character_id != character_id:
                continue
            quotes = self._extract_quoted_lines(rec.raw_text or "")
            for position, line in enumerate(quotes):
                meta: dict[str, Any] = {
                    "character_id": character_id,
                    "source_update_number": int(rec.update_number),
                    "position": position,
                }
                # ``scene_index`` is not tracked at the narrative-record
                # level; surface only when a caller has been nice enough
                # to set it on ``state_diff`` (future-proof hook).
                diff = rec.state_diff if isinstance(rec.state_diff, dict) else {}
                scene_idx = diff.get("scene_index")
                if isinstance(scene_idx, int):
                    meta["scene_index"] = scene_idx
                pooled.append((line, meta))

        if pooled:
            results: list[Result] = []
            for idx, (line, meta) in enumerate(pooled[: max(int(k), 0)]):
                results.append(
                    Result(
                        source_id=f"voice/{self._quest_id}/{character_id}/{idx}",
                        text=line,
                        score=1.0 - (idx * 0.01),
                        metadata=meta,
                    )
                )
            return results

        # Cold start: fall back to the character's entity-stored samples.
        seeds = self._seed_voice_samples(character_id)
        if not seeds:
            return []
        results = []
        for idx, sample in enumerate(seeds[: max(int(k), 0)]):
            results.append(
                Result(
                    source_id=f"voice/seed/{character_id}/{idx}",
                    text=sample,
                    score=1.0 - (idx * 0.01),
                    metadata={
                        "character_id": character_id,
                        "seed": True,
                    },
                )
            )
        return results
