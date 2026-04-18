"""Per-character metaphor voice tracker.

Each POV character has a signature imagery family (derived from the
seed's voice config) with a target frequency. The tracker maintains a
ring buffer of per-chapter, per-character family counts and produces:

1. Writer guidance: which families to use, which to avoid, how many of
   the signature family to deploy
2. Critic input: whether the chapter exceeded the signature target or
   lacked variety
3. Cross-chapter context: cumulative family usage for the scorer's
   metaphor_variety rubric

The signature family is a *controlled character trait*, not an emergent
tic. Tristan narrates in gambling metaphors (2-3/chapter). Angharad
thinks in honor/weight/duty imagery. When POV switches, the entire
imagery register should shift.

All imagery classification uses the LLM classifier (no keyword matching).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.world.schema import Entity


# Map seed signature_tic descriptions to imagery family names.
_SIGNATURE_MAP: dict[str, str] = {
    "gambling": "gambling",
    "gamble": "gambling",
    "odds": "gambling",
    "predator": "predator_prey",
    "prey": "predator_prey",
    "hunt": "predator_prey",
    "water": "water_ocean",
    "ocean": "water_ocean",
    "tide": "water_ocean",
    "mechanical": "mechanical",
    "clockwork": "mechanical",
    "gear": "mechanical",
    "fire": "fire_light",
    "flame": "fire_light",
    "light": "fire_light",
    "weight": "weight_gravity",
    "gravity": "weight_gravity",
    "burden": "weight_gravity",
    "honor": "weight_gravity",
    "duty": "weight_gravity",
}


def detect_signature_family(entity: Entity) -> str | None:
    """Detect the signature imagery family from an entity's voice config."""
    data = getattr(entity, "data", None) or {}
    voice = data.get("voice") or getattr(entity, "voice", None)
    if not voice:
        # Also check worldview in data
        worldview = data.get("worldview", "")
        if worldview:
            voice = {"worldview": worldview}
        else:
            return None
    voice_str = str(voice).lower()
    for keyword, family in _SIGNATURE_MAP.items():
        if keyword in voice_str:
            return family
    return None


@dataclass
class ChapterVoiceSnapshot:
    chapter_index: int
    pov_character_id: str
    family_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class CharacterVoiceTracker:
    """Per-character imagery ring buffer.

    Usage:
        tracker = CharacterVoiceTracker.from_entities(entities)
        snapshot = await tracker.record_chapter(client, chapter_index, pov_id, prose)
        guidance = tracker.get_writer_guidance(pov_id)
        context = tracker.get_critic_context()

    Call ``record_chapter`` after each chapter is written. Call
    ``get_writer_guidance`` before each chapter to inform the writer.
    """
    signatures: dict[str, str | None] = field(default_factory=dict)
    history: list[ChapterVoiceSnapshot] = field(default_factory=list)
    max_history: int = 10
    signature_target: int = 3

    @classmethod
    def from_entities(
        cls, entities: list[Entity], max_history: int = 10,
        signature_target: int = 3,
    ) -> CharacterVoiceTracker:
        sigs: dict[str, str | None] = {}
        for e in entities:
            if e.entity_type.value == "character":
                sigs[e.id] = detect_signature_family(e)
        return cls(
            signatures=sigs, max_history=max_history,
            signature_target=signature_target,
        )

    async def record_chapter(
        self, client: Any, chapter_index: int,
        pov_character_id: str, prose: str,
    ) -> ChapterVoiceSnapshot:
        """Classify imagery families in the prose via LLM and add to ring buffer."""
        counts: dict[str, int] = {}
        try:
            from app.planning.metaphor_critic import classify_metaphors_llm
            classification = await classify_metaphors_llm(client, prose)
            for fam, data in classification.get("families", {}).items():
                c = data.get("count", 0)
                if c > 0:
                    counts[fam] = c
        except Exception:
            pass  # empty counts on failure

        snap = ChapterVoiceSnapshot(
            chapter_index=chapter_index,
            pov_character_id=pov_character_id,
            family_counts=counts,
        )
        self.history.append(snap)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return snap

    def _cumulative_for_character(self, character_id: str) -> dict[str, int]:
        """Sum family counts across all buffered chapters for this character."""
        totals: dict[str, int] = defaultdict(int)
        for snap in self.history:
            if snap.pov_character_id == character_id:
                for fam, count in snap.family_counts.items():
                    totals[fam] += count
        return dict(totals)

    def _cumulative_all(self) -> dict[str, int]:
        """Sum family counts across all buffered chapters, all characters."""
        totals: dict[str, int] = defaultdict(int)
        for snap in self.history:
            for fam, count in snap.family_counts.items():
                totals[fam] += count
        return dict(totals)

    def get_writer_guidance(self, pov_character_id: str) -> str:
        """Produce a guidance block for the writer prompt.

        Shows signature family usage vs target, warns about overuse.
        """
        sig = self.signatures.get(pov_character_id)
        cumulative = self._cumulative_for_character(pov_character_id)

        lines: list[str] = ["## Imagery register guidance"]

        if sig:
            sig_total = cumulative.get(sig, 0)
            n_chapters = sum(
                1 for s in self.history
                if s.pov_character_id == pov_character_id
            )
            if n_chapters > 0:
                avg = sig_total / n_chapters
                lines.append(
                    f"Signature family '{sig}': averaging {avg:.1f}/chapter "
                    f"(target: {self.signature_target})."
                )
                if avg > self.signature_target + 1:
                    lines.append(
                        f"WARNING: '{sig}' is running high. Pull back to "
                        f"{self.signature_target}/chapter."
                    )
            else:
                lines.append(
                    f"Signature family '{sig}': target {self.signature_target}/chapter."
                )

        # Top overused non-signature families
        for fam, total in sorted(cumulative.items(), key=lambda x: -x[1]):
            if fam == sig:
                continue
            n_chapters = sum(
                1 for s in self.history
                if s.pov_character_id == pov_character_id
            )
            if n_chapters > 0 and total / n_chapters > 4:
                lines.append(
                    f"Non-signature family '{fam}' averaging "
                    f"{total / n_chapters:.1f}/chapter — reduce."
                )

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def get_critic_context(self) -> dict[str, int]:
        """Cumulative family counts across all chapters for scorer context.

        This feeds into the metaphor_variety scoring rubric as
        ``prior_chapter_families``.
        """
        return self._cumulative_all()

    async def get_chapter_report(
        self, client: Any, chapter_index: int,
        pov_character_id: str, prose: str,
    ) -> dict:
        """Analyze a chapter and return a structured report.

        Includes: per-family counts, signature usage vs target,
        variety assessment.
        """
        counts: dict[str, int] = {}
        try:
            from app.planning.metaphor_critic import classify_metaphors_llm
            classification = await classify_metaphors_llm(client, prose)
            for fam, data in classification.get("families", {}).items():
                c = data.get("count", 0)
                if c > 0:
                    counts[fam] = c
        except Exception:
            pass

        sig = self.signatures.get(pov_character_id)
        sig_count = counts.get(sig, 0) if sig else 0
        total_imagery = sum(counts.values())
        n_families_used = len(counts)

        return {
            "pov_character_id": pov_character_id,
            "signature_family": sig,
            "signature_count": sig_count,
            "signature_target": self.signature_target,
            "total_imagery": total_imagery,
            "families_used": n_families_used,
            "family_counts": counts,
            "variety": "good" if n_families_used >= 3 else "low",
        }
