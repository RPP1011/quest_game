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
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.planning.metaphor_critic import IMAGERY_FAMILIES, _count_family
from app.world.schema import Entity


# Map seed signature_tic descriptions to imagery family names.
# The seed says things like "narrates to himself in gambling metaphors"
# — we map that to the "gambling" family key.
_TIC_TO_FAMILY: dict[str, str] = {
    "gambling": "gambling",
    "gamble": "gambling",
    "odds": "gambling",
    "dice": "gambling",
    "bet": "gambling",
    "honor": "weight_gravity",
    "duty": "weight_gravity",
    "predator": "predator_prey",
    "hunt": "predator_prey",
    "prey": "predator_prey",
    "water": "water_ocean",
    "sea": "water_ocean",
    "ocean": "water_ocean",
    "fish": "water_ocean",
    "mechanical": "mechanical",
    "clock": "mechanical",
    "gear": "mechanical",
    "fire": "fire_light",
    "flame": "fire_light",
    "light": "fire_light",
}


def detect_signature_family(entity: Entity) -> str | None:
    """Infer the character's signature imagery family from their seed data.

    Checks ``data.voice.signature_tics`` and ``data.worldview`` for
    keywords that map to a known family. Returns the family name or
    None if no clear signal.
    """
    voice = entity.data.get("voice", {})
    tics = voice.get("signature_tics", []) if isinstance(voice, dict) else []
    worldview = entity.data.get("worldview", "")

    text = " ".join(tics) + " " + worldview
    text_lower = text.lower()

    # Count keyword hits per family
    hits: dict[str, int] = defaultdict(int)
    for keyword, family in _TIC_TO_FAMILY.items():
        if keyword in text_lower:
            hits[family] += 1

    if not hits:
        return None
    return max(hits, key=hits.get)


@dataclass
class ChapterVoiceSnapshot:
    """One chapter's metaphor family counts for a specific POV character."""
    chapter_index: int
    pov_character_id: str
    family_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class CharacterVoiceTracker:
    """Ring buffer of per-chapter metaphor usage, per POV character.

    Call ``record_chapter`` after each chapter is written. Call
    ``get_writer_guidance`` before each chapter to get the instruction
    block for the writer prompt. Call ``get_critic_context`` to get
    cross-chapter data for the metaphor_variety scorer.
    """
    # {character_id: signature_family_name}
    signatures: dict[str, str | None] = field(default_factory=dict)
    # Ring buffer: last N chapters
    history: list[ChapterVoiceSnapshot] = field(default_factory=list)
    max_history: int = 10
    # Target frequency for the signature family per chapter
    signature_target: int = 3

    @classmethod
    def from_entities(
        cls, entities: list[Entity], *, max_history: int = 10,
        signature_target: int = 3,
    ) -> "CharacterVoiceTracker":
        """Build a tracker from seeded character entities."""
        sigs: dict[str, str | None] = {}
        for e in entities:
            if e.entity_type.value == "character":
                sigs[e.id] = detect_signature_family(e)
        return cls(
            signatures=sigs, max_history=max_history,
            signature_target=signature_target,
        )

    def record_chapter(
        self, chapter_index: int, pov_character_id: str, prose: str,
    ) -> ChapterVoiceSnapshot:
        """Count imagery families in the prose and add to the ring buffer."""
        counts: dict[str, int] = {}
        for family_name, phrases in IMAGERY_FAMILIES.items():
            c = _count_family(prose, phrases)
            if c > 0:
                counts[family_name] = c
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
        """Generate a writer-prompt block for the upcoming chapter's POV.

        Tells the writer:
        - What the character's signature family is + target frequency
        - Which families have been overused across recent chapters
        - Which families to reach for instead
        """
        sig = self.signatures.get(pov_character_id)
        cumulative = self._cumulative_for_character(pov_character_id)
        all_families = set(IMAGERY_FAMILIES.keys())

        lines: list[str] = ["## Imagery register guidance"]

        if sig:
            lines.append(
                f"\nThis POV character's signature imagery is **{sig}**. "
                f"Use it {self.signature_target}× this chapter — enough to "
                f"be a recognizable voice trait, not enough to dominate."
            )
        else:
            lines.append(
                "\nThis POV character has no signature imagery family. "
                "Draw from multiple registers."
            )

        # Identify overused families (>5× per chapter average in buffer)
        n_chapters = sum(
            1 for s in self.history
            if s.pov_character_id == pov_character_id
        )
        overused: list[str] = []
        if n_chapters > 0:
            for fam, total in sorted(cumulative.items(), key=lambda x: -x[1]):
                avg = total / n_chapters
                if avg > 5 and fam != sig:
                    overused.append(f"{fam} ({avg:.0f}/ch avg)")
                elif fam == sig and avg > self.signature_target * 2:
                    overused.append(
                        f"{fam} ({avg:.0f}/ch avg — target is {self.signature_target})"
                    )

        if overused:
            lines.append(
                f"\n**Overused in recent chapters** (reduce these): "
                + ", ".join(overused)
            )

        # Suggest underused families
        used_families = set(cumulative.keys())
        unused = all_families - used_families - {sig or ""}
        if unused:
            lines.append(
                f"\n**Underused registers to draw from**: "
                + ", ".join(sorted(unused))
            )

        # Other characters' signatures (avoid these in this POV)
        other_sigs = {
            cid: fam for cid, fam in self.signatures.items()
            if cid != pov_character_id and fam and fam != sig
        }
        if other_sigs:
            avoid = ", ".join(
                f"{fam} ({cid})" for cid, fam in other_sigs.items()
            )
            lines.append(
                f"\n**Other characters' signatures** (avoid in this POV): "
                + avoid
            )

        return "\n".join(lines)

    def get_critic_context(self) -> dict[str, int]:
        """Return cumulative family counts across all recent chapters.

        Used by the metaphor_variety scorer as cross-chapter context.
        """
        return self._cumulative_all()

    def get_chapter_report(
        self, chapter_index: int, pov_character_id: str, prose: str,
    ) -> dict:
        """Analyze a chapter and return a structured report.

        Includes: per-family counts, signature usage vs target,
        variety assessment.
        """
        counts: dict[str, int] = {}
        for family_name, phrases in IMAGERY_FAMILIES.items():
            c = _count_family(prose, phrases)
            if c > 0:
                counts[family_name] = c

        sig = self.signatures.get(pov_character_id)
        sig_count = counts.get(sig, 0) if sig else 0
        total_imagery = sum(counts.values())
        n_families_used = len(counts)

        return {
            "pov_character_id": pov_character_id,
            "signature_family": sig,
            "signature_count": sig_count,
            "signature_target": self.signature_target,
            "signature_over": sig_count > self.signature_target * 2,
            "family_counts": counts,
            "total_imagery": total_imagery,
            "families_used": n_families_used,
            "variety_ok": n_families_used >= 3,
        }
