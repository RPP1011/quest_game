"""Day 10 stress-test personas.

Four simple personality archetypes that each pick a one-line write-in action
given the tail of the last committed narrative record. Deterministic:
no LLM call, no randomness — we cycle through personas so the 50-chapter
stress test is reproducible and isn't gated by the judge model.

Usage::

    from tools.stress_personas import PERSONAS, pick_action
    p = PERSONAS[update_number % len(PERSONAS)]
    action = pick_action(p, last_prose)

The personas are intentionally broad and reusable across genres; the picks
come from tiny template banks so the prose pipeline still gets semantic
variation across the 50 updates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StressPersona:
    id: str
    name: str
    # Action templates. ``{tail}`` is filled with a 1-word noun pulled
    # from the end of the last narrative record; empty tail ⇒ no-op fallback.
    templates: tuple[str, ...]


PERSONAS: tuple[StressPersona, ...] = (
    StressPersona(
        id="investigator",
        name="The Investigator",
        templates=(
            "I ask who was the last person to see them.",
            "I study the doorway, looking for which direction they went.",
            "I retrace the last minute — what did I miss the first time?",
            "I ask a pointed question about the cargo and watch for the flinch.",
            "I press for a name, not a description.",
            "I check the ledgers — entries and erasures both.",
            "I lay out what I already know and let the silence do the rest.",
            "I ask the same question three different ways.",
        ),
    ),
    StressPersona(
        id="intuitionist",
        name="The Intuitionist",
        templates=(
            "I stop talking and wait for whatever wants to come.",
            "I notice the detail that doesn't belong and let it sit.",
            "I follow the feeling, not the facts.",
            "I let the quiet thicken until someone else breaks it.",
            "I look at whose hands are shaking and pretend not to.",
            "I trust the hunch — it's been right before.",
            "I give them space enough to say the thing they're not saying.",
            "I listen for what the room is not saying.",
        ),
    ),
    StressPersona(
        id="bruiser",
        name="The Bruiser",
        templates=(
            "I close the distance and make the threat plain.",
            "I flip the table and see who moves first.",
            "I block the door and ask again, slower.",
            "I break the glass and keep talking like I didn't.",
            "I grab the nearest heavy thing and wait.",
            "I plant my feet and tell them the truth flat out.",
            "I call their bluff and raise the cost.",
            "I make it expensive to lie to me.",
        ),
    ),
    StressPersona(
        id="schemer",
        name="The Schemer",
        templates=(
            "I let them think I believe them and plant the trap.",
            "I share exactly enough to make them share more.",
            "I lie about what I know; see which lie they catch.",
            "I borrow their frame and turn it inside out.",
            "I pay them in small truths to buy a large one.",
            "I pretend to leave and double back.",
            "I set the two of them against each other with a single sentence.",
            "I withhold — wait until they fill the silence with something useful.",
        ),
    ),
)


def pick_action(persona: StressPersona, cycle_index: int) -> str:
    """Deterministically pick an action template from the persona.

    ``cycle_index`` cycles through templates so consecutive visits to the
    same persona (every 4th update) still vary the action. No LLM call.
    """
    t = persona.templates
    return t[cycle_index % len(t)]


def persona_for(update_number: int) -> tuple[StressPersona, int]:
    """Return (persona, cycle_index_within_persona) for a given update.

    Update 1 -> investigator[0]; update 2 -> intuitionist[0]; update 5 ->
    investigator[1]; etc. Cycling is intentional: we want each persona to
    touch every story phase (early, middle, late) to surface degradation
    that's specific to a tone (e.g. bruiser destabilising late-arc reveals).
    """
    p = PERSONAS[(update_number - 1) % len(PERSONAS)]
    cycle_index = (update_number - 1) // len(PERSONAS)
    return p, cycle_index


__all__ = ["StressPersona", "PERSONAS", "pick_action", "persona_for"]
