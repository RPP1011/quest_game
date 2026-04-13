"""Information-asymmetry tracking (Gap G7).

Mutates persistent ``InformationState`` rows post-commit from a dramatic
plan's ``DramaticScene.reveals`` / ``.withholds``, and derives in-memory
``InformationAsymmetry`` views (dramatic irony, mystery, secret, false
belief) for use by the dramatic planner and tool scorer.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from app.planning.schemas import DramaticPlan, DramaticScene
from app.world.schema import (
    AsymmetryKind,
    EntityType,
    InformationAsymmetry,
    InformationState,
    KnowledgeEntry,
)
from app.world.state_manager import WorldStateManager


READER = "reader"
NARRATOR = "narrator"


def _fact_id(quest_id: str, fact: str) -> str:
    digest = hashlib.sha1(f"{quest_id}:{fact.strip().lower()}".encode()).hexdigest()[:10]
    return f"fact_{digest}"


def _find_matching_fact(
    existing: list[InformationState], fact_text: str
) -> InformationState | None:
    """Case-insensitive substring match in either direction.

    Matches on equality first, then substring — mirrors the loose matcher used
    for reader-model questions/expectations.
    """
    needle = fact_text.strip().lower()
    if not needle:
        return None
    for st in existing:
        hay = st.fact.strip().lower()
        if hay == needle:
            return st
    for st in existing:
        hay = st.fact.strip().lower()
        if needle in hay or hay in needle:
            return st
    return None


def _pov_entities(scene: DramaticScene) -> list[str]:
    """POV + on-screen characters for a scene."""
    out: list[str] = []
    if scene.pov_character_id:
        out.append(scene.pov_character_id)
    for cid in scene.characters_present:
        if cid not in out:
            out.append(cid)
    return out


def apply_dramatic_plan_reveals(
    *,
    world: WorldStateManager,
    quest_id: str,
    dramatic: DramaticPlan,
    update_number: int,
) -> list[InformationState]:
    """Mutate information_states from a committed dramatic plan.

    For each ``scene.reveals`` entry:
      * match an existing fact (case-insensitive substring) or create a new
        ``InformationState`` row;
      * add a ``KnowledgeEntry`` for every POV/present character in the scene
        AND for the literal ``"reader"``.

    ``scene.withholds`` are intentionally left as *absence* of entries — the
    gap spec says so. If a withhold implies a false belief, that's a follow-
    up (TODO) — we record nothing here.

    Returns the list of states that were written (new or updated), in the
    order they were touched.
    """
    existing = world.list_information_states(quest_id)
    # Mutable index by id for in-session merges.
    by_id: dict[str, InformationState] = {s.id: s for s in existing}
    touched_order: list[str] = []

    for scene in dramatic.scenes:
        learners = _pov_entities(scene) + [READER]
        for fact_text in scene.reveals:
            if not fact_text or not fact_text.strip():
                continue
            match = _find_matching_fact(list(by_id.values()), fact_text)
            if match is None:
                new = InformationState(
                    id=_fact_id(quest_id, fact_text),
                    quest_id=quest_id,
                    fact=fact_text.strip(),
                    ground_truth=True,
                    known_by={},
                )
                by_id[new.id] = new
                match = new

            new_known = dict(match.known_by)
            for eid in learners:
                if eid in new_known:
                    continue  # keep the earliest learning record
                new_known[eid] = KnowledgeEntry(
                    learned_at_update=update_number,
                    learned_how="reveal",
                    believes=True,
                    confidence=0.9,
                )
            by_id[match.id] = match.model_copy(update={"known_by": new_known})
            if match.id not in touched_order:
                touched_order.append(match.id)

            # TODO(G7 followup): scene.withholds that imply a false belief
            # should mutate known_by with believes=False for the unaware
            # parties. For now, absence is the signal.

    written: list[InformationState] = []
    for fid in touched_order:
        state = by_id[fid]
        world.upsert_information_state(state)
        written.append(state)
    return written


def _story_characters(world: WorldStateManager) -> list[str]:
    try:
        ents = world.list_entities(EntityType.CHARACTER)
    except Exception:
        ents = []
    return [e.id for e in ents if getattr(e, "status", None) in (
        None, "active", "dormant",
    ) or True]  # permissive — any character counts for asymmetry checks


def _tension_potential(
    kind: AsymmetryKind,
    updates_standing: int | None,
) -> float:
    """Heuristic: dramatic irony is most potent when recently planted but not
    yet paid off. Mysteries are potent when old. Secrets ramp with age.
    """
    u = updates_standing if updates_standing is not None else 0
    if kind == AsymmetryKind.DRAMATIC_IRONY:
        # Fresh -> high; stale (>5) -> decays but still present.
        if u <= 1:
            return 0.9
        if u <= 3:
            return 0.75
        if u <= 6:
            return 0.55
        return 0.35
    if kind == AsymmetryKind.MYSTERY:
        return min(0.4 + 0.05 * u, 0.9)
    if kind == AsymmetryKind.SECRET:
        return min(0.5 + 0.05 * u, 0.85)
    if kind == AsymmetryKind.FALSE_BELIEF:
        return 0.8
    return 0.0


def compute_asymmetries(
    world: WorldStateManager,
    quest_id: str,
    *,
    current_update: int | None = None,
) -> list[InformationAsymmetry]:
    """Derive the live asymmetries for a quest from its information_states.

    Pure (in-memory) computation — no DB writes. ``current_update`` is used
    to compute ``updates_standing`` (defaults to the max timeline update).
    """
    states = world.list_information_states(quest_id)
    if not states:
        return []

    if current_update is None:
        try:
            tl = world.list_timeline()
            current_update = max((e.update_number for e in tl), default=0)
        except Exception:
            current_update = 0

    # Story-present characters (candidates for secret / mystery relationships).
    story_chars = set(_story_characters(world))

    out: list[InformationAsymmetry] = []
    for st in states:
        known_char_ids = [
            k for k in st.known_by.keys() if k not in (READER, NARRATOR)
        ]
        reader_knows = READER in st.known_by

        reader_entry = st.known_by.get(READER)
        earliest_char_learned = min(
            (e.learned_at_update for k, e in st.known_by.items()
             if k not in (READER, NARRATOR)),
            default=None,
        )

        # false_belief — any party whose `believes` disagrees with ground_truth
        for kid, entry in st.known_by.items():
            if entry.believes != st.ground_truth:
                out.append(InformationAsymmetry(
                    kind=AsymmetryKind.FALSE_BELIEF,
                    fact_id=st.id,
                    fact=st.fact,
                    knowers=list(st.known_by.keys()),
                    unaware=[],
                    believer_id=kid,
                    tension_potential=_tension_potential(
                        AsymmetryKind.FALSE_BELIEF, None,
                    ),
                ))

        # dramatic_irony — reader knows, at least one story-present character does not.
        if reader_knows and story_chars:
            unaware = [c for c in story_chars if c not in st.known_by]
            if unaware:
                u_standing = (
                    (current_update - reader_entry.learned_at_update)
                    if reader_entry else None
                )
                out.append(InformationAsymmetry(
                    kind=AsymmetryKind.DRAMATIC_IRONY,
                    fact_id=st.id,
                    fact=st.fact,
                    knowers=[READER] + known_char_ids,
                    unaware=sorted(unaware),
                    updates_standing=u_standing,
                    tension_potential=_tension_potential(
                        AsymmetryKind.DRAMATIC_IRONY, u_standing,
                    ),
                ))

        # mystery — character(s) know, reader does not.
        if known_char_ids and not reader_knows:
            u_standing = (
                (current_update - earliest_char_learned)
                if earliest_char_learned is not None else None
            )
            out.append(InformationAsymmetry(
                kind=AsymmetryKind.MYSTERY,
                fact_id=st.id,
                fact=st.fact,
                knowers=known_char_ids,
                unaware=[READER],
                updates_standing=u_standing,
                tension_potential=_tension_potential(
                    AsymmetryKind.MYSTERY, u_standing,
                ),
            ))

        # secret — some story-characters know, others don't.
        if story_chars:
            aware = [c for c in known_char_ids if c in story_chars]
            unaware = [c for c in story_chars if c not in st.known_by]
            if aware and unaware:
                u_standing = (
                    (current_update - earliest_char_learned)
                    if earliest_char_learned is not None else None
                )
                out.append(InformationAsymmetry(
                    kind=AsymmetryKind.SECRET,
                    fact_id=st.id,
                    fact=st.fact,
                    knowers=sorted(aware),
                    unaware=sorted(unaware),
                    updates_standing=u_standing,
                    tension_potential=_tension_potential(
                        AsymmetryKind.SECRET, u_standing,
                    ),
                ))

    return out


def ripe_asymmetry_count(
    asymmetries: Iterable[InformationAsymmetry],
    *,
    standing_threshold: int = 3,
) -> int:
    """Number of asymmetries that have been standing > ``standing_threshold``
    updates. Dramatic irony that's been live a while is the canonical "ripe
    for revelation" signal.
    """
    n = 0
    for a in asymmetries:
        if a.kind == AsymmetryKind.DRAMATIC_IRONY and (
            a.updates_standing or 0
        ) > standing_threshold:
            n += 1
    return n
