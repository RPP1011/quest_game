# tests/world/test_retcon.py
from __future__ import annotations
import pytest
from app.world.db import open_db
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
)
from app.world.retcon import RetconSpec, RetconResult
from app.world.schema import (
    Entity,
    EntityType,
    NarrativeRecord,
    Relationship,
    TimelineEvent,
)
from app.world.state_manager import (
    InvalidDeltaError,
    WorldStateManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sm(db) -> WorldStateManager:
    return WorldStateManager(db)


def _seed_character(sm: WorldStateManager, eid: str, name: str, update: int = 1) -> None:
    delta = StateDelta(
        entity_creates=[
            EntityCreate(entity=Entity(id=eid, entity_type=EntityType.CHARACTER, name=name))
        ]
    )
    sm.apply_delta(delta, update_number=update)


def _write_narrative(sm: WorldStateManager, update: int, text: str) -> None:
    sm.write_narrative(NarrativeRecord(update_number=update, raw_text=text))


# ---------------------------------------------------------------------------
# Test 1: retcon applies entity updates and returns correct new_update_number
# ---------------------------------------------------------------------------

def test_retcon_applies_entity_update(db):
    """Retcon patches an existing entity and returns a synthetic update beyond current max."""
    sm = _make_sm(db)
    _seed_character(sm, "hero", "Aldric", update=1)
    # Narrative at update 2
    _write_narrative(sm, 2, "Aldric defeats the dragon.")
    # Advance timeline to update 3
    sm.append_timeline_event(TimelineEvent(
        update_number=3, event_index=0, description="later event",
        involved_entities=["hero"],
    ))

    spec = RetconSpec(
        target_update=1,
        delta=StateDelta(
            entity_updates=[EntityUpdate(id="hero", patch={"name": "Eldric"})]
        ),
        reason="Retconning hero name from Aldric to Eldric",
    )
    result = sm.retcon(spec)

    assert isinstance(result, RetconResult)
    assert result.applied_update == 1
    # new update number must be strictly greater than current max (3)
    assert result.new_update_number > 3
    # The entity name should be updated in the DB
    assert sm.get_entity("hero").name == "Eldric"


# ---------------------------------------------------------------------------
# Test 2: retcon creates a timeline event with [retcon] prefix
# ---------------------------------------------------------------------------

def test_retcon_creates_timeline_event_with_retcon_prefix(db):
    """A [retcon] timeline event is appended at the new update number."""
    sm = _make_sm(db)
    _seed_character(sm, "wizard", "Galadon", update=1)

    spec = RetconSpec(
        target_update=1,
        delta=StateDelta(
            entity_updates=[EntityUpdate(id="wizard", patch={"data": {"power": "fire"}})]
        ),
        reason="Wizard always had fire affinity",
    )
    result = sm.retcon(spec)

    events = sm.list_timeline(result.new_update_number)
    assert len(events) >= 1
    retcon_events = [e for e in events if e.description.startswith("[retcon]")]
    assert retcon_events, "Expected at least one [retcon] timeline event"
    assert "Wizard always had fire affinity" in retcon_events[0].description


# ---------------------------------------------------------------------------
# Test 3: affected_narrative lists records that mention touched entity by name
# ---------------------------------------------------------------------------

def test_retcon_affected_narrative_mentions_touched_entity(db):
    """Narrative records >= target_update that mention the entity name are flagged."""
    sm = _make_sm(db)
    _seed_character(sm, "rogue", "Kira", update=1)

    # Narrative records at various updates
    _write_narrative(sm, 1, "Kira slips into the shadows.")
    _write_narrative(sm, 2, "Kira steals the artifact.")
    _write_narrative(sm, 3, "The dragon roars.")  # doesn't mention Kira

    spec = RetconSpec(
        target_update=1,
        delta=StateDelta(
            entity_updates=[EntityUpdate(id="rogue", patch={"data": {"class": "assassin"}})]
        ),
        reason="Kira was always an assassin",
    )
    result = sm.retcon(spec)

    # Updates 1 and 2 mention "Kira" → should be in affected list
    assert 1 in result.affected_narrative
    assert 2 in result.affected_narrative


# ---------------------------------------------------------------------------
# Test 4: records NOT mentioning touched entities are NOT flagged
# ---------------------------------------------------------------------------

def test_retcon_unrelated_narrative_not_flagged(db):
    """Narrative records that don't mention the touched entity are excluded."""
    sm = _make_sm(db)
    _seed_character(sm, "knight", "Brann", update=1)
    _seed_character(sm, "merchant", "Voss", update=1)

    _write_narrative(sm, 2, "Voss opens his stall at dawn.")
    _write_narrative(sm, 3, "Rain falls over the market.")

    spec = RetconSpec(
        target_update=1,
        delta=StateDelta(
            entity_updates=[EntityUpdate(id="knight", patch={"data": {"rank": "paladin"}})]
        ),
        reason="Brann was always a paladin",
    )
    result = sm.retcon(spec)

    # Neither narrative mentions "Brann" → should not be flagged
    assert 2 not in result.affected_narrative
    assert 3 not in result.affected_narrative


# ---------------------------------------------------------------------------
# Test 5: retcon on invalid delta raises InvalidDeltaError
# ---------------------------------------------------------------------------

def test_retcon_invalid_delta_raises_error(db):
    """Passing a delta that references a non-existent entity raises InvalidDeltaError."""
    sm = _make_sm(db)

    spec = RetconSpec(
        target_update=1,
        delta=StateDelta(
            entity_updates=[EntityUpdate(id="ghost_entity_does_not_exist", patch={"status": "dormant"})]
        ),
        reason="Trying to update a ghost entity",
    )
    with pytest.raises(InvalidDeltaError):
        sm.retcon(spec)


# ---------------------------------------------------------------------------
# Test 6: retcon records below target_update are not flagged (boundary check)
# ---------------------------------------------------------------------------

def test_retcon_narrative_before_target_update_not_flagged(db):
    """Narrative records with update_number < target_update are excluded even if they mention the entity."""
    sm = _make_sm(db)
    _seed_character(sm, "bard", "Lyria", update=1)

    # Update 1 is before target_update=2
    _write_narrative(sm, 1, "Lyria sings a ballad.")
    _write_narrative(sm, 3, "Lyria performs at the inn.")

    spec = RetconSpec(
        target_update=2,
        delta=StateDelta(
            entity_updates=[EntityUpdate(id="bard", patch={"data": {"instrument": "lute"}})]
        ),
        reason="Lyria always played the lute",
    )
    result = sm.retcon(spec)

    # Update 1 is before target_update → should NOT be flagged
    assert 1 not in result.affected_narrative
    # Update 3 mentions Lyria and is >= target_update → should be flagged
    assert 3 in result.affected_narrative


# ---------------------------------------------------------------------------
# Test 7: retcon touching a relationship flags narrative mentioning source/target ids
# ---------------------------------------------------------------------------

def test_retcon_relationship_change_flags_related_narrative(db):
    """Narrative mentioning source_id or target_id of a changed relationship is flagged."""
    sm = _make_sm(db)
    _seed_character(sm, "elf", "Sylvara", update=1)
    _seed_character(sm, "dwarf", "Dorin", update=1)

    # Add a relationship to modify
    sm.add_relationship(Relationship(
        source_id="elf", target_id="dwarf", rel_type="ally",
        established_at_update=1,
    ))

    # Narrative at update 2 mentions elf by its id
    _write_narrative(sm, 2, "The elf (id: elf) and dwarf forge a truce.")

    spec = RetconSpec(
        target_update=1,
        delta=StateDelta(
            relationship_changes=[RelChange(
                action="modify",
                relationship=Relationship(
                    source_id="elf", target_id="dwarf", rel_type="ally",
                    data={"strength": "strong"},
                ),
            )]
        ),
        reason="Elf-dwarf alliance was always strong",
    )
    result = sm.retcon(spec)

    # The narrative at 2 mentions "elf" (a source_id/target_id) → should be flagged
    assert 2 in result.affected_narrative
