from __future__ import annotations
from app.world.schema import Entity, EntityType, Relationship, TimelineEvent
from app.world.delta import (
    EntityCreate,
    EntityUpdate,
    FSUpdate,
    PTUpdate,
    RelChange,
    StateDelta,
    TimelineEventOp,
    ValidationIssue,
    ValidationResult,
)


def test_empty_delta_is_valid_shape():
    d = StateDelta()
    assert d.entity_creates == []
    assert d.entity_updates == []
    assert d.relationship_changes == []
    assert d.timeline_events == []
    assert d.foreshadowing_updates == []
    assert d.plot_thread_updates == []


def test_entity_create_wraps_entity():
    e = Entity(id="char:alice", entity_type=EntityType.CHARACTER, name="Alice")
    op = EntityCreate(entity=e)
    assert op.entity.id == "char:alice"


def test_entity_update_carries_patch():
    op = EntityUpdate(id="char:alice", patch={"status": "dormant"})
    assert op.patch["status"] == "dormant"


def test_rel_change_variants():
    r = Relationship(source_id="a", target_id="b", rel_type="ally")
    add = RelChange(action="add", relationship=r)
    remove = RelChange(
        action="remove",
        relationship=Relationship(source_id="a", target_id="b", rel_type="ally"),
    )
    assert add.action == "add"
    assert remove.action == "remove"


def test_timeline_event_op_wraps_event():
    ev = TimelineEvent(update_number=1, event_index=0, description="x")
    op = TimelineEventOp(event=ev)
    assert op.event.description == "x"


def test_fs_update_transition():
    op = FSUpdate(id="fs:001", new_status="paid_off", paid_off_at_update=7)
    assert op.new_status == "paid_off"


def test_pt_update_partial():
    op = PTUpdate(id="pt:main", patch={"status": "resolved"})
    assert op.patch == {"status": "resolved"}


def test_delta_composes_all_ops():
    d = StateDelta(
        entity_creates=[EntityCreate(entity=Entity(
            id="x", entity_type=EntityType.ITEM, name="X"))],
        entity_updates=[EntityUpdate(id="y", patch={"status": "dormant"})],
    )
    assert len(d.entity_creates) == 1
    assert len(d.entity_updates) == 1


def test_validation_result_is_ok_when_no_violations():
    r = ValidationResult(issues=[])
    assert r.ok
    r2 = ValidationResult(issues=[ValidationIssue(severity="error", message="x")])
    assert not r2.ok


def test_validation_result_ok_allows_warnings():
    r = ValidationResult(issues=[ValidationIssue(severity="warning", message="w")])
    assert r.ok
