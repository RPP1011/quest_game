from __future__ import annotations
from app.world.schema import (
    ArcPosition,
    ForeshadowingHook,
    HookStatus,
    NarrativeRecord,
    PlotThread,
    ThreadStatus,
    TimelineEvent,
    WorldRule,
)
from app.world.state_manager import WorldStateManager


def test_append_and_list_timeline(db):
    sm = WorldStateManager(db)
    sm.append_timeline_event(TimelineEvent(update_number=1, event_index=0, description="x"))
    sm.append_timeline_event(TimelineEvent(update_number=1, event_index=1, description="y"))
    events = sm.list_timeline(update_number=1)
    assert [e.event_index for e in events] == [0, 1]


def test_write_and_get_narrative(db):
    sm = WorldStateManager(db)
    n = NarrativeRecord(update_number=1, raw_text="She walked in.", player_action="enter")
    sm.write_narrative(n)
    got = sm.get_narrative(1)
    assert got.raw_text == "She walked in."
    assert got.player_action == "enter"


def test_list_narrative_ordered(db):
    sm = WorldStateManager(db)
    sm.write_narrative(NarrativeRecord(update_number=2, raw_text="b"))
    sm.write_narrative(NarrativeRecord(update_number=1, raw_text="a"))
    records = sm.list_narrative(limit=10)
    assert [r.update_number for r in records] == [1, 2]


def test_foreshadowing_crud(db):
    sm = WorldStateManager(db)
    h = ForeshadowingHook(id="fs:1", description="...", planted_at_update=1, payoff_target="...")
    sm.add_foreshadowing(h)
    assert sm.get_foreshadowing("fs:1") == h
    sm.update_foreshadowing(
        "fs:1",
        {"status": HookStatus.PAID_OFF, "paid_off_at_update": 5, "references": [3, 4]},
    )
    got = sm.get_foreshadowing("fs:1")
    assert got.status == HookStatus.PAID_OFF
    assert got.paid_off_at_update == 5
    assert got.references == [3, 4]


def test_plot_thread_crud(db):
    sm = WorldStateManager(db)
    pt = PlotThread(id="pt:1", name="Main", description="d",
                    involved_entities=["a"], arc_position=ArcPosition.RISING)
    sm.add_plot_thread(pt)
    assert sm.get_plot_thread("pt:1").name == "Main"
    sm.update_plot_thread("pt:1", {"status": ThreadStatus.RESOLVED, "priority": 8})
    got = sm.get_plot_thread("pt:1")
    assert got.status == ThreadStatus.RESOLVED
    assert got.priority == 8


def test_world_rules(db):
    sm = WorldStateManager(db)
    r = WorldRule(id="r:1", category="magic", description="No magic in zone 3.",
                  constraints={"zone": 3})
    sm.add_rule(r)
    assert sm.list_rules()[0].description == "No magic in zone 3."
