from __future__ import annotations
import pytest
from app.world.db import open_db
from app.world.state_manager import WorldStateManager


@pytest.fixture
def sm(tmp_path):
    conn = open_db(tmp_path / "test.db")
    wsm = WorldStateManager(conn)
    yield wsm
    conn.close()


def test_create_and_get_foreshadow_triple(sm):
    sm.create_foreshadow_triple(
        id="ft_abc12345",
        hook_id="fs:pistol",
        foreshadow_text="Tristan notices the pistol's unusual weight",
        trigger_pred={"type": "chapter_gte", "value": 5},
        payoff_text="The pistol fires unexpectedly, revealing its cursed nature",
        planted_chapter=2,
        deadline_chapter=8,
    )
    triple = sm.get_foreshadow_triple("ft_abc12345")
    assert triple["hook_id"] == "fs:pistol"
    assert triple["status"] == "planted"
    assert triple["trigger_pred"] == {"type": "chapter_gte", "value": 5}
    assert triple["deadline_chapter"] == 8
    assert triple["verified_planted"] is None


def test_update_foreshadow_triple_status(sm):
    sm.create_foreshadow_triple(
        id="ft_abc12345",
        hook_id="fs:pistol",
        foreshadow_text="pistol weight",
        trigger_pred={"type": "chapter_gte", "value": 5},
        payoff_text="pistol fires",
        planted_chapter=2,
        deadline_chapter=8,
    )
    sm.update_foreshadow_triple("ft_abc12345", status="triggered")
    assert sm.get_foreshadow_triple("ft_abc12345")["status"] == "triggered"

    sm.update_foreshadow_triple("ft_abc12345", verified_planted=0.85)
    assert sm.get_foreshadow_triple("ft_abc12345")["verified_planted"] == pytest.approx(0.85)


def test_list_foreshadow_triples_by_status(sm):
    for i, status in enumerate(["planted", "planted", "triggered", "paid_off"]):
        sm.create_foreshadow_triple(
            id=f"ft_{i:08d}",
            hook_id=f"fs:hook{i}",
            foreshadow_text=f"text {i}",
            trigger_pred={"type": "chapter_gte", "value": i + 1},
            payoff_text=f"payoff {i}",
            planted_chapter=1,
        )
        if status != "planted":
            sm.update_foreshadow_triple(f"ft_{i:08d}", status=status)

    planted = sm.list_foreshadow_triples(status="planted")
    assert len(planted) == 2
    triggered = sm.list_foreshadow_triples(status="triggered")
    assert len(triggered) == 1


def test_list_overdue_foreshadow_triples(sm):
    sm.create_foreshadow_triple(
        id="ft_overdue1",
        hook_id="fs:overdue",
        foreshadow_text="overdue hook",
        trigger_pred={"type": "chapter_gte", "value": 3},
        payoff_text="should have fired",
        planted_chapter=1,
        deadline_chapter=5,
    )
    overdue = sm.list_overdue_foreshadow_triples(current_chapter=6)
    assert len(overdue) == 1
    assert overdue[0]["id"] == "ft_overdue1"

    not_overdue = sm.list_overdue_foreshadow_triples(current_chapter=4)
    assert len(not_overdue) == 0
