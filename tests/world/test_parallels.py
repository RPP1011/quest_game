from __future__ import annotations

import pytest

from app.world.schema import Parallel, ParallelStatus
from app.world.state_manager import WorldStateError, WorldStateManager


def test_parallel_crud(db):
    sm = WorldStateManager(db)
    p = Parallel(
        id="par:1",
        quest_id="q:main",
        source_update=3,
        source_description="Hero refuses the crown in triumph.",
        inversion_axis="power reversed",
        target_description="Hero accepts the crown in defeat.",
        target_update_range_min=8,
        target_update_range_max=10,
        theme_ids=["t:power", "t:fate"],
    )
    sm.add_parallel(p)
    got = sm.get_parallel("par:1")
    assert got.status == ParallelStatus.PLANTED
    assert got.theme_ids == ["t:power", "t:fate"]
    assert got.target_update_range_min == 8
    assert got.target_update_range_max == 10


def test_parallel_delivery_lifecycle(db):
    sm = WorldStateManager(db)
    sm.add_parallel(Parallel(
        id="par:1", quest_id="q:main", source_update=3,
        source_description="A", inversion_axis="hope -> despair",
        target_description="B",
    ))
    sm.update_parallel("par:1", {"status": ParallelStatus.SCHEDULED})
    assert sm.get_parallel("par:1").status == ParallelStatus.SCHEDULED

    sm.update_parallel("par:1", {
        "status": ParallelStatus.DELIVERED,
        "delivered_at_update": 9,
    })
    got = sm.get_parallel("par:1")
    assert got.status == ParallelStatus.DELIVERED
    assert got.delivered_at_update == 9


def test_list_parallels_filters(db):
    sm = WorldStateManager(db)
    for i, status in enumerate([
        ParallelStatus.PLANTED,
        ParallelStatus.DELIVERED,
        ParallelStatus.PLANTED,
    ]):
        sm.add_parallel(Parallel(
            id=f"par:{i}", quest_id="q", source_update=i,
            source_description="s", inversion_axis="a",
            target_description="t", status=status,
        ))
    active = sm.list_parallels(statuses=[ParallelStatus.PLANTED])
    assert {p.id for p in active} == {"par:0", "par:2"}
    assert len(sm.list_parallels()) == 3


def test_parallel_missing_raises(db):
    sm = WorldStateManager(db)
    with pytest.raises(WorldStateError):
        sm.get_parallel("par:missing")


def test_parallels_in_snapshot(db):
    sm = WorldStateManager(db)
    sm.add_parallel(Parallel(
        id="par:1", quest_id="q", source_update=1,
        source_description="s", inversion_axis="a", target_description="t",
    ))
    snap = sm.snapshot()
    assert len(snap.parallels) == 1
    assert snap.parallels[0].id == "par:1"


def test_parallel_rollback_reverts_delivery(db):
    sm = WorldStateManager(db)
    sm.add_parallel(Parallel(
        id="par:1", quest_id="q", source_update=2,
        source_description="s", inversion_axis="a", target_description="t",
    ))
    sm.update_parallel("par:1", {
        "status": ParallelStatus.DELIVERED,
        "delivered_at_update": 9,
    })
    # A parallel planted AFTER to_update should be deleted.
    sm.add_parallel(Parallel(
        id="par:late", quest_id="q", source_update=12,
        source_description="s", inversion_axis="a", target_description="t",
    ))
    sm.rollback(to_update=5)
    got = sm.get_parallel("par:1")
    assert got.status == ParallelStatus.PLANTED
    assert got.delivered_at_update is None
    with pytest.raises(WorldStateError):
        sm.get_parallel("par:late")
