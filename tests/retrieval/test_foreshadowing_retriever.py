"""Tests for ``ForeshadowingRetriever`` (Wave 4b).

The retriever is structured (no embeddings), so the fixture seeds a
:class:`WorldStateManager` with a mix of hooks in every bucket the
ripeness computation cares about:

* Overdue — ``current_update`` past a synthetic ``target_update_max``.
* Ripe — ``current_update`` inside a synthetic target window.
* Aging — no target window, planted at least 5 updates ago.
* Fresh — no target window, planted within the last 5 updates.
* PAID_OFF / ABANDONED — eligible-by-age but dropped by status filter.

Because the current :class:`~app.world.schema.ForeshadowingHook` model
doesn't expose ``target_update_min``/``target_update_max``, the tests
use a lightweight subclass that adds those attributes purely in memory
so the ripeness branches exercise without a schema migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval import ForeshadowingRetriever, Query
from app.retrieval.foreshadowing_retriever import _ScoredHook
from app.world.db import open_db
from app.world.schema import ForeshadowingHook, HookStatus
from app.world.state_manager import WorldStateManager


# -- Helpers -------------------------------------------------------------


class _WindowedHook(ForeshadowingHook):
    """Hook subtype carrying optional target-window bounds.

    Pydantic models reject unknown attributes by default, so the
    subclass declares the fields explicitly. The DB layer ignores
    these extras — they live only on the in-memory ``ForeshadowingHook``
    list the retriever reads through ``snapshot()``.
    """

    target_update_min: int | None = None
    target_update_max: int | None = None


def _make_world(tmp_path: Path, hooks: list[ForeshadowingHook]) -> WorldStateManager:
    conn = open_db(tmp_path / "quest.db")
    wsm = WorldStateManager(conn)
    for h in hooks:
        wsm.add_foreshadowing(h)
    return wsm


def _patch_snapshot(wsm: WorldStateManager, hooks: list[ForeshadowingHook]) -> None:
    """Override ``snapshot()`` to return our windowed hook objects.

    The retriever reads via ``world.snapshot().foreshadowing``; the DB
    round-trip strips the extra window fields, so we monkey-patch the
    method on the instance to hand back the in-memory objects with
    their windows intact. This keeps the retriever tested end-to-end
    without needing a schema change upstream.
    """
    from dataclasses import replace

    orig = wsm.snapshot

    def _snap():
        real = orig()
        return replace(real, foreshadowing=list(hooks))

    wsm.snapshot = _snap  # type: ignore[method-assign]


# -- Tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_overdue_ranks_above_ripe_above_aging(tmp_path):
    overdue = _WindowedHook(
        id="fs:overdue",
        description="An overdue hook.",
        planted_at_update=2,
        payoff_target="An overdue payoff.",
        target_update_min=5,
        target_update_max=8,
    )
    ripe = _WindowedHook(
        id="fs:ripe",
        description="A ripe hook.",
        planted_at_update=4,
        payoff_target="A ripe payoff.",
        target_update_min=8,
        target_update_max=12,
    )
    aging = _WindowedHook(
        id="fs:aging",
        description="An aging hook without a target window.",
        planted_at_update=1,
        payoff_target="An aging payoff.",
    )

    wsm = _make_world(tmp_path, [overdue, ripe, aging])
    _patch_snapshot(wsm, [overdue, ripe, aging])

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(
        Query(filters={"current_update": 10}), k=10
    )

    assert [r.metadata["hook_id"] for r in results] == [
        "fs:overdue",
        "fs:ripe",
        "fs:aging",
    ]
    assert results[0].score == 1.0
    assert results[1].score == pytest.approx(0.7)
    assert results[2].score == pytest.approx(0.3)
    assert results[0].metadata["ripeness_status"] == "overdue"
    assert results[1].metadata["ripeness_status"] == "ripe"
    assert results[2].metadata["ripeness_status"] == "aging"
    # source_id format matches the spec contract.
    assert results[0].source_id == "hook/q1/fs:overdue"
    # payoff_description is surfaced on the metadata (maps from payoff_target).
    assert results[0].metadata["payoff_description"] == "An overdue payoff."


@pytest.mark.asyncio
async def test_resolved_and_abandoned_hooks_dropped(tmp_path):
    planted = _WindowedHook(
        id="fs:planted",
        description="Still live.",
        planted_at_update=1,
        payoff_target="Live payoff.",
    )
    paid_off = _WindowedHook(
        id="fs:paid",
        description="Already resolved.",
        planted_at_update=1,
        payoff_target="Old payoff.",
        status=HookStatus.PAID_OFF,
        paid_off_at_update=7,
    )
    abandoned = _WindowedHook(
        id="fs:abandoned",
        description="Given up.",
        planted_at_update=1,
        payoff_target="Ignored payoff.",
        status=HookStatus.ABANDONED,
    )

    wsm = _make_world(tmp_path, [planted, paid_off, abandoned])
    _patch_snapshot(wsm, [planted, paid_off, abandoned])

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(
        Query(filters={"current_update": 10}), k=10
    )

    ids = [r.metadata["hook_id"] for r in results]
    assert ids == ["fs:planted"]


@pytest.mark.asyncio
async def test_referenced_status_is_eligible(tmp_path):
    """REFERENCED maps onto the spec's PARTIAL bucket — must not be dropped."""
    referenced = _WindowedHook(
        id="fs:partial",
        description="Half-seen.",
        planted_at_update=1,
        payoff_target="Partial payoff.",
        status=HookStatus.REFERENCED,
    )
    wsm = _make_world(tmp_path, [referenced])
    _patch_snapshot(wsm, [referenced])

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(
        Query(filters={"current_update": 10}), k=3
    )
    assert [r.metadata["hook_id"] for r in results] == ["fs:partial"]
    assert results[0].metadata["status"] == "referenced"


@pytest.mark.asyncio
async def test_missing_target_window_aging_when_old_enough(tmp_path):
    old_no_window = _WindowedHook(
        id="fs:old",
        description="Ancient hook.",
        planted_at_update=0,
        payoff_target="Old payoff.",
    )
    young_no_window = _WindowedHook(
        id="fs:young",
        description="Brand new.",
        planted_at_update=8,
        payoff_target="Too-fresh payoff.",
    )

    wsm = _make_world(tmp_path, [old_no_window, young_no_window])
    _patch_snapshot(wsm, [old_no_window, young_no_window])

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(
        Query(filters={"current_update": 10}), k=5
    )

    ids = [r.metadata["hook_id"] for r in results]
    # old_no_window: planted=0, current=10 → aging branch.
    # young_no_window: planted=8, current=10 → Fresh (dropped).
    assert ids == ["fs:old"]
    assert results[0].metadata["ripeness_status"] == "aging"


@pytest.mark.asyncio
async def test_k_limit_respected(tmp_path):
    hooks = [
        _WindowedHook(
            id=f"fs:{i}",
            description=f"hook {i}",
            planted_at_update=i,
            payoff_target=f"payoff {i}",
            target_update_min=1,
            target_update_max=2,
        )
        for i in range(5)
    ]
    wsm = _make_world(tmp_path, hooks)
    _patch_snapshot(wsm, hooks)

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(
        Query(filters={"current_update": 100}), k=2
    )
    # All five are overdue at current_update=100 (target_update_max=2);
    # ranking ties break by planted_at_update ascending.
    assert len(results) == 2
    assert [r.metadata["hook_id"] for r in results] == ["fs:0", "fs:1"]


@pytest.mark.asyncio
async def test_tie_break_by_oldest_planted(tmp_path):
    """Two overdue hooks in the same bucket → oldest planted ranks higher."""
    newer = _WindowedHook(
        id="fs:newer",
        description="Newer plant, overdue.",
        planted_at_update=5,
        payoff_target="x",
        target_update_min=6,
        target_update_max=7,
    )
    older = _WindowedHook(
        id="fs:older",
        description="Older plant, overdue.",
        planted_at_update=1,
        payoff_target="x",
        target_update_min=2,
        target_update_max=3,
    )
    wsm = _make_world(tmp_path, [newer, older])
    _patch_snapshot(wsm, [newer, older])

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(
        Query(filters={"current_update": 20}), k=5
    )
    assert [r.metadata["hook_id"] for r in results] == ["fs:older", "fs:newer"]


@pytest.mark.asyncio
async def test_missing_current_update_defaults_to_zero(tmp_path):
    """Without ``current_update``, only hooks that cleanly satisfy a
    non-temporal branch can surface — aging needs planted_at + 5 <= 0,
    which no real hook can reach. So the result is empty."""
    hook = _WindowedHook(
        id="fs:1",
        description="Live.",
        planted_at_update=1,
        payoff_target="x",
    )
    wsm = _make_world(tmp_path, [hook])
    _patch_snapshot(wsm, [hook])

    retriever = ForeshadowingRetriever(wsm, quest_id="q1")
    results = await retriever.retrieve(Query(), k=5)
    assert results == []


def test_scored_hook_is_frozen():
    """The internal scored-hook dataclass is frozen (immutable)."""
    hook = ForeshadowingHook(
        id="fs:1", description="d", planted_at_update=1, payoff_target="x"
    )
    scored = _ScoredHook(
        hook=hook,
        score=0.5,
        ripeness_status="aging",
        target_update_min=None,
        target_update_max=None,
    )
    with pytest.raises(Exception):
        scored.score = 0.9  # type: ignore[misc]
