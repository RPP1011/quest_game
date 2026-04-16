from __future__ import annotations
from pathlib import Path

import pytest

from app.refinement.selectors import (
    SiblingOutscoredSelector, UnpaidHookSelector, WeakChapterSelector,
)
from app.world.db import open_db
from app.world.schema import (
    ArcSkeleton, ForeshadowingHook, HookPlacement, RolloutChapter,
    RolloutRun, SkeletonChapter, StoryCandidate,
)
from app.world.state_manager import WorldStateManager


@pytest.fixture
def sm(tmp_path: Path):
    conn = open_db(tmp_path / "w.db")
    wsm = WorldStateManager(conn)
    wsm.add_story_candidate(StoryCandidate(
        id="cand_1", quest_id="q1", title="T", synopsis="S",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=3,
    ))
    yield wsm
    conn.close()


def _seed_rollout_with_scores(
    sm: WorldStateManager, rid: str, scores_by_chapter: dict[int, dict[str, float]],
    proses: dict[int, str] | None = None,
) -> None:
    sm.create_rollout(RolloutRun(
        id=rid, quest_id="q1", candidate_id="cand_1",
        profile_id="impulsive", total_chapters_target=max(scores_by_chapter, default=0),
    ))
    for ci, dims in scores_by_chapter.items():
        sm.save_rollout_chapter(RolloutChapter(
            rollout_id=rid, chapter_index=ci,
            player_action=f"act{ci}",
            prose=(proses or {}).get(ci, f"prose chapter {ci}"),
        ))
        sm.save_chapter_scores(rid, ci, {
            d: {"score": v, "rationale": f"because {d} was {v}"}
            for d, v in dims.items()
        })


# ---- WeakChapterSelector --------------------------------------------------

def test_weak_chapter_selects_below_threshold(sm):
    _seed_rollout_with_scores(sm, "r1", {
        1: {"tension_execution": 0.4, "voice_distinctiveness": 0.5},  # mean 0.45 < 0.55
        2: {"tension_execution": 0.7, "voice_distinctiveness": 0.7},  # mean 0.7 above
        3: {"tension_execution": 0.5, "voice_distinctiveness": 0.5},  # mean 0.5 < 0.55
    })
    sel = WeakChapterSelector(sm, threshold=0.55)
    targets = sel.select(quest_id="q1")
    chapters = [t.chapter_index for t in targets]
    assert chapters == [1, 3]  # lowest first
    # Guidance references the worst dim
    assert "tension_execution" in targets[0].guidance


def test_weak_chapter_respects_max(sm):
    _seed_rollout_with_scores(sm, "r1", {
        i: {"a": 0.3} for i in range(1, 6)
    })
    sel = WeakChapterSelector(sm)
    targets = sel.select(quest_id="q1", max_targets=2)
    assert len(targets) == 2


def test_weak_chapter_filter_by_rollout(sm):
    _seed_rollout_with_scores(sm, "r1", {1: {"a": 0.3}})
    _seed_rollout_with_scores(sm, "r2", {1: {"a": 0.3}})
    sel = WeakChapterSelector(sm)
    targets = sel.select(quest_id="q1", rollout_id="r2")
    assert len(targets) == 1
    assert targets[0].rollout_id == "r2"


# ---- UnpaidHookSelector ---------------------------------------------------

def _seed_hook_skeleton(sm: WorldStateManager, hooks: list[HookPlacement]) -> None:
    sm.add_foreshadowing(ForeshadowingHook(
        id="fs:1", description="The bell tolls",
        planted_at_update=0, payoff_target="ring at climax",
    ))
    sm.add_foreshadowing(ForeshadowingHook(
        id="fs:2", description="The pact",
        planted_at_update=0, payoff_target="reveal in ch 5",
    ))
    sm.save_arc_skeleton(ArcSkeleton(
        id="sk_1", candidate_id="cand_1", quest_id="q1",
        chapters=[SkeletonChapter(
            chapter_index=1, dramatic_question="?", required_plot_beats=["x"],
            target_tension=0.5,
        )],
        hook_schedule=hooks,
    ))


def test_unpaid_hook_finds_overdue(sm):
    _seed_hook_skeleton(sm, [
        HookPlacement(hook_id="fs:1", planted_by_chapter=1, paid_off_by_chapter=2),
        HookPlacement(hook_id="fs:2", planted_by_chapter=1, paid_off_by_chapter=5),
    ])
    _seed_rollout_with_scores(sm, "r1", {
        1: {"a": 0.5}, 2: {"a": 0.5}, 3: {"a": 0.5},
    })
    # Mark fs:2 as paid (early — fine). fs:1 not paid by deadline (2).
    sm.save_hook_payoff(
        quest_id="q1", rollout_id="r1", hook_id="fs:2",
        paid_off_at_chapter=2,
    )
    sel = UnpaidHookSelector(sm)
    targets = sel.select(quest_id="q1")
    assert len(targets) == 1
    assert targets[0].chapter_index == 2  # the deadline chapter
    assert "fs:1" in targets[0].reason
    assert "bell tolls" in targets[0].guidance


def test_unpaid_hook_skips_future_deadline(sm):
    _seed_hook_skeleton(sm, [
        HookPlacement(hook_id="fs:1", planted_by_chapter=1, paid_off_by_chapter=10),
    ])
    _seed_rollout_with_scores(sm, "r1", {1: {"a": 0.5}, 2: {"a": 0.5}})
    sel = UnpaidHookSelector(sm)
    targets = sel.select(quest_id="q1")
    # Only 2 chapters in; deadline is 10 → not yet overdue
    assert targets == []


def test_unpaid_hook_skips_paid(sm):
    _seed_hook_skeleton(sm, [
        HookPlacement(hook_id="fs:1", planted_by_chapter=1, paid_off_by_chapter=2),
    ])
    _seed_rollout_with_scores(sm, "r1", {1: {"a": 0.5}, 2: {"a": 0.5}})
    sm.save_hook_payoff(
        quest_id="q1", rollout_id="r1", hook_id="fs:1", paid_off_at_chapter=2,
    )
    sel = UnpaidHookSelector(sm)
    assert sel.select(quest_id="q1") == []


# ---- SiblingOutscoredSelector ---------------------------------------------

def test_sibling_outscored_finds_better_sibling(sm):
    _seed_rollout_with_scores(sm, "r1", {
        1: {"voice_distinctiveness": 0.5},
    }, proses={1: "Bad chapter prose."})
    _seed_rollout_with_scores(sm, "r2", {
        1: {"voice_distinctiveness": 0.8},  # +0.3 over r1
    }, proses={1: "Good chapter prose with rich voice."})
    sel = SiblingOutscoredSelector(sm, min_delta=0.15)
    targets = sel.select(quest_id="q1", rollout_id="r1")
    assert len(targets) == 1
    assert targets[0].rollout_id == "r1"
    assert targets[0].chapter_index == 1
    assert "+0.30" in targets[0].reason
    # Sibling prose appears in guidance for the LLM to crib from
    assert "Good chapter prose" in targets[0].guidance


def test_sibling_outscored_requires_min_delta(sm):
    _seed_rollout_with_scores(sm, "r1", {1: {"a": 0.5}}, proses={1: "x"})
    _seed_rollout_with_scores(sm, "r2", {1: {"a": 0.55}}, proses={1: "y"})  # only +0.05
    sel = SiblingOutscoredSelector(sm, min_delta=0.15)
    assert sel.select(quest_id="q1", rollout_id="r1") == []


def test_sibling_outscored_no_siblings(sm):
    _seed_rollout_with_scores(sm, "r1", {1: {"a": 0.5}}, proses={1: "x"})
    sel = SiblingOutscoredSelector(sm)
    assert sel.select(quest_id="q1") == []
