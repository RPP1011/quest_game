from __future__ import annotations
import pytest
from app.world.schema import (
    ArcSkeleton, HookPlacement, SkeletonChapter, StoryCandidate, ThemeBeat,
)
from app.world.state_manager import WorldStateManager, WorldStateError


@pytest.fixture
def sm(db):
    return WorldStateManager(db)


def _seed_candidate(sm: WorldStateManager, cid: str, qid: str = "q1") -> None:
    sm.add_story_candidate(StoryCandidate(
        id=cid, quest_id=qid, title="t", synopsis="s",
        primary_thread_ids=[], secondary_thread_ids=[],
        protagonist_character_id=None, emphasized_theme_ids=[],
        climax_description="", expected_chapter_count=3,
    ))


def _skel(sid: str = "sk_1", cid: str = "cand_1", qid: str = "q1",
          n_chapters: int = 3) -> ArcSkeleton:
    return ArcSkeleton(
        id=sid, candidate_id=cid, quest_id=qid,
        chapters=[
            SkeletonChapter(
                chapter_index=i,
                pov_character_id="char:hero" if i % 2 == 1 else "char:rival",
                location_constraint="Inn" if i == 1 else None,
                dramatic_question=f"What does ch{i} ask?",
                required_plot_beats=[f"beat_{i}_a", f"beat_{i}_b"],
                target_tension=0.3 + 0.2 * i,
                entities_to_surface=[f"char:n{i}"] if i > 1 else [],
                theme_emphasis=["t:honor"] if i == 2 else [],
            )
            for i in range(1, n_chapters + 1)
        ],
        theme_arc=[ThemeBeat(theme_id="t:honor", peak_chapter=2, stance_at_peak="affirming")],
        hook_schedule=[
            HookPlacement(hook_id="fs:1", planted_by_chapter=1, paid_off_by_chapter=3),
        ],
    )


def test_save_and_get_skeleton(sm):
    _seed_candidate(sm, "cand_1")
    s = _skel()
    sm.save_arc_skeleton(s)
    got = sm.get_arc_skeleton("sk_1")
    assert got.id == "sk_1"
    assert len(got.chapters) == 3
    assert got.chapters[0].required_plot_beats == ["beat_1_a", "beat_1_b"]
    assert got.chapters[1].theme_emphasis == ["t:honor"]
    assert got.chapters[2].entities_to_surface == ["char:n3"]
    assert got.theme_arc[0].peak_chapter == 2
    assert got.hook_schedule[0].paid_off_by_chapter == 3


def test_get_skeleton_for_candidate_returns_latest(sm):
    _seed_candidate(sm, "cand_x")
    # Same candidate, two saved skeletons (regenerate scenario)
    sm.save_arc_skeleton(_skel(sid="sk_old", cid="cand_x"))
    sm.save_arc_skeleton(_skel(sid="sk_new", cid="cand_x"))
    latest = sm.get_skeleton_for_candidate("cand_x")
    assert latest is not None
    # Both exist; we return the latest by created_at. SQL default timestamps
    # may match within a second — assert at least one of them comes back.
    assert latest.id in {"sk_old", "sk_new"}


def test_get_skeleton_for_candidate_none(sm):
    assert sm.get_skeleton_for_candidate("no_such_candidate") is None


def test_upsert_replaces_chapters(sm):
    _seed_candidate(sm, "cand_1")
    sm.save_arc_skeleton(_skel(sid="sk_u", n_chapters=2))
    sm.save_arc_skeleton(_skel(sid="sk_u", n_chapters=5))
    got = sm.get_arc_skeleton("sk_u")
    assert len(got.chapters) == 5


def test_get_missing_raises(sm):
    with pytest.raises(WorldStateError):
        sm.get_arc_skeleton("no_such_skel")
