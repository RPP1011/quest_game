from pathlib import Path
import pytest
from app.craft.library import CraftLibrary
from app.craft.schemas import Arc


ROOT = Path(__file__).parent.parent.parent / "app" / "craft" / "data"


@pytest.fixture
def lib():
    return CraftLibrary(ROOT)


def _arc(phase_index: int, gap_hint: str = "lagging", phase_progress: float = 0.5):
    """Build an arc whose observed tension forces a given gap polarity."""
    if gap_hint == "lagging":
        observed = [(1, 0.1), (2, 0.15), (3, 0.2)]  # well below most targets
    elif gap_hint == "hot":
        observed = [(1, 0.95), (2, 0.95), (3, 0.95)]  # well above most targets
    else:
        observed = []
    return Arc(id="a", name="a", scale="chapter", structure_id="three_act",
               current_phase_index=phase_index, phase_progress=phase_progress,
               tension_observed=observed)


def test_recommend_pulls_from_phase_expected_beats(lib):
    arc = _arc(phase_index=0, gap_hint="neutral")  # setup phase
    rec = lib.recommend_tools(arc, lib.structure("three_act"))
    ids = [t.id for t in rec]
    # setup expects chekhovs_gun and scene_sequel
    assert "chekhovs_gun" in ids
    assert "scene_sequel" in ids


def test_recommend_boosts_tension_tools_when_lagging(lib):
    arc = _arc(phase_index=2, gap_hint="lagging")  # midpoint, tension too low
    rec = lib.recommend_tools(arc, lib.structure("three_act"))
    ids = [t.id for t in rec]
    # Expect reversal/midpoint_shift/false_victory ranking high
    assert "reversal" in ids or "midpoint_shift" in ids or "false_victory" in ids


def test_recommend_penalizes_recent_tools(lib):
    arc = _arc(phase_index=1, gap_hint="lagging")
    baseline = [t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"), recent_tool_ids=None, limit=6)]
    with_recent = [t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"),
        recent_tool_ids=["try_fail_cycle"], limit=6)]
    # try_fail_cycle should drop or disappear
    if "try_fail_cycle" in baseline and "try_fail_cycle" in with_recent:
        assert with_recent.index("try_fail_cycle") > baseline.index("try_fail_cycle")
    else:
        assert "try_fail_cycle" not in with_recent


def test_recommend_honors_required_beats(lib):
    arc = _arc(phase_index=0, gap_hint="neutral")
    arc = arc.model_copy(update={"required_beats_remaining": ["false_victory"]})
    rec = lib.recommend_tools(arc, lib.structure("three_act"))
    assert rec[0].id == "false_victory"


def test_recommend_limit(lib):
    arc = _arc(phase_index=1, gap_hint="lagging")
    rec = lib.recommend_tools(arc, lib.structure("three_act"), limit=2)
    assert len(rec) == 2
