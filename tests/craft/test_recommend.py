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


def test_recommend_boosts_theme_anchored_tools(lib):
    """A theme whose key_scenes matches the current phase should boost tools
    that serve that phase (Gap G4: theme-based tool scoring)."""
    from app.planning.world_extensions import Theme

    structure = lib.structure("three_act")
    phase_name = structure.phases[0].name

    # Arc with no explicit recent tools; neutral tension so phase_expected
    # bonus is the main signal. Add a theme whose key_scenes references the
    # current phase — expected-beat tools should gain a +3 bonus.
    arc = _arc(phase_index=0, gap_hint="neutral")
    theme = Theme(
        id="t:anchor",
        proposition="the world has a debt that must be named",
        stance="exploring",
        key_scenes=[phase_name],
    )

    baseline = lib.recommend_tools(arc, structure, limit=10)
    boosted = lib.recommend_tools(arc, structure, limit=10, themes=[theme])

    baseline_ids = [t.id for t in baseline]
    boosted_ids = [t.id for t in boosted]

    # Every expected-beat tool from this phase should still be in the boosted
    # output — and among the top results (since its score gained +3).
    expected = set(structure.phases[0].expected_beats)
    anchored = expected & set(baseline_ids)
    assert anchored, "precondition: phase has expected-beat tools"
    # Anchored tool(s) must appear no later in the boosted list than baseline.
    for tid in anchored:
        assert boosted_ids.index(tid) <= baseline_ids.index(tid)


def test_recommend_no_theme_match_is_noop(lib):
    from app.planning.world_extensions import Theme

    arc = _arc(phase_index=0, gap_hint="neutral")
    theme = Theme(id="t:misc", proposition="x", key_scenes=["nonexistent-phase"])
    without = [t.id for t in lib.recommend_tools(arc, lib.structure("three_act"))]
    with_theme = [t.id for t in lib.recommend_tools(
        arc, lib.structure("three_act"), themes=[theme],
    )]
    assert without == with_theme


def test_recommend_limit(lib):
    arc = _arc(phase_index=1, gap_hint="lagging")
    rec = lib.recommend_tools(arc, lib.structure("three_act"), limit=2)
    assert len(rec) == 2
