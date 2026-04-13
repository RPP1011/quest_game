# tests/craft/test_arc.py
import pytest
from app.craft.arc import (
    advance_phase, global_progress, tension_gap, tension_target,
)
from app.craft.schemas import Arc, ArcPhase, Structure


@pytest.fixture
def s():
    return Structure(
        id="s", name="s", description="x", scales=["scene"],
        phases=[
            ArcPhase(name="a", position=0, tension_target=0.2, description="x"),
            ArcPhase(name="b", position=1, tension_target=0.7, description="x"),
            ArcPhase(name="c", position=2, tension_target=0.3, description="x"),
        ],
        tension_curve=[(0.0, 0.1), (0.5, 0.7), (1.0, 0.2)],
    )


def test_global_progress_inside_first_phase(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=0, phase_progress=0.5)
    # 3 phases, so each is 1/3. First phase midpoint = 1/6.
    assert abs(global_progress(a, s) - (1 / 6)) < 1e-6


def test_global_progress_at_phase_boundary(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0)
    assert abs(global_progress(a, s) - (1 / 3)) < 1e-6


def test_tension_target_interpolates(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0)
    # global_progress = 1/3 ≈ 0.333, curve from (0.0, 0.1) to (0.5, 0.7)
    # interpolated ≈ 0.1 + (0.333/0.5)*(0.7-0.1) ≈ 0.1 + 0.4 = 0.5
    assert abs(tension_target(a, s) - 0.5) < 0.02


def test_tension_gap_uses_recent_observations(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0,
            tension_observed=[(1, 0.2), (2, 0.25), (3, 0.3)])
    # target ≈ 0.5, avg of last 3 observed = 0.25, gap = 0.25
    gap = tension_gap(a, s, window=3)
    assert abs(gap - 0.25) < 0.02


def test_tension_gap_no_observations_returns_target(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.0)
    gap = tension_gap(a, s, window=3)
    assert abs(gap - tension_target(a, s)) < 1e-6


def test_advance_phase_resets_progress(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=1, phase_progress=0.8)
    b = advance_phase(a, s)
    assert b.current_phase_index == 2
    assert b.phase_progress == 0.0


def test_advance_phase_clamps_at_last(s):
    a = Arc(id="a", name="a", scale="scene", structure_id="s",
            current_phase_index=2, phase_progress=0.5)
    b = advance_phase(a, s)
    assert b.current_phase_index == 2  # clamped
    assert b.phase_progress == 1.0
