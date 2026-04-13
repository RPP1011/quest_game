from __future__ import annotations
from .schemas import Arc, Structure


def global_progress(arc: Arc, structure: Structure) -> float:
    n = len(structure.phases)
    if n == 0:
        return 0.0
    phase_size = 1.0 / n
    idx = min(arc.current_phase_index, n - 1)
    start = idx * phase_size
    prog = arc.phase_progress if arc.current_phase_index < n else 1.0
    return start + phase_size * prog


def tension_target(arc: Arc, structure: Structure) -> float:
    pos = global_progress(arc, structure)
    curve = structure.tension_curve
    if not curve:
        return 0.5
    if pos <= curve[0][0]:
        return curve[0][1]
    if pos >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= pos <= x1:
            if x1 == x0:
                return y0
            t = (pos - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return curve[-1][1]


def tension_gap(arc: Arc, structure: Structure, window: int = 3) -> float:
    """Target minus the average of the last `window` observed readings.

    Positive gap = story is lagging behind the target curve.
    Negative gap = story is hotter than the target curve (often fine).
    Returns the full target when there are no observations.
    """
    target = tension_target(arc, structure)
    if not arc.tension_observed:
        return target
    recent = arc.tension_observed[-window:]
    avg = sum(v for _, v in recent) / len(recent)
    return target - avg


def advance_phase(arc: Arc, structure: Structure) -> Arc:
    last_index = len(structure.phases) - 1
    if arc.current_phase_index >= last_index:
        return arc.model_copy(update={
            "current_phase_index": last_index, "phase_progress": 1.0,
        })
    return arc.model_copy(update={
        "current_phase_index": arc.current_phase_index + 1,
        "phase_progress": 0.0,
    })
