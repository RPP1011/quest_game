"""Reader-model update logic (Gap G6).

Accumulates what the reader knows / expects / how long since the last major
beat by mutating :class:`ReaderState` from each committed :class:`DramaticPlan`.

The counters (`updates_since_major_event`, `updates_since_revelation`,
`updates_since_emotional_peak`) are best-effort heuristics driven off signals
available in the dramatic plan + optional emotional plan. They increment every
commit and reset when the corresponding event type is detected.
"""
from __future__ import annotations

import hashlib
from typing import Any

from app.planning.schemas import DramaticPlan
from app.world.schema import (
    Expectation,
    ExpectationStatus,
    OpenQuestion,
    ReaderState,
)


# Threshold: when `updates_since_major_event` exceeds this many commits without
# a major event, `recommend_tools` gets a patience boost. Default is 3.
DEFAULT_PATIENCE_THRESHOLD = 3


def _stable_id(prefix: str, text: str, update_number: int) -> str:
    digest = hashlib.sha1(f"{update_number}:{text}".encode()).hexdigest()[:8]
    return f"{prefix}_{update_number}_{digest}"


def _text_match(a: str, b: str) -> bool:
    """Loose match for closing questions / subverting expectations.

    The dramatic planner emits free-form strings so we cannot guarantee exact
    text matches between the opener and the closer. We normalize whitespace +
    case and accept either substring direction as a match. An explicit id
    match (``"id:<id>"`` prefix) short-circuits.
    """
    return a.strip().lower() == b.strip().lower() or (
        a.strip().lower() in b.strip().lower()
        or b.strip().lower() in a.strip().lower()
    )


def _detect_major_event(dramatic: DramaticPlan) -> bool:
    """Heuristic: a major event fires when tension is high, a thread resolves,
    or a scene outcome contains a rupture keyword."""
    if dramatic.update_tension_target >= 0.75:
        return True
    for adv in dramatic.thread_advances:
        if adv.advance_type in ("resolves", "resurfaces"):
            return True
    # Any scene whose tension is climactic or function mentions climax/death
    for scene in dramatic.scenes:
        if scene.tension_target >= 0.8:
            return True
        if any(kw in scene.dramatic_function.lower()
               for kw in ("climax", "rupture", "turning", "revelation")):
            return True
    return False


def _detect_revelation(dramatic: DramaticPlan) -> bool:
    """Heuristic: a revelation fires when any scene `reveals` is non-empty
    or closes a dramatic question."""
    if dramatic.questions_closed:
        return True
    for scene in dramatic.scenes:
        if scene.reveals:
            return True
    return False


def _detect_emotional_peak(dramatic: DramaticPlan, emotional: Any | None) -> bool:
    """Heuristic: peak fires when emotional plan (if present) has high
    intensity, or when tension target is high."""
    if emotional is not None:
        try:
            for sp in emotional.scenes:
                if getattr(sp, "intensity", 0.0) >= 0.8:
                    return True
        except AttributeError:
            pass
    if dramatic.update_tension_target >= 0.75:
        return True
    return False


def apply_dramatic_plan(
    state: ReaderState,
    dramatic: DramaticPlan,
    *,
    update_number: int,
    emotional: Any | None = None,
) -> ReaderState:
    """Return a new :class:`ReaderState` with `dramatic` accumulated.

    Pure function — caller is responsible for persisting the result.
    """
    # --- open_questions: append new, remove closed ---
    open_qs = list(state.open_questions)
    for q_text in dramatic.questions_opened:
        open_qs.append(OpenQuestion(
            id=_stable_id("q", q_text, update_number),
            text=q_text,
            priority=5,
            opened_at_update=update_number,
        ))
    for closer in dramatic.questions_closed:
        # Support "id:<id>" explicit match; fall back to loose text match.
        if closer.startswith("id:"):
            target_id = closer[3:].strip()
            open_qs = [q for q in open_qs if q.id != target_id]
        else:
            open_qs = [q for q in open_qs if not _text_match(q.text, closer)]

    # --- expectations: append pending; mark matching ones subverted ---
    expectations = list(state.expectations)
    for e_text in dramatic.expectations_set if hasattr(dramatic, "expectations_set") else []:
        expectations.append(Expectation(
            id=_stable_id("e", e_text, update_number),
            text=e_text,
            confidence=0.5,
            status=ExpectationStatus.PENDING,
            set_at_update=update_number,
        ))
    for sub_text in (dramatic.expectations_subverted
                     if hasattr(dramatic, "expectations_subverted") else []):
        updated: list[Expectation] = []
        matched_once = False
        for exp in expectations:
            if (not matched_once
                    and exp.status == ExpectationStatus.PENDING
                    and (sub_text.startswith("id:") and exp.id == sub_text[3:].strip()
                         or _text_match(exp.text, sub_text))):
                updated.append(exp.model_copy(update={"status": ExpectationStatus.SUBVERTED}))
                matched_once = True
            else:
                updated.append(exp)
        expectations = updated

    # --- counters: increment all three, then reset those whose event fired ---
    major = _detect_major_event(dramatic)
    revelation = _detect_revelation(dramatic)
    peak = _detect_emotional_peak(dramatic, emotional)

    since_major = 0 if major else state.updates_since_major_event + 1
    since_rev = 0 if revelation else state.updates_since_revelation + 1
    since_peak = 0 if peak else state.updates_since_emotional_peak + 1

    # --- emotional valence: follow scene tension as a crude proxy in [-1, 1] ---
    # Map tension target (0..1) onto valence (-1..1). High tension -> negative
    # valence (dread/fear); low tension -> positive (calm). Best-effort.
    valence = 1.0 - 2.0 * float(dramatic.update_tension_target)
    valence = max(-1.0, min(1.0, valence))

    return state.model_copy(update={
        "open_questions": open_qs,
        "expectations": expectations,
        "current_emotional_valence": valence,
        "updates_since_major_event": since_major,
        "updates_since_revelation": since_rev,
        "updates_since_emotional_peak": since_peak,
    })
