"""Tests for the Phase-2 Claude v2 SFT rater."""
from __future__ import annotations

import json
from pathlib import Path

from tools.sft.claude_rater_v2 import (
    _build_rationale,
    _cliche_count,
    _has_dialogue,
    _pov_drift,
    _extract_pov_mode,
    pick_best,
    pick_and_persist,
    score_candidate,
)


def _make_cand(index: int, prose: str, overall: float = 0.7) -> dict:
    return {
        "index": index,
        "prose": prose,
        "overall_score": overall,
        "weighted_score": overall,
    }


def test_cliche_count_detects_known_phrases():
    assert _cliche_count("Her heart pounded as the weight settled.") == 2
    assert _cliche_count("She walked through the door.") == 0


def test_has_dialogue_detects_quoted_line():
    assert _has_dialogue('He said, "tell me the truth."')
    assert _has_dialogue('"Yes," she answered.')
    assert not _has_dialogue("He said tell me the truth.")


def test_has_dialogue_handles_curly_quotes():
    assert _has_dialogue("He said, \u201ctell me the truth.\u201d")


def test_pov_drift_second_person_flags_i_and_my():
    text = "You walk into the room. I notice my hand shaking."
    assert _pov_drift(text, "second") >= 2


def test_pov_drift_clean_second_person():
    text = "You walk into the room and close the door behind you."
    assert _pov_drift(text, "second") == 0


def test_pov_drift_ignores_dialogue():
    text = 'You walk in. "I was here already," she says.'
    # "I" inside quotes should not count as drift
    assert _pov_drift(text, "second") == 0


def test_extract_pov_mode_defaults_to_second():
    assert _extract_pov_mode("") == "second"
    assert _extract_pov_mode("Write in first person past tense.") == "first"
    assert _extract_pov_mode("Write in second-person past tense.") == "second"


def test_score_candidate_prefers_concrete_over_mood():
    brief = "Write in second person past tense."
    concrete = _make_cand(
        0,
        "You set the cup on the bar. Merrin's knuckles whiten on the ledger "
        "and her pen stills. The lamp guttered once.",
    )
    mood = _make_cand(
        1,
        "You feel the tension in the air, the atmosphere thick with foreboding "
        "and unease. Every fibre of your being screams danger.",
    )
    s_concrete = score_candidate(concrete, brief)
    s_mood = score_candidate(mood, brief)
    assert s_concrete.score > s_mood.score


def test_score_candidate_penalises_foreign_token_leak():
    brief = "Write in second person past tense."
    clean = _make_cand(0, "You walked into the inn and closed the door.")
    leak = _make_cand(1, "You walked into the inn 然后 you closed the door.")
    s_clean = score_candidate(clean, brief)
    s_leak = score_candidate(leak, brief)
    assert s_leak.foreign_penalty > 0
    assert s_clean.score > s_leak.score


def test_pick_best_breaks_ties_by_lowest_index():
    brief = "Write in second person past tense."
    record = {
        "craft_brief": brief,
        "candidates": [
            _make_cand(0, "You walked and the rain fell."),
            _make_cand(1, "You walked and the rain fell."),
        ],
    }
    winner, _reports = pick_best(record)
    assert winner.index == 0


def test_pick_best_prefers_dialogue_when_brief_calls_for_it():
    brief = "This is a dialogue scene. Have the protagonist ask a question."
    cand_no_dialogue = _make_cand(
        0, "You stood at the bar, watching Merrin polish a glass."
    )
    cand_dialogue = _make_cand(
        1, 'You stood at the bar. "Who paid you?" you asked her.'
    )
    record = {"craft_brief": brief, "candidates": [cand_no_dialogue, cand_dialogue]}
    winner, _ = pick_best(record)
    assert winner.index == 1


def test_pick_and_persist_writes_sidecar(tmp_path: Path):
    src = tmp_path / "u1_s1_demo.json"
    record = {
        "quest_id": "demo",
        "update_number": 1,
        "scene_index": 1,
        "craft_brief": "Write in second person past tense.",
        "candidates": [
            _make_cand(0, "You turned toward the window. The street was quiet."),
            _make_cand(1, "You turned 然后 the street was quiet."),
        ],
    }
    src.write_text(json.dumps(record))
    dst = pick_and_persist(src)
    assert dst.exists()
    out = json.loads(dst.read_text())
    assert out["claude_pick"]["chosen_index"] == 0
    assert "rationale" in out["claude_pick"]
    assert out["claude_pick"]["model"].startswith("claude-opus-4-6")
    all_scores = out["claude_pick"]["all_scores"]
    assert len(all_scores) == 2
    for row in all_scores:
        assert "score" in row
        assert "cliche_count" in row


def test_build_rationale_marks_weak_field():
    reports = [
        type("R", (), {"score": -2.0, "index": 0, "rationale_bits": ["clean POV"]})(),
        type("R", (), {"score": -2.3, "index": 1, "rationale_bits": []})(),
    ]
    r = _build_rationale(reports[0], reports)
    assert "weak" in r.lower() or "least" in r.lower()
