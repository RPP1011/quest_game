from __future__ import annotations
import pytest
from app.engine.typed_edits import apply_edits, EDIT_TYPES


def test_edit_types_taxonomy():
    assert "cliche" in EDIT_TYPES
    assert "forced_metaphor" in EDIT_TYPES
    assert "continuity_break" in EDIT_TYPES
    assert len(EDIT_TYPES) >= 10


def test_apply_single_edit():
    prose = "The odds were shifting beneath him like a gambler's last coin on the table."
    edits = [
        {
            "span_start": 0,
            "span_end": 71,
            "original_text": "The odds were shifting beneath him like a gambler's last coin on the",
            "edit_type": "forced_metaphor",
            "reason": "gambling family over budget",
            "replacement": "The ground was tilting beneath him like a floor giving way under the",
        }
    ]
    result = apply_edits(prose, edits)
    assert "ground was tilting" in result
    assert "odds were shifting" not in result


def test_apply_multiple_edits_reverse_order():
    prose = "AAA BBB CCC DDD EEE"
    edits = [
        {"span_start": 0, "span_end": 3, "original_text": "AAA", "edit_type": "cliche",
         "reason": "", "replacement": "XXX"},
        {"span_start": 8, "span_end": 11, "original_text": "CCC", "edit_type": "cliche",
         "reason": "", "replacement": "YYY"},
        {"span_start": 16, "span_end": 19, "original_text": "EEE", "edit_type": "cliche",
         "reason": "", "replacement": "ZZZ"},
    ]
    result = apply_edits(prose, edits)
    assert result == "XXX BBB YYY DDD ZZZ"


def test_apply_edits_empty_list():
    prose = "Unchanged prose."
    assert apply_edits(prose, []) == prose


def test_apply_edits_validates_original_text():
    prose = "The quick brown fox."
    edits = [
        {"span_start": 0, "span_end": 3, "original_text": "WRONG",
         "edit_type": "cliche", "reason": "", "replacement": "RIGHT"},
    ]
    result = apply_edits(prose, edits)
    assert result == prose  # unchanged — edit skipped
