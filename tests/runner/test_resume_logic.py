"""Pure-logic tests for resume detection. No real pipeline, no LLM.

The narrative-row reader is injected as a callable so tests can supply
canned rows; production wires it to WorldStateManager.list_narrative.
"""
from app.runner_resume import (
    ResumeDecision,
    ResumeMismatchError,
    ConfigDriftError,
    WrongDatabaseError,
    decide_resume,
)


def _row(update_number, player_action):
    return {"update_number": update_number, "player_action": player_action}


ACTIONS = ["A1", "A2", "A3", "A4", "A5"]


def test_no_rows_means_fresh_start():
    decision = decide_resume(rows=[], actions=ACTIONS,
                             db_quest_id=None, config_quest_id="q")
    assert decision.start_from == 1
    assert decision.skipped == 0


def test_three_committed_rows_resume_at_four():
    rows = [_row(1, "A1"), _row(2, "A2"), _row(3, "A3")]
    decision = decide_resume(rows=rows, actions=ACTIONS,
                             db_quest_id="q", config_quest_id="q")
    assert decision.start_from == 4
    assert decision.skipped == 3


def test_action_drift_at_index_one_raises_mismatch():
    rows = [_row(1, "A1"), _row(2, "DIFFERENT"), _row(3, "A3")]
    try:
        decide_resume(rows=rows, actions=ACTIONS,
                      db_quest_id="q", config_quest_id="q")
    except ResumeMismatchError as e:
        assert e.index == 1
        assert e.db_action == "DIFFERENT"
        assert e.config_action == "A2"
    else:
        raise AssertionError("expected ResumeMismatchError")


def test_more_db_rows_than_config_actions_raises_drift():
    rows = [_row(i, f"A{i}") for i in range(1, 6)]
    short_actions = ["A1", "A2", "A3"]
    try:
        decide_resume(rows=rows, actions=short_actions,
                      db_quest_id="q", config_quest_id="q")
    except ConfigDriftError:
        return
    raise AssertionError("expected ConfigDriftError")


def test_wrong_quest_id_raises():
    rows = [_row(1, "A1")]
    try:
        decide_resume(rows=rows, actions=ACTIONS,
                      db_quest_id="other_quest", config_quest_id="q")
    except WrongDatabaseError as e:
        assert "other_quest" in str(e)
        assert "q" in str(e)
    else:
        raise AssertionError("expected WrongDatabaseError")


def test_quest_id_check_skipped_when_db_has_no_quest_id():
    # An empty DB (no arc/reader rows) reports db_quest_id=None.
    # We should not raise WrongDatabaseError on a fresh-bootstrapped DB.
    rows = []
    decision = decide_resume(rows=rows, actions=ACTIONS,
                             db_quest_id=None, config_quest_id="q")
    assert decision.start_from == 1


def test_resume_skips_using_max_update_number_not_count():
    # If a flagged row is at update_number=5 but only 3 committed rows
    # exist, MAX(update_number)=5 and start_from=6 — pipeline never
    # writes gaps, but document the behavior.
    rows = [_row(1, "A1"), _row(2, "A2"), _row(3, "A3"),
            _row(4, "A4"), _row(5, "A5")]
    decision = decide_resume(rows=rows, actions=ACTIONS,
                             db_quest_id="q", config_quest_id="q")
    assert decision.start_from == 6
    assert decision.skipped == 5
