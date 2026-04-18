from __future__ import annotations
import pytest
from app.world.db import open_db
from app.world.state_manager import WorldStateManager


@pytest.fixture
def sm(tmp_path):
    conn = open_db(tmp_path / "test.db")
    wsm = WorldStateManager(conn)
    yield wsm
    conn.close()


def test_create_and_get_foreshadow_triple(sm):
    sm.create_foreshadow_triple(
        id="ft_abc12345",
        hook_id="fs:pistol",
        foreshadow_text="Tristan notices the pistol's unusual weight",
        trigger_pred={"type": "chapter_gte", "value": 5},
        payoff_text="The pistol fires unexpectedly, revealing its cursed nature",
        planted_chapter=2,
        deadline_chapter=8,
    )
    triple = sm.get_foreshadow_triple("ft_abc12345")
    assert triple["hook_id"] == "fs:pistol"
    assert triple["status"] == "planted"
    assert triple["trigger_pred"] == {"type": "chapter_gte", "value": 5}
    assert triple["deadline_chapter"] == 8
    assert triple["verified_planted"] is None


def test_update_foreshadow_triple_status(sm):
    sm.create_foreshadow_triple(
        id="ft_abc12345",
        hook_id="fs:pistol",
        foreshadow_text="pistol weight",
        trigger_pred={"type": "chapter_gte", "value": 5},
        payoff_text="pistol fires",
        planted_chapter=2,
        deadline_chapter=8,
    )
    sm.update_foreshadow_triple("ft_abc12345", status="triggered")
    assert sm.get_foreshadow_triple("ft_abc12345")["status"] == "triggered"

    sm.update_foreshadow_triple("ft_abc12345", verified_planted=0.85)
    assert sm.get_foreshadow_triple("ft_abc12345")["verified_planted"] == pytest.approx(0.85)


def test_list_foreshadow_triples_by_status(sm):
    for i, status in enumerate(["planted", "planted", "triggered", "paid_off"]):
        sm.create_foreshadow_triple(
            id=f"ft_{i:08d}",
            hook_id=f"fs:hook{i}",
            foreshadow_text=f"text {i}",
            trigger_pred={"type": "chapter_gte", "value": i + 1},
            payoff_text=f"payoff {i}",
            planted_chapter=1,
        )
        if status != "planted":
            sm.update_foreshadow_triple(f"ft_{i:08d}", status=status)

    planted = sm.list_foreshadow_triples(status="planted")
    assert len(planted) == 2
    triggered = sm.list_foreshadow_triples(status="triggered")
    assert len(triggered) == 1


def test_list_overdue_foreshadow_triples(sm):
    sm.create_foreshadow_triple(
        id="ft_overdue1",
        hook_id="fs:overdue",
        foreshadow_text="overdue hook",
        trigger_pred={"type": "chapter_gte", "value": 3},
        payoff_text="should have fired",
        planted_chapter=1,
        deadline_chapter=5,
    )
    overdue = sm.list_overdue_foreshadow_triples(current_chapter=6)
    assert len(overdue) == 1
    assert overdue[0]["id"] == "ft_overdue1"

    not_overdue = sm.list_overdue_foreshadow_triples(current_chapter=4)
    assert len(not_overdue) == 0


from app.planning.foreshadow_pool import evaluate_predicate


def test_chapter_gte_predicate():
    pred = {"type": "chapter_gte", "value": 5}
    state = {"current_chapter": 4, "active_entities": [], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False
    state["current_chapter"] = 5
    assert evaluate_predicate(pred, state) is True


def test_entity_active_predicate():
    pred = {"type": "entity_active", "entity_id": "char:cozme"}
    state = {"current_chapter": 1, "active_entities": ["char:tristan"], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False
    state["active_entities"].append("char:cozme")
    assert evaluate_predicate(pred, state) is True


def test_entity_present_predicate():
    pred = {"type": "entity_present", "entity_id": "char:cozme"}
    state = {"current_chapter": 1, "active_entities": [], "present_entities": ["char:tristan"], "events": []}
    assert evaluate_predicate(pred, state) is False
    state["present_entities"].append("char:cozme")
    assert evaluate_predicate(pred, state) is True


def test_event_occurred_predicate():
    pred = {"type": "event_occurred", "event": "tristan_confronts_cozme"}
    state = {"current_chapter": 1, "active_entities": [], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False
    state["events"].append("tristan_confronts_cozme")
    assert evaluate_predicate(pred, state) is True


def test_compound_and_predicate():
    pred = {
        "type": "and",
        "children": [
            {"type": "chapter_gte", "value": 3},
            {"type": "entity_active", "entity_id": "char:cozme"},
        ],
    }
    state = {"current_chapter": 3, "active_entities": [], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is False
    state["active_entities"].append("char:cozme")
    assert evaluate_predicate(pred, state) is True


def test_compound_or_predicate():
    pred = {
        "type": "or",
        "children": [
            {"type": "chapter_gte", "value": 10},
            {"type": "entity_active", "entity_id": "char:cozme"},
        ],
    }
    state = {"current_chapter": 3, "active_entities": ["char:cozme"], "present_entities": [], "events": []}
    assert evaluate_predicate(pred, state) is True


@pytest.mark.asyncio
async def test_verify_prose_contains_element():
    """Test the verification function structure (mocked LLM)."""
    from app.planning.foreshadow_pool import verify_prose_reference
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.chat_with_logprobs = AsyncMock()

    # Simulate high-confidence YES
    mock_logprob = MagicMock()
    mock_logprob.token = "YES"
    mock_logprob.logprob = -0.1  # ~0.90 probability
    mock_logprob.top_logprobs = {"YES": -0.1, "NO": -2.3}
    mock_result = MagicMock()
    mock_result.content = "YES"
    mock_result.token_logprobs = [mock_logprob]
    mock_client.chat_with_logprobs.return_value = mock_result

    confidence = await verify_prose_reference(
        client=mock_client,
        element_text="Tristan notices the pistol's unusual weight",
        prose="He hefted the pistol. It was heavier than it should have been.",
    )
    assert confidence > 0.8
    mock_client.chat_with_logprobs.assert_called_once()


@pytest.mark.asyncio
async def test_verify_prose_low_confidence():
    """Test low confidence when prose doesn't reference element."""
    from app.planning.foreshadow_pool import verify_prose_reference
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.chat_with_logprobs = AsyncMock()

    mock_logprob = MagicMock()
    mock_logprob.token = "NO"
    mock_logprob.logprob = -0.2
    mock_logprob.top_logprobs = {"YES": -3.0, "NO": -0.2}
    mock_result = MagicMock()
    mock_result.content = "NO"
    mock_result.token_logprobs = [mock_logprob]
    mock_client.chat_with_logprobs.return_value = mock_result

    confidence = await verify_prose_reference(
        client=mock_client,
        element_text="Tristan notices the pistol's unusual weight",
        prose="The sun was setting over the harbor.",
    )
    assert confidence < 0.3
