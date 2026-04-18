from __future__ import annotations

from app.planning.metaphor_critic import check_metaphor_variety, _count_family, IMAGERY_FAMILIES


def test_catches_gambling_dominance():
    prose = (
        "The odds were shifting. The house always takes its cut. "
        "He felt the dice rolling in his head. The deck was stacked. "
        "A bad bet, a losing hand. The stakes were rising. "
        "He played the hand he was dealt. The pot was growing."
    )
    issues = check_metaphor_variety(prose, max_per_family=5)
    assert len(issues) >= 1
    gambling_issue = next(i for i in issues if i["family"] == "gambling")
    assert gambling_issue["count"] > 5
    assert "gambling" in gambling_issue["suggested_fix"]


def test_no_issue_when_variety():
    prose = (
        "The odds shifted. The tide was rising. "
        "A hawk circled. The weight pressed down. "
        "The gears turned. The flame flickered."
    )
    issues = check_metaphor_variety(prose, max_per_family=5)
    assert issues == []


def test_count_family_word_boundary():
    """'played the hand' should match but 'handle' should not."""
    count = _count_family(
        "He played the hand well. The handle broke.",
        IMAGERY_FAMILIES["gambling"],
    )
    # "played the hand" matches. "the hand" was removed (too ambiguous).
    # "handle" does NOT match. Total: 1.
    assert count == 1


def test_threshold_configurable():
    prose = "the odds the odds the odds"
    assert check_metaphor_variety(prose, max_per_family=2) != []
    assert check_metaphor_variety(prose, max_per_family=10) == []


def test_real_chapter_gambling_count():
    """Regression check: a chapter with known high gambling count."""
    # Simulated heavy-gambling prose (similar to rollout ch4)
    prose = " ".join([
        "The odds were thin.", "The house always wins.",
        "A bad bet.", "The deck was stacked.",
        "Roll the die.", "The stakes rose.",
    ] * 5)
    issues = check_metaphor_variety(prose, max_per_family=5)
    gambling_issues = [i for i in issues if i["family"] == "gambling"]
    assert len(gambling_issues) == 1
    assert gambling_issues[0]["count"] >= 25
