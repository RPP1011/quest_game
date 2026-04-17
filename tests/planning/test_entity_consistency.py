from __future__ import annotations

from app.planning.entity_consistency_critic import check_entity_consistency
from app.planning.opening_critic import check_opening_repetition
from app.world.schema import Entity, EntityType


# --- Entity consistency ---

def test_catches_vessel_type_mismatch():
    entities = [Entity(
        id="loc:ship", entity_type=EntityType.LOCATION, name="Bluebell",
        data={"description": "A sturdy old cog."},
    )]
    issues = check_entity_consistency(
        "The Bluebell, a shallow-bellied skiff, fought the current.", entities,
    )
    assert len(issues) == 1
    assert "cog" in issues[0]["message"]
    assert "skiff" in issues[0]["message"]


def test_no_issue_when_correct_vessel():
    entities = [Entity(
        id="loc:ship", entity_type=EntityType.LOCATION, name="Bluebell",
        data={"description": "A sturdy old cog."},
    )]
    issues = check_entity_consistency(
        "The Bluebell, a heavy cog, rolled in the swell.", entities,
    )
    assert issues == []


def test_catches_character_attribute_mismatch():
    entities = [Entity(
        id="char:x", entity_type=EntityType.CHARACTER, name="Tristan",
        data={"description": "A thin, dark-haired man."},
    )]
    issues = check_entity_consistency(
        "Tristan, broad and blonde, stepped forward.", entities,
    )
    # Should flag both hair and build
    assert len(issues) >= 1
    assert any("dark-haired" in i["message"] for i in issues)


def test_no_issue_when_attributes_match():
    entities = [Entity(
        id="char:x", entity_type=EntityType.CHARACTER, name="Tristan",
        data={"description": "A thin, dark-haired man."},
    )]
    issues = check_entity_consistency(
        "Tristan, thin and dark-haired, moved through the shadows.", entities,
    )
    assert issues == []


def test_no_issue_when_entity_not_mentioned():
    entities = [Entity(
        id="loc:x", entity_type=EntityType.LOCATION, name="Bluebell",
        data={"description": "A cog."},
    )]
    issues = check_entity_consistency("The ship sailed on.", entities)
    assert issues == []


# --- Opening repetition ---

def test_catches_repeated_opening():
    current = "The Bluebell did not merely roll; it bucked."
    priors = [
        "The Bluebell did not merely float; it fought the current.",
    ]
    issues = check_opening_repetition(current, priors)
    assert len(issues) == 1
    assert issues[0]["jaccard"] > 0.2


def test_no_issue_with_different_openings():
    current = "Angharad stood at the prow, watching the horizon."
    priors = [
        "The Bluebell did not merely float; it fought.",
        "Tristan sat in the corner where shadows pooled.",
    ]
    issues = check_opening_repetition(current, priors)
    assert issues == []


def test_no_issue_with_too_few_priors():
    current = "The Bluebell bucked."
    issues = check_opening_repetition(current, [], min_prior=1)
    assert issues == []
