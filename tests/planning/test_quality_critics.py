"""Tests for heuristic quality critics distilled from the voter-rollout experiment."""
from app.planning.critics import (
    validate_action_fidelity,
    validate_named_entity_presence,
    validate_pov_adherence,
)


# ---- POV ----

def test_pov_adherence_clean_second_person():
    prose = "You step into the hall. The door closes behind you. You wait."
    assert validate_pov_adherence(prose) == []


def test_pov_adherence_warns_on_first_person():
    prose = "I watch the corners, feel the weight of choices pressing against me."
    issues = validate_pov_adherence(prose)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "POV drift" in issues[0].message


def test_pov_adherence_handles_empty_prose():
    # neither 'you' nor 'I' → no signal to judge; don't flag.
    assert validate_pov_adherence("The fire burned low.") == []


def test_pov_adherence_mixed_but_dominant_second_person():
    # Real quest prose occasionally has "I" in dialogue; 0.9+ ratio is fine.
    prose = (
        "You step forward. 'I will return,' she said. "
        "You hear her footsteps fade and you keep walking and you do not look back."
    )
    assert validate_pov_adherence(prose, min_ratio=0.7) == []


# ---- named entity presence ----

def test_named_entity_presence_hit():
    prose = "Captain Marek stood at the map. Helga poured another drink."
    assert validate_named_entity_presence(prose, ["Marek", "Helga"]) == []


def test_named_entity_presence_missing():
    prose = "You move through the shadows. Every breath is a test."
    issues = validate_named_entity_presence(prose, ["Marek", "Helga"])
    assert len(issues) == 1
    assert "active named entity" in issues[0].message.lower()


def test_named_entity_presence_empty_list_no_issue():
    assert validate_named_entity_presence("any prose here", []) == []


# ---- action fidelity ----

def test_action_fidelity_match():
    prose = "You confront Marek about the orders and demand the truth."
    assert validate_action_fidelity(
        prose, "Confront Marek about the orders", min_ratio=0.25
    ) == []


def test_action_fidelity_miss():
    prose = "You move through shadows, breath steady, choices pressing."
    issues = validate_action_fidelity(
        prose, "Confront Marek about the hidden orders", min_ratio=0.25
    )
    assert len(issues) == 1
    assert "Action fidelity" in issues[0].message


def test_action_fidelity_stopwords_ignored():
    # action is almost all stopwords
    prose = "Unrelated narration with no content overlap."
    assert validate_action_fidelity(prose, "with this", min_ratio=0.25) == []
