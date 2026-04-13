"""Tests for app/planning/perception.py — Gap G9 detail grounding."""
from __future__ import annotations

from app.planning.perception import (
    current_preoccupations,
    default_detail_principle,
    perceptual_profile_for,
)
from app.planning.schemas import DetailPrinciple, PerceptualProfile
from app.world.schema import Entity, EntityType


def _char(id_: str, **data) -> Entity:
    return Entity(id=id_, entity_type=EntityType.CHARACTER, name=id_, data=data)


def test_perceptual_profile_for_none_cases():
    assert perceptual_profile_for(None) is None
    assert perceptual_profile_for(_char("a")) is None
    loc = Entity(
        id="here", entity_type=EntityType.LOCATION, name="here",
        data={"perception": {"permanent_preoccupations": ["exits"]}},
    )
    assert perceptual_profile_for(loc) is None


def test_perceptual_profile_for_parses_dict():
    ent = _char("sol", perception={
        "permanent_preoccupations": ["exits", "sightlines", "boot quality"],
        "emotional_preoccupations": {"dread": ["footsteps", "hands"]},
        "detail_mode": "precise",
        "triple_duty_targets": ["wounds that echo the theme"],
    })
    pp = perceptual_profile_for(ent)
    assert isinstance(pp, PerceptualProfile)
    assert pp.permanent_preoccupations == ["exits", "sightlines", "boot quality"]
    assert pp.emotional_preoccupations == {"dread": ["footsteps", "hands"]}
    assert pp.detail_mode == "precise"


def test_current_preoccupations_combines_permanent_and_emotion():
    pp = PerceptualProfile(
        permanent_preoccupations=["exits", "sightlines"],
        emotional_preoccupations={
            "dread": ["footsteps", "exits"],    # note dupe with permanent
            "grief": ["absences"],
        },
    )
    got = current_preoccupations(pp, primary_emotion="dread")
    # dedup: exits appears once; footsteps added
    assert got[:2] == ["exits", "sightlines"]
    assert "footsteps" in got
    assert got.count("exits") == 1


def test_current_preoccupations_handles_secondary_and_unknown_emotion():
    pp = PerceptualProfile(
        permanent_preoccupations=["exits"],
        emotional_preoccupations={"dread": ["footsteps"], "relief": ["sky"]},
    )
    got = current_preoccupations(
        pp, primary_emotion="dread", secondary_emotion="relief",
    )
    assert "footsteps" in got and "sky" in got
    # unknown emotion is a no-op
    got2 = current_preoccupations(pp, primary_emotion="nonsense")
    assert got2 == ["exits"]


def test_current_preoccupations_case_insensitive_match():
    pp = PerceptualProfile(
        permanent_preoccupations=[],
        emotional_preoccupations={"Dread": ["footsteps"]},
    )
    got = current_preoccupations(pp, primary_emotion="DREAD")
    assert got == ["footsteps"]


def test_default_detail_principle_is_grounded():
    pp = PerceptualProfile(
        permanent_preoccupations=["exits", "sightlines"],
        emotional_preoccupations={"dread": ["footsteps"]},
        detail_mode="precise",
        triple_duty_targets=["wound=history=theme"],
    )
    dp = default_detail_principle(
        "hero", pp, primary_emotion="dread",
    )
    assert isinstance(dp, DetailPrinciple)
    assert dp.perceiving_character_id == "hero"
    assert "exits" in dp.perceptual_preoccupations
    assert "footsteps" in dp.perceptual_preoccupations
    assert dp.triple_duty_targets == ["wound=history=theme"]
    # detail_mode maps into the plan enum
    assert dp.detail_mode in {
        "character_revealing", "world_establishing", "thematic_resonant",
        "mood_setting", "foreshadowing", "ironic",
    }
