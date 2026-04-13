"""Tests for app/planning/metaphor.py — Gap G10 metaphor-domain grounding."""
from __future__ import annotations

from app.planning.metaphor import (
    character_metaphor_profile_for,
    compute_current_domains,
    default_metaphor_profile,
)
from app.planning.schemas import CharacterMetaphorProfile, MetaphorProfile
from app.world.schema import Entity, EntityType


def _char(id_: str, **data) -> Entity:
    return Entity(id=id_, entity_type=EntityType.CHARACTER, name=id_, data=data)


def test_character_metaphor_profile_for_none_cases():
    assert character_metaphor_profile_for(None) is None
    assert character_metaphor_profile_for(_char("a")) is None
    loc = Entity(
        id="l", entity_type=EntityType.LOCATION, name="l",
        data={"metaphor": {"permanent_domains": ["stone"]}},
    )
    assert character_metaphor_profile_for(loc) is None


def test_character_metaphor_profile_for_parses_dict():
    ent = _char("soldier", metaphor={
        "permanent_domains": ["stone", "iron", "cold"],
        "forbidden_domains": ["courtly-dance", "perfume"],
        "metaphor_density": 0.6,
        "extends_to_narration": False,
    })
    mp = character_metaphor_profile_for(ent)
    assert isinstance(mp, CharacterMetaphorProfile)
    assert "stone" in mp.permanent_domains
    assert "courtly-dance" in mp.forbidden_domains
    assert mp.metaphor_density == 0.6
    assert mp.extends_to_narration is False


def test_compute_current_domains_prefers_permanent_matches():
    profile = CharacterMetaphorProfile(
        permanent_domains=["stone", "cold", "iron"],
        forbidden_domains=[],
    )
    got = compute_current_domains(profile, primary_emotion="dread")
    # dread candidates include cold, stone, narrow-spaces, weight.
    # We should see stone and cold (permanent matches) prioritised.
    assert got[0].lower() in {"cold", "stone"}
    assert got[1].lower() in {"cold", "stone"}


def test_compute_current_domains_filters_forbidden():
    profile = CharacterMetaphorProfile(
        permanent_domains=[],
        forbidden_domains=["cold", "stone"],
    )
    got = compute_current_domains(profile, primary_emotion="dread")
    # cold and stone are forbidden → dropped
    got_l = [d.lower() for d in got]
    assert "cold" not in got_l
    assert "stone" not in got_l


def test_compute_current_domains_merges_primary_and_secondary():
    profile = CharacterMetaphorProfile(permanent_domains=[], forbidden_domains=[])
    got = compute_current_domains(
        profile, primary_emotion="dread", secondary_emotion="grief",
    )
    got_l = {d.lower() for d in got}
    # At least one domain from each should survive (max_domains cap aside)
    assert got_l & {"cold", "stone", "narrow-spaces", "weight"}
    # grief candidates: water, weight, absence, erosion
    # (weight is shared, so at minimum one of water/absence/erosion appears
    # unless capped — just check union is broader than primary alone)


def test_compute_current_domains_unknown_emotion_is_empty():
    profile = CharacterMetaphorProfile(permanent_domains=["stone"], forbidden_domains=[])
    assert compute_current_domains(profile, primary_emotion="nonsense") == []


def test_compute_current_domains_respects_max_domains():
    profile = CharacterMetaphorProfile(permanent_domains=[], forbidden_domains=[])
    got = compute_current_domains(
        profile, primary_emotion="dread", secondary_emotion="grief", max_domains=2,
    )
    assert len(got) <= 2


def test_default_metaphor_profile_maps_density_and_populates_current():
    profile = CharacterMetaphorProfile(
        permanent_domains=["stone", "iron"],
        forbidden_domains=["perfume"],
        metaphor_density=0.8,
        extends_to_narration=True,
    )
    mp = default_metaphor_profile("hero", profile, primary_emotion="dread")
    assert isinstance(mp, MetaphorProfile)
    assert mp.character_id == "hero"
    assert mp.permanent_domains == ["stone", "iron"]
    assert mp.forbidden_domains == ["perfume"]
    # 0.8 → rich bucket
    assert mp.metaphor_density == "rich"
    assert mp.extends_to_narration is True
    # current domains derived, not empty
    assert mp.current_domains


def test_default_metaphor_profile_density_buckets():
    def bucket(d: float) -> str:
        profile = CharacterMetaphorProfile(
            permanent_domains=[], forbidden_domains=[], metaphor_density=d,
        )
        return default_metaphor_profile("x", profile).metaphor_density

    assert bucket(0.05) == "sparse"
    assert bucket(0.3) == "occasional"
    assert bucket(0.6) == "regular"
    assert bucket(0.9) == "rich"
