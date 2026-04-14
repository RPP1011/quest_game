from pathlib import Path

from app.calibration import load_manifest


MANIFEST = Path("data/calibration/manifest.yaml")

EXPECTED_WORK_IDS = {
    "mrs_dalloway", "sun_also_rises", "madame_bovary", "ulysses",
    "blood_meridian", "pride_and_prejudice", "brothers_karamazov",
    "left_hand_of_darkness", "get_shorty", "remains_of_the_day",
    "blade_itself", "way_of_kings",
    "marked_for_death", "forge_of_destiny", "practical_guide_evil",
    "pale_lights",
    "my_man_jeeves", "three_men_in_a_boat",
}

QUEST_WORKS = {"marked_for_death", "forge_of_destiny",
               "practical_guide_evil", "pale_lights"}

# Works added with the expanded 10-slot passage layout. All earlier works
# use the original 2-3 slot layout.
LARGE_SLOT_WORKS = {"my_man_jeeves", "three_men_in_a_boat"}


def test_manifest_loads_all_sixteen_works():
    m = load_manifest(MANIFEST)
    ids = {w.id for w in m.works}
    assert ids == EXPECTED_WORK_IDS
    assert len(m.works) == len(EXPECTED_WORK_IDS)


def test_quest_flag_set_correctly():
    m = load_manifest(MANIFEST)
    for w in m.works:
        assert w.is_quest == (w.id in QUEST_WORKS), w.id


def test_every_work_has_passage_slots():
    m = load_manifest(MANIFEST)
    for w in m.works:
        if w.id in LARGE_SLOT_WORKS:
            assert len(w.passages) == 10, f"{w.id} has {len(w.passages)} passages"
        else:
            assert 2 <= len(w.passages) <= 3, f"{w.id} has {len(w.passages)} passages"
        for p in w.passages:
            assert p.sha256 == "PENDING" or (
                len(p.sha256) == 64 and all(c in "0123456789abcdef" for c in p.sha256)
            ), f"{w.id}/{p.id} has invalid sha256: {p.sha256!r}"
            assert p.expected_high
            assert p.expected_low


def test_expected_scores_in_unit_interval():
    m = load_manifest(MANIFEST)
    for w in m.works:
        for dim, score in w.expected.items():
            assert 0.0 <= score <= 1.0, f"{w.id}/{dim}={score}"


def test_quest_only_dims_absent_from_novels():
    m = load_manifest(MANIFEST)
    quest_only = {"choice_hook_quality", "update_self_containment",
                  "choice_meaningfulness", "world_state_legibility",
                  "action_fidelity"}
    for w in m.works:
        if w.is_quest:
            continue
        collisions = quest_only & set(w.expected.keys())
        assert not collisions, f"{w.id} has quest-only dims: {collisions}"
