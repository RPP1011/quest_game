"""Validate data/calibration/scenes_manifest.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml


SCENES_MANIFEST = Path("data/calibration/scenes_manifest.yaml")

ARC_DIMS = {
    "tension_execution",
    "choice_hook_quality",
    "update_self_containment",
    "choice_meaningfulness",
    "world_state_legibility",
}
QUEST_ONLY = {
    "choice_hook_quality",
    "update_self_containment",
    "choice_meaningfulness",
    "world_state_legibility",
}
MIN_WORKS = 12


def test_scenes_manifest_loads():
    data = yaml.safe_load(SCENES_MANIFEST.read_text(encoding="utf-8"))
    assert data["version"] == 1
    works = data["works"]
    assert len(works) >= MIN_WORKS
    ids = [w["id"] for w in works]
    assert len(set(ids)) == len(ids), "duplicate work ids"


def test_every_entry_is_valid():
    data = yaml.safe_load(SCENES_MANIFEST.read_text(encoding="utf-8"))
    for w in data["works"]:
        assert {"id", "title", "author", "pov", "is_quest", "expected", "passages"} <= w.keys()
        # Every expected dim must be a valid arc dim.
        for dim, val in w["expected"].items():
            assert dim in ARC_DIMS, f"{w['id']}: unknown arc dim {dim}"
            assert 0.0 <= float(val) <= 1.0
            # Quest-only dims only allowed when is_quest.
            if dim in QUEST_ONLY:
                assert w["is_quest"], f"{w['id']}: quest-only dim on non-quest"
        # Five passage slots s01..s05.
        assert len(w["passages"]) == 5
        for i, p in enumerate(w["passages"], 1):
            assert p["id"] == f"s{i:02d}"
            assert "sha256" in p
            for label in ("expected_high", "expected_low"):
                for dim in p.get(label, []) or []:
                    assert dim in ARC_DIMS
                    if dim in QUEST_ONLY:
                        assert w["is_quest"], (
                            f"{w['id']}/{p['id']}: quest-only dim on novel"
                        )
