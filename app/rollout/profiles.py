"""Virtual-player profiles for rollouts.

A profile is a small YAML file describing how a synthetic player chooses
among suggested_choices at each tick. The rollout harness loads the
profile once per run; the action_selector consults the rubric on each
chapter's choices.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


PROFILES_DIR = Path(__file__).parent / "profiles"


class VirtualPlayerProfile(BaseModel):
    id: str
    description: str
    action_selection_rubric: str


def load_profile(profile_id: str) -> VirtualPlayerProfile:
    """Load a profile by id from the bundled profiles/ directory.

    Raises FileNotFoundError if the profile doesn't exist.
    """
    path = PROFILES_DIR / f"{profile_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no profile {profile_id!r} at {path}")
    data = yaml.safe_load(path.read_text())
    return VirtualPlayerProfile(**data)


def list_profiles() -> list[VirtualPlayerProfile]:
    """Return every profile bundled in the profiles/ directory."""
    out: list[VirtualPlayerProfile] = []
    for p in sorted(PROFILES_DIR.glob("*.yaml")):
        data = yaml.safe_load(p.read_text())
        out.append(VirtualPlayerProfile(**data))
    return out
