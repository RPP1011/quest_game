from __future__ import annotations
from dataclasses import dataclass, field
from .delta import StateDelta


@dataclass
class RetconSpec:
    target_update: int  # the update to patch as-if it happened at
    delta: StateDelta   # changes to apply
    reason: str         # human-readable rationale


@dataclass
class RetconResult:
    applied_update: int
    new_update_number: int        # the synthetic update created for the retcon
    affected_narrative: list[int] = field(default_factory=list)  # update_numbers of stale records
