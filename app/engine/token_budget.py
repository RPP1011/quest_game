from __future__ import annotations
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    total: int = 200_000
    system_prompt: int = 15_000
    world_state: int = 30_000
    narrative_history: int = 40_000
    style_config: int = 10_000
    prior_stage_outputs: int = 15_000
    generation_headroom: int = 20_000
    safety_margin: int = 10_000

    def remaining(self, used: dict[str, int]) -> int:
        return self.total - sum(used.values()) - self.safety_margin

    def fits(self, used: dict[str, int]) -> bool:
        return self.remaining(used) >= 0
