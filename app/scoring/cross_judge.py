"""Cross-judge scoring — dual-model evaluation for self-preference detection.

Runs M-Prometheus-14B (on CPU, port 8083) alongside Gemma 4 (GPU, port 8082)
to score each chapter. Disagreement > 1.5 pts on any dim flags self-preference.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.runtime.client import InferenceClient
from app.rollout.scorer import score_chapter_independent, COLLAPSED_DIMS


def compute_agreement(a: dict[str, float], b: dict[str, float]) -> float:
    """Mean absolute difference across shared dims."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    return sum(abs(a[d] - b[d]) for d in shared) / len(shared)


@dataclass
class JudgePair:
    gemma_scores: dict[str, float]
    prometheus_scores: dict[str, float]

    @property
    def agreement(self) -> float:
        return compute_agreement(self.gemma_scores, self.prometheus_scores)

    @property
    def self_preference_flag(self) -> bool:
        for dim in set(self.gemma_scores) & set(self.prometheus_scores):
            if abs(self.gemma_scores[dim] - self.prometheus_scores[dim]) > 0.15:
                return True
        return False


async def score_with_cross_judge(
    *, gemma_client: InferenceClient,
    prometheus_client: InferenceClient,
    chapter_text: str,
    dims: list[str] | None = None,
) -> JudgePair:
    """Score a chapter with both judges in parallel."""
    import asyncio
    use_dims = dims or list(COLLAPSED_DIMS)

    gemma_task = score_chapter_independent(
        client=gemma_client, chapter_text=chapter_text, dims=use_dims,
    )
    prometheus_task = score_chapter_independent(
        client=prometheus_client, chapter_text=chapter_text, dims=use_dims,
    )

    gemma_raw, prometheus_raw = await asyncio.gather(gemma_task, prometheus_task)

    return JudgePair(
        gemma_scores={d: v["score"] for d, v in gemma_raw.items()},
        prometheus_scores={d: v["score"] for d, v in prometheus_raw.items()},
    )


def persist_judge_pair(
    conn, rollout_id: str, chapter_index: int, pair: JudgePair,
) -> None:
    """Save both judges' scores to cross_judge_scores table."""
    for model, scores in [("gemma", pair.gemma_scores), ("prometheus", pair.prometheus_scores)]:
        for dim, score in scores.items():
            conn.execute(
                "INSERT OR REPLACE INTO cross_judge_scores "
                "(rollout_id, chapter_index, judge_model, dim, score) "
                "VALUES (?, ?, ?, ?, ?)",
                (rollout_id, chapter_index, model, dim, score),
            )
    conn.commit()
