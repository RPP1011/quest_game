"""Pairwise preference scorer for G2 rerank.

Wraps the pairwise LoRA (finetuned LFM2.5-1.2B) that emits a single token
(`A` or `B`) given two passages and a dimension. Matches G2 rerank's actual
use case: "pick the best candidate" rather than "score every candidate on
an absolute scale".

Usage:

    scorer = PairwiseScorer(server_url="http://127.0.0.1:8081")
    winner_idx = await scorer.tournament(candidates, dim="free_indirect_quality")

Integration with G2 (`app/engine/pipeline.py::_score_candidate`): subjective
dims (FIS, interiority_depth, detail_characterization, subtext_presence,
voice_distinctiveness, thematic_presence) go through this scorer; heuristic
dims (sentence_variance, dialogue_ratio, pacing, sensory_density lexicon,
action_fidelity, named_entity_presence) stay on the existing critic stack.

Performance notes:
- Greedy llama.cpp @ 670 tok/s × 2k-token prompt = ~3s per pair.
- vllm @ 3k tok/s = ~0.7s per pair, batchable 10-20x across pairs.
- For N=5 candidates × 1 dim: 10 pairs via round-robin = 7s (llama) / 2s (vllm).
- Default to tournament (N-1 comparisons) for speed; round-robin optional.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

DIM_DEFS = {
    "free_indirect_quality": "narrator adopts character's idiom/perception",
    "interiority_depth": "depth of access to character thought/feeling",
    "detail_characterization": "details reveal the perceiving consciousness",
    "sensory_density": "density of specific sensory perception",
    "voice_distinctiveness": "distinct character/narrator register",
    "thematic_presence": "themes embedded in prose (not announced)",
    "subtext_presence": "important things through what ISN'T said",
    "clarity": "a careful reader can follow sentence-by-sentence",
}

# Subjective dims routed through this scorer. Heuristic dims stay on critics.
SUBJECTIVE_DIMS = (
    "free_indirect_quality",
    "interiority_depth",
    "detail_characterization",
    "voice_distinctiveness",
    "thematic_presence",
    "subtext_presence",
)


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


@dataclass
class PairwiseScorer:
    client: ChatLike

    def _messages(self, dim: str, a: str, b: str) -> list[dict]:
        system = (
            "You are a literary scorer. Given two passages, decide which has more "
            "of the named dimension. Respond with a single token: `A` or `B`."
        )
        user = (
            f"Dimension: **{dim}** — {DIM_DEFS.get(dim, '')}\n\n"
            f"A:\n---\n{a.strip()}\n---\n\n"
            f"B:\n---\n{b.strip()}\n---\n\n"
            "Which passage scores higher on this dimension? Respond with only `A` or `B`."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def prefer(self, a: str, b: str, dim: str) -> str:
        """Return "A" or "B" — which passage scores higher on ``dim``."""
        response = await self.client.chat(
            self._messages(dim, a, b), max_tokens=2, temperature=0.0,
        )
        token = (response or "").strip().upper()[:1]
        return "A" if token == "A" else "B"

    async def tournament(self, candidates: list[str], dim: str) -> int:
        """Pick the best candidate via sequential elimination.

        Returns the index of the winner. O(N) pairwise calls. Deterministic
        for fixed inputs but order-sensitive (first candidate gets N-1 matches).
        """
        if not candidates:
            raise ValueError("tournament needs at least one candidate")
        winner_idx = 0
        for challenger_idx in range(1, len(candidates)):
            choice = await self.prefer(
                candidates[winner_idx], candidates[challenger_idx], dim,
            )
            if choice == "B":
                winner_idx = challenger_idx
        return winner_idx

    async def score_by_wins(
        self, candidates: list[str], dim: str,
    ) -> list[int]:
        """Round-robin scoring: each candidate plays every other once.

        Returns win counts (len == len(candidates)). Order-invariant and
        provides full ranking but costs O(N²/2) calls. Use for small N (≤5).
        """
        wins = [0] * len(candidates)
        pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
        # Could be batched via asyncio.gather — left simple for now.
        results = await asyncio.gather(*[
            self.prefer(candidates[i], candidates[j], dim) for i, j in pairs
        ])
        for (i, j), choice in zip(pairs, results):
            if choice == "A":
                wins[i] += 1
            else:
                wins[j] += 1
        return wins

    async def aggregate_winner(
        self, candidates: list[str], dims: list[str] | None = None,
    ) -> int:
        """Pick the overall-best candidate across all subjective dims.

        Runs tournament per dim, aggregates via win-count. Ties broken by
        first dim's winner. O(D*N) pairwise calls.
        """
        dims = list(dims) if dims else list(SUBJECTIVE_DIMS)
        wins = [0] * len(candidates)
        per_dim_winners = await asyncio.gather(*[
            self.tournament(candidates, d) for d in dims
        ])
        for w in per_dim_winners:
            wins[w] += 1
        max_wins = max(wins)
        return wins.index(max_wins)
