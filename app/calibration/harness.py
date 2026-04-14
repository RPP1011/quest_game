"""Harness: walk the manifest, score each passage, compare to expected, roll up."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .heuristics import run_heuristics
from .judges import BatchJudge, dims_for
from .loader import Manifest, Passage, Work, load_manifest, verify_passage, PENDING_SHA
from .scorer import AggregateStats, aggregate


@dataclass
class PassageScore:
    work_id: str
    passage_id: str
    dimensions: dict[str, float]  # dim -> model score
    expected: dict[str, float]    # dim -> expected score (from work-level expected)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class DimensionStats:
    dimension: str
    n: int
    mae: float
    rmse: float
    pearson: float


@dataclass
class Report:
    passages: list[PassageScore]
    per_dim: list[DimensionStats]
    overall: AggregateStats

    def failing_correlation(self, threshold: float = 0.7) -> list[DimensionStats]:
        return [s for s in self.per_dim if s.pearson < threshold]


class Harness:
    def __init__(
        self,
        manifest: Manifest | str | Path,
        passages_dir: str | Path,
        *,
        client: Any | None = None,
        prompts_dir: str | Path | None = None,
        judge: BatchJudge | None = None,
        player_actions: dict[tuple[str, str], str] | None = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.manifest = load_manifest(manifest)
        else:
            self.manifest = manifest
        self.passages_dir = Path(passages_dir)
        self.client = client
        if judge is None:
            pdir = Path(prompts_dir) if prompts_dir else _default_prompts_dir()
            judge = BatchJudge(pdir)
        self.judge = judge
        self.player_actions = player_actions or {}

    def _passage_path(self, work: Work, passage: Passage) -> Path:
        return self.passages_dir / work.id / f"{passage.id}.txt"

    async def score_passage(self, work: Work, passage: Passage) -> PassageScore:
        ps = PassageScore(
            work_id=work.id,
            passage_id=passage.id,
            dimensions={},
            expected=dict(work.expected),
        )
        path = self._passage_path(work, passage)
        if not path.is_file():
            ps.skipped = True
            ps.skip_reason = f"missing file: {path}"
            return ps
        if passage.sha256 == PENDING_SHA:
            ps.skipped = True
            ps.skip_reason = "sha256 PENDING; run `calibrate init` first"
            return ps
        ok, actual = verify_passage(path, passage.sha256)
        if not ok:
            raise RuntimeError(
                f"sha256 mismatch for {work.id}/{passage.id}: "
                f"expected {passage.sha256}, got {actual}"
            )
        text = path.read_text(encoding="utf-8")

        # Heuristics.
        action = self.player_actions.get((work.id, passage.id), "")
        ps.dimensions.update(
            run_heuristics(text, is_quest=work.is_quest, player_action=action)
        )

        # LLM judges (skip if no client).
        if self.client is not None:
            try:
                judged = await self.judge.score(
                    client=self.client,
                    passage=text,
                    work_id=work.id,
                    pov=work.pov,
                    is_quest=work.is_quest,
                )
                for name, js in judged.items():
                    ps.dimensions[name] = js.score
            except Exception as exc:  # noqa: BLE001
                ps.errors.append(f"judge failure: {type(exc).__name__}: {exc}")

        return ps

    async def run(self) -> Report:
        passage_scores: list[PassageScore] = []
        for work in self.manifest.works:
            for p in work.passages:
                passage_scores.append(await self.score_passage(work, p))
        return self._build_report(passage_scores)

    def _build_report(self, passage_scores: list[PassageScore]) -> Report:
        # Aggregate passage-level scores to work-level before correlation.
        # The manifest's expected scores are per-work, so per-passage comparisons
        # introduce within-work noise. Mean the passages for each (work, dim).
        work_dim_scores: dict[tuple[str, str], list[float]] = {}
        work_dim_expected: dict[tuple[str, str], float] = {}
        for ps in passage_scores:
            if ps.skipped:
                continue
            for dim, model_score in ps.dimensions.items():
                if dim not in ps.expected:
                    continue
                work_dim_scores.setdefault((ps.work_id, dim), []).append(model_score)
                work_dim_expected[(ps.work_id, dim)] = ps.expected[dim]

        per_dim_pairs: dict[str, list[tuple[float, float]]] = {}
        all_pairs: list[tuple[float, float]] = []
        for (work_id, dim), scores in work_dim_scores.items():
            mean_score = sum(scores) / len(scores)
            pair = (mean_score, work_dim_expected[(work_id, dim)])
            per_dim_pairs.setdefault(dim, []).append(pair)
            all_pairs.append(pair)

        per_dim = [
            DimensionStats(
                dimension=dim,
                n=len(pairs),
                mae=aggregate(pairs).mae,
                rmse=aggregate(pairs).rmse,
                pearson=aggregate(pairs).pearson,
            )
            for dim, pairs in sorted(per_dim_pairs.items())
        ]
        return Report(
            passages=passage_scores,
            per_dim=per_dim,
            overall=aggregate(all_pairs),
        )


def _default_prompts_dir() -> Path:
    # app/calibration/harness.py -> project root / prompts
    return Path(__file__).resolve().parent.parent.parent / "prompts"
