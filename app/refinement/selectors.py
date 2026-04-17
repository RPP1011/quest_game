"""Refinement target selectors (Phase 5 of story-rollout).

Three pluggable selectors that read KB / rollout / skeleton state and
return ``RefinementTarget`` lists. Each strategy targets a different
diagnosed weakness:

- WeakChapterSelector  → chapters with low overall score
- UnpaidHookSelector   → chapters that should pay off a hook but didn't
- SiblingOutscoredSelector → chapters where another rollout's same
                              slot scored materially higher
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.world.state_manager import WorldStateManager

from .framework import RefinementTarget


# ---------------------------------------------------------------------------
# Strategy 1: weak chapter
# ---------------------------------------------------------------------------

class WeakChapterSelector:
    """Selects chapters whose mean dim score is below ``threshold``.

    Default threshold (0.55) targets chapters in the "below baseline"
    zone where a rewrite has clear room to improve. Lower threshold
    (e.g. 0.45) is more aggressive; higher (0.65) is conservative.
    """
    name = "weak_chapter"

    def __init__(self, world: WorldStateManager, *, threshold: float = 0.55) -> None:
        self._world = world
        self._threshold = threshold

    def select(
        self, *, quest_id: str, rollout_id: str | None = None,
        max_targets: int = 3,
    ) -> list[RefinementTarget]:
        rollouts = self._world.list_rollouts(quest_id=quest_id)
        if rollout_id is not None:
            rollouts = [r for r in rollouts if r.id == rollout_id]
        candidates: list[tuple[float, RefinementTarget]] = []
        for r in rollouts:
            scores_rows = self._world.list_chapter_scores(r.id)
            # Group by chapter
            by_chapter: dict[int, dict[str, float]] = defaultdict(dict)
            rationales: dict[int, dict[str, str]] = defaultdict(dict)
            for s in scores_rows:
                ci = s["chapter_index"]
                by_chapter[ci][s["dim"]] = s["score"]
                rationales[ci][s["dim"]] = s["rationale"]
            for ci, dims in by_chapter.items():
                if not dims:
                    continue
                mean = sum(dims.values()) / len(dims)
                if mean >= self._threshold:
                    continue
                # Pick the worst dim for guidance
                worst_dim = min(dims, key=dims.get)
                worst_score = dims[worst_dim]
                worst_rationale = rationales[ci].get(worst_dim, "")
                guidance = (
                    f"This chapter scored low overall (mean {mean:.2f}). "
                    f"The weakest dimension was {worst_dim} ({worst_score:.2f}): "
                    f"{worst_rationale}. Rewrite the chapter to specifically "
                    f"address {worst_dim} while preserving the chapter's events."
                )
                candidates.append((mean, RefinementTarget(
                    rollout_id=r.id, chapter_index=ci, quest_id=quest_id,
                    strategy=self.name,
                    reason=f"mean dim {mean:.2f} < threshold {self._threshold:.2f}; weakest: {worst_dim}",
                    guidance=guidance,
                    baseline_scores=dict(dims),
                )))
        # Lowest-scoring first
        candidates.sort(key=lambda x: x[0])
        return [t for _, t in candidates[:max_targets]]


# ---------------------------------------------------------------------------
# Strategy 2: unpaid hooks
# ---------------------------------------------------------------------------

class UnpaidHookSelector:
    """Selects chapters where a skeleton-scheduled hook should have paid
    off by now but ``kb_hook_payoffs`` shows no payoff.

    Reads the picked candidate's ArcSkeleton.hook_schedule and compares
    each ``paid_off_by_chapter`` against the actual KB. For each unpaid
    hook past its deadline, target the most-recent chapter at or before
    the deadline.
    """
    name = "unpaid_hook"

    def __init__(self, world: WorldStateManager) -> None:
        self._world = world

    def _hook_descriptions(self) -> dict[str, str]:
        """Map hook_id → description from the foreshadowing table."""
        try:
            rows = self._world._conn.execute(
                "SELECT id, description FROM foreshadowing"
            ).fetchall()
            return {r["id"]: r["description"] or "" for r in rows}
        except Exception:
            return {}

    def select(
        self, *, quest_id: str, rollout_id: str | None = None,
        max_targets: int = 3,
    ) -> list[RefinementTarget]:
        rollouts = self._world.list_rollouts(quest_id=quest_id)
        if rollout_id is not None:
            rollouts = [r for r in rollouts if r.id == rollout_id]
        descriptions = self._hook_descriptions()

        targets: list[RefinementTarget] = []
        for r in rollouts:
            # Need the skeleton to know when each hook should pay off
            try:
                skel = self._world.get_skeleton_for_candidate(r.candidate_id)
            except Exception:
                skel = None
            if skel is None:
                continue
            # Build {hook_id: paid_off_by_chapter} from skeleton
            scheduled = {h.hook_id: h.paid_off_by_chapter for h in skel.hook_schedule}
            # Actual payoffs for this rollout
            actual = {
                row["hook_id"]: row["paid_off_at_chapter"]
                for row in self._world.list_hook_payoffs(quest_id)
                if row["rollout_id"] == r.id
            }
            # Number of completed chapters in this rollout
            chapters = self._world.list_rollout_chapters(r.id)
            max_committed = max([c.chapter_index for c in chapters], default=0)

            for hook_id, deadline in scheduled.items():
                if max_committed < deadline:
                    continue  # not yet past deadline
                if actual.get(hook_id) is not None:
                    continue  # paid off — fine
                # Unpaid past deadline. Target the deadline chapter.
                target_ch = min(deadline, max_committed)
                # Pull baseline scores for this chapter
                score_rows = self._world.list_chapter_scores(r.id, chapter_index=target_ch)
                baseline = {s["dim"]: s["score"] for s in score_rows}
                desc = descriptions.get(hook_id, "(no description)")
                guidance = (
                    f"This chapter was scheduled to pay off the foreshadowing "
                    f"hook {hook_id!r} (\"{desc}\") by chapter {deadline}, but "
                    f"it didn't land. Rewrite the chapter to deliver that "
                    f"payoff explicitly — name the planted thing, resolve the "
                    f"setup that's been dangling. Preserve the existing scene "
                    f"shape; insert the payoff beat where it has the most weight."
                )
                targets.append(RefinementTarget(
                    rollout_id=r.id, chapter_index=target_ch, quest_id=quest_id,
                    strategy=self.name,
                    reason=f"hook {hook_id!r} scheduled for ch {deadline}, unpaid",
                    guidance=guidance,
                    baseline_scores=baseline,
                ))

        return targets[:max_targets]


# ---------------------------------------------------------------------------
# Strategy 3-lite: sibling outscored
# ---------------------------------------------------------------------------

class SiblingOutscoredSelector:
    """Selects chapters where another rollout's same chapter_index scored
    materially higher (delta ≥ ``min_delta``) on at least one dim.

    Lite version: passes the higher-scoring sibling's prose to the
    regenerator as a "reference" for the LLM to crib ideas from. Doesn't
    do structural beat splicing.
    """
    name = "sibling_outscored"

    def __init__(
        self, world: WorldStateManager, *, min_delta: float = 0.15,
    ) -> None:
        self._world = world
        self._min_delta = min_delta

    def select(
        self, *, quest_id: str, rollout_id: str | None = None,
        max_targets: int = 3,
    ) -> list[RefinementTarget]:
        rollouts = self._world.list_rollouts(quest_id=quest_id)
        if not rollouts or len(rollouts) < 2:
            return []  # need siblings to compare against
        target_rollouts = (
            [r for r in rollouts if r.id == rollout_id] if rollout_id
            else rollouts
        )

        # Build {(rollout_id, chapter_index, dim): score} for all rollouts
        all_scores: dict[tuple[str, int, str], float] = {}
        all_chapters: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
        for r in rollouts:
            for s in self._world.list_chapter_scores(r.id):
                key = (r.id, s["chapter_index"], s["dim"])
                all_scores[key] = s["score"]
            for c in self._world.list_rollout_chapters(r.id):
                all_chapters[(r.id, c.chapter_index)] = {
                    "prose": c.prose, "player_action": c.player_action,
                }

        targets: list[RefinementTarget] = []
        for r in target_rollouts:
            for s in self._world.list_chapter_scores(r.id):
                ci, dim, my_score = s["chapter_index"], s["dim"], s["score"]
                # Find any sibling chapter at the same index that beats us by ≥ min_delta
                best_sibling: tuple[str, float] | None = None
                for other in rollouts:
                    if other.id == r.id:
                        continue
                    other_score = all_scores.get((other.id, ci, dim))
                    if other_score is None:
                        continue
                    delta = other_score - my_score
                    if delta >= self._min_delta and (
                        best_sibling is None or delta > best_sibling[1]
                    ):
                        best_sibling = (other.id, delta)
                if best_sibling is None:
                    continue
                sibling_id, delta = best_sibling
                sibling_prose = all_chapters.get((sibling_id, ci), {}).get("prose", "")
                if not sibling_prose:
                    continue
                # Pull baseline scores for this chapter
                baseline_rows = self._world.list_chapter_scores(r.id, chapter_index=ci)
                baseline = {row["dim"]: row["score"] for row in baseline_rows}
                # Trim sibling prose for prompt budget
                sibling_excerpt = sibling_prose[:1500]
                if len(sibling_prose) > 1500:
                    sibling_excerpt += "\n\n[...sibling prose continues — truncated]"
                guidance = (
                    f"A sibling rollout's chapter at this same index scored "
                    f"{delta:+.2f} higher on {dim}. Look at how the sibling "
                    f"version handled this scene — the technique that worked "
                    f"there for {dim}. Adapt that approach to this chapter "
                    f"without copying prose. Sibling chapter excerpt:\n\n"
                    f"<<<\n{sibling_excerpt}\n>>>"
                )
                targets.append(RefinementTarget(
                    rollout_id=r.id, chapter_index=ci, quest_id=quest_id,
                    strategy=self.name,
                    reason=f"sibling rollout {sibling_id} scored +{delta:.2f} on {dim} for ch{ci}",
                    guidance=guidance,
                    baseline_scores=baseline,
                ))
                break  # one target per (rollout, chapter)

        # Dedupe (rollout_id, chapter_index)
        seen: set[tuple[str, int]] = set()
        out: list[RefinementTarget] = []
        for t in targets:
            key = (t.rollout_id, t.chapter_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out[:max_targets]
