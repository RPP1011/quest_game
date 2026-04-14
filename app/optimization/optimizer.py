"""Prompt optimizer: scan scorecards, propose targeted mutations, A/B test.

The optimizer is a thin orchestration layer. All the heavy lifting —
scoring, persistence, inference — lives in the existing Day 2 / Day 6
substrate:

- ``WorldStateManager`` owns ``scorecards`` + ``dimension_scores``, so
  :meth:`PromptOptimizer.identify_weak_dimensions` is a pure read.
- Mutation proposal calls an injected ``mutation_proposer`` callable
  (signature: ``(dim, current_prompt, bad_examples) -> (new_prompt, rationale)``)
  so tests can stub it without an LLM call; production wires it to the
  existing ``InferenceClient`` wrapped by :func:`claude_mutation_proposer`.
- Replay reuses the caller's ``pipeline_factory`` — we never reconstruct
  a pipeline here. The factory receives a resolved ``ReplayContext`` and
  returns whatever object it wants (typically the committed prose string,
  though the structure is opaque to the optimizer).

What replay is **not**
----------------------

Replay reconstruction is lossy by design (see
:class:`ReplayContext` for the exact fields). We cannot perfectly
reproduce a historical run because:

- The world snapshot at the time of the original update has drifted
  (subsequent updates mutated entities, added narrative, shifted
  emotional arcs). Replay uses the current ``WorldStateManager`` state,
  not a point-in-time snapshot.
- Retrieval embeddings and voice anchors that were available at run
  time may no longer exist (or may have been re-indexed).
- LLM sampling is non-deterministic; two calls with identical prompts
  rarely yield identical prose.
- Day 6 LLM-judge dims depend on Claude's own rubric runs and can
  swing ±0.1 between calls.

Callers who need a rigorous before/after should ensure
``pipeline_factory`` sets ``temperature=0.0``, fixes any seed the
backend accepts, and disables the LLM-judge dims during replay.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class WeakDim(BaseModel):
    """One dimension flagged as underperforming.

    ``mean_score`` is computed over the most recent ``sample_size`` scorecards
    for the scoped ``quest_id`` (or across all quests if unscoped).
    ``recent_examples`` are 2-3 prose snippets drawn from the scorecards with
    the lowest scores on this dimension — used both as in-context examples for
    :meth:`PromptOptimizer.propose_mutation` and for human review.
    """

    dimension: str
    mean_score: float
    sample_size: int = Field(ge=0)
    recent_examples: list[str] = Field(default_factory=list)


class Mutation(BaseModel):
    """A proposed edit to a single prompt template.

    Apply is deferred — the caller chooses whether to write ``after_text``
    over ``prompt_path`` after inspecting the A/B result.
    """

    dimension: str
    prompt_path: str
    before_text: str
    after_text: str
    rationale: str = ""


class ABResult(BaseModel):
    """Structured outcome of :meth:`PromptOptimizer.replay_ab`."""

    mutation: Mutation
    n_replays: int = Field(ge=0)
    mean_before: float = Field(ge=0.0, le=1.0)
    mean_after: float = Field(ge=0.0, le=1.0)
    accept_recommended: bool = False
    notes: str = ""


@dataclass
class ReplayContext:
    """Minimal context reconstructed for a historical replay.

    Populated by :meth:`PromptOptimizer._build_replay_context`; passed to
    ``pipeline_factory(ctx, mutated_prompt)``. Fields are best-effort —
    ``None`` means "could not recover from storage"; the factory decides
    whether to skip the replay.
    """

    trace_id: str
    player_action: str | None
    update_number: int | None
    narrative_text: str | None
    baseline_dim_score: float | None


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


MutationProposer = Callable[
    [str, str, Sequence[str]],  # (dimension, current_prompt, bad_examples)
    tuple[str, str],             # (new_prompt, rationale)
]
"""Injectable prompt-mutator. Must be synchronous, deterministic given inputs."""


PipelineFactory = Callable[
    [ReplayContext, str],  # (context, mutated_prompt_text)
    dict[str, float],      # returns {dimension: new_score, ...}
]
"""Replays one trace with the mutated prompt. Returns the new per-dim scores.

The factory is opaque to the optimizer — typical implementations spin up a
Pipeline with the mutated template pre-rendered, run the affected stage on
the reconstructed context, score the output with :class:`app.scoring.Scorer`,
and return the resulting dim map. Tests stub this with a deterministic
function (see ``tests/optimization/test_optimizer.py``).
"""


class PromptOptimizer:
    """Scan scorecards, propose prompt mutations, run replay A/B.

    Construction is cheap (no I/O). Reuse one instance across calls.
    """

    def __init__(
        self,
        world: Any,
        *,
        mutation_proposer: MutationProposer | None = None,
        prompts_root: str | Path | None = None,
        trace_store: Any | None = None,
    ) -> None:
        self._world = world
        self._mutation_proposer = mutation_proposer
        self._prompts_root = (
            Path(prompts_root) if prompts_root is not None
            else _default_prompts_dir()
        )
        self._trace_store = trace_store

    # ---- 1) identify weak dimensions ----

    def identify_weak_dimensions(
        self,
        quest_id: str | None = None,
        *,
        threshold: float = 0.5,
        n: int = 3,
        recent_limit: int = 50,
    ) -> list[WeakDim]:
        """Return up to ``n`` underperforming dimensions.

        Strategy
        --------
        1. Pull the most recent ``recent_limit`` scorecards for
           ``quest_id`` (all quests if ``None``).
        2. For each dimension that appears in any scorecard, compute the
           arithmetic mean of its scores.
        3. Sort dimensions ascending by mean. Return the first ``n`` whose
           mean is strictly less than ``threshold``. If fewer than ``n``
           fall below ``threshold``, pad with the lowest remaining dims up
           to ``n`` so callers always get something actionable.
        4. Attach 2-3 lowest-scoring prose snippets per dim via
           ``narrative.raw_text`` on the matching ``pipeline_trace_id``.

        A quest with no scorecards returns ``[]``.
        """
        rows = self._load_dim_rows(quest_id, recent_limit)
        if not rows:
            return []

        # dim -> list[(score, scorecard_id, trace_id)]
        by_dim: dict[str, list[tuple[float, int, str | None]]] = {}
        for dim, score, sid, tid in rows:
            by_dim.setdefault(dim, []).append((score, sid, tid))

        aggregated = [
            (
                dim,
                statistics.fmean(s for s, _, _ in entries),
                entries,
            )
            for dim, entries in by_dim.items()
        ]
        aggregated.sort(key=lambda t: (t[1], t[0]))  # mean asc, then dim name

        below = [t for t in aggregated if t[1] < threshold]
        if len(below) >= n:
            chosen = below[:n]
        else:
            # Pad with the next-lowest dims so the optimizer always has
            # something to chew on even if the quest is generally healthy.
            remainder = [t for t in aggregated if t not in below]
            chosen = below + remainder[: max(0, n - len(below))]

        out: list[WeakDim] = []
        for dim, mean, entries in chosen:
            entries.sort(key=lambda row: row[0])  # worst first
            snippets = self._collect_snippets(entries[:3])
            out.append(WeakDim(
                dimension=dim,
                mean_score=float(mean),
                sample_size=len(entries),
                recent_examples=snippets,
            ))
        return out

    # ---- 2) propose a mutation ----

    def propose_mutation(
        self,
        weak_dim: WeakDim,
        prompt_path: str | Path,
        *,
        mutation_proposer: MutationProposer | None = None,
    ) -> Mutation:
        """Read ``prompt_path``, ask the proposer for ONE targeted edit.

        The proposer receives the dim name, the current prompt body, and
        2-3 low-scoring example snippets. It returns ``(new_prompt,
        rationale)`` — both strings. The optimizer does not apply the
        edit; the returned :class:`Mutation` carries before/after for the
        caller to inspect.

        ``mutation_proposer`` falls back to the constructor-level default.
        If neither is set, raises ``RuntimeError`` so failures are loud.
        """
        proposer = mutation_proposer or self._mutation_proposer
        if proposer is None:
            raise RuntimeError(
                "propose_mutation(): no mutation_proposer wired "
                "(pass one to the constructor or this call)"
            )

        path = self._resolve_prompt_path(prompt_path)
        before_text = path.read_text()
        new_text, rationale = proposer(
            weak_dim.dimension,
            before_text,
            tuple(weak_dim.recent_examples),
        )
        return Mutation(
            dimension=weak_dim.dimension,
            prompt_path=str(path),
            before_text=before_text,
            after_text=new_text,
            rationale=rationale,
        )

    # ---- 3) A/B replay ----

    def replay_ab(
        self,
        mutation: Mutation,
        trace_ids: Sequence[str],
        pipeline_factory: PipelineFactory,
        *,
        accept_delta: float = 0.02,
    ) -> ABResult:
        """Replay each historical trace with the mutated prompt and compare.

        For each trace id, build a :class:`ReplayContext` from the
        narrative + scorecard store, hand it plus ``mutation.after_text``
        to ``pipeline_factory``, collect the returned dim scores, and
        record the mutation's dim score. Compare mean before / after; if
        ``mean_after - mean_before > accept_delta``, recommend accepting.

        Traces that cannot be reconstructed (missing scorecard, missing
        narrative, or factory raising) are skipped — the ``n_replays`` in
        the returned :class:`ABResult` reflects only successful replays.
        A zero-replay run returns ``accept_recommended=False`` with a
        ``notes`` field explaining why.
        """
        before_scores: list[float] = []
        after_scores: list[float] = []
        skip_reasons: list[str] = []

        for tid in trace_ids:
            ctx = self._build_replay_context(tid, mutation.dimension)
            if ctx is None or ctx.baseline_dim_score is None:
                skip_reasons.append(f"{tid}: no baseline scorecard")
                continue
            try:
                new_dims = pipeline_factory(ctx, mutation.after_text)
            except Exception as e:  # pragma: no cover - factory fault
                skip_reasons.append(f"{tid}: factory raised {type(e).__name__}")
                continue
            if not isinstance(new_dims, dict):
                skip_reasons.append(f"{tid}: factory returned non-dict")
                continue
            new_score = new_dims.get(mutation.dimension)
            if new_score is None:
                skip_reasons.append(f"{tid}: dim missing from factory output")
                continue
            before_scores.append(float(ctx.baseline_dim_score))
            after_scores.append(float(new_score))

        n = len(before_scores)
        mean_before = statistics.fmean(before_scores) if before_scores else 0.0
        mean_after = statistics.fmean(after_scores) if after_scores else 0.0
        accept = n > 0 and (mean_after - mean_before) > accept_delta

        notes_parts: list[str] = []
        if n == 0:
            notes_parts.append("no successful replays")
        if skip_reasons:
            notes_parts.append(f"{len(skip_reasons)} skipped")
        return ABResult(
            mutation=mutation,
            n_replays=n,
            mean_before=_clip01(mean_before),
            mean_after=_clip01(mean_after),
            accept_recommended=accept,
            notes="; ".join(notes_parts),
        )

    # ---- 4) apply (deferred write) ----

    def apply_mutation(self, mutation: Mutation) -> Path:
        """Back up the current prompt and write the mutation's after_text.

        The backup is written to ``<path>.bak.<unix_ts>``. Returns the
        backup path. Raises ``FileNotFoundError`` if the prompt path does
        not resolve under ``prompts_root``.
        """
        path = self._resolve_prompt_path(mutation.prompt_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        backup.write_text(path.read_text())
        path.write_text(mutation.after_text)
        return backup

    # ---- internals ----

    def _load_dim_rows(
        self, quest_id: str | None, recent_limit: int,
    ) -> list[tuple[str, float, int, str | None]]:
        """Return ``[(dim, score, scorecard_id, trace_id), ...]`` newest-first.

        Pulls directly against the DB because ``list_scorecards`` returns
        validated ``Scorecard`` objects that don't expose partial / Day 6
        dims. This path tolerates both the 12-dim Day 2 set and the
        15-dim Day 6 set.
        """
        conn = getattr(self._world, "_conn", None)
        if conn is None:
            return []
        if quest_id is None:
            sql = (
                "SELECT ds.dimension, ds.score, s.id AS scorecard_id, "
                "s.pipeline_trace_id "
                "FROM scorecards s JOIN dimension_scores ds "
                "ON ds.scorecard_id = s.id "
                "ORDER BY s.id DESC, ds.dimension ASC"
            )
            params: tuple[Any, ...] = ()
        else:
            sql = (
                "SELECT ds.dimension, ds.score, s.id AS scorecard_id, "
                "s.pipeline_trace_id "
                "FROM scorecards s JOIN dimension_scores ds "
                "ON ds.scorecard_id = s.id "
                "WHERE s.quest_id = ? "
                "ORDER BY s.id DESC, ds.dimension ASC"
            )
            params = (quest_id,)

        rows = conn.execute(sql, params).fetchall()
        # Clip to the most recent N scorecards, not the most recent N rows.
        seen_cards: set[int] = set()
        out: list[tuple[str, float, int, str | None]] = []
        for r in rows:
            sid = int(r["scorecard_id"])
            if sid not in seen_cards:
                if len(seen_cards) >= recent_limit:
                    continue
                seen_cards.add(sid)
            if sid not in seen_cards:
                continue
            out.append((
                str(r["dimension"]),
                float(r["score"]),
                sid,
                r["pipeline_trace_id"],
            ))
        return out

    def _collect_snippets(
        self,
        entries: Iterable[tuple[float, int, str | None]],
    ) -> list[str]:
        """Look up narrative ``raw_text`` for each (scorecard, trace) pair.

        Returns up to 3 short (≤400-char) prose previews. Missing narratives
        silently drop out.
        """
        conn = getattr(self._world, "_conn", None)
        if conn is None:
            return []
        snippets: list[str] = []
        for _score, sid, trace_id in entries:
            if trace_id is None:
                continue
            row = conn.execute(
                "SELECT raw_text FROM narrative WHERE pipeline_trace_id=? "
                "ORDER BY update_number DESC LIMIT 1",
                (trace_id,),
            ).fetchone()
            if row is None:
                # Fall back to scorecard.update_number based lookup.
                sc = conn.execute(
                    "SELECT update_number FROM scorecards WHERE id=?",
                    (sid,),
                ).fetchone()
                if sc is None:
                    continue
                row = conn.execute(
                    "SELECT raw_text FROM narrative WHERE update_number=?",
                    (int(sc["update_number"]),),
                ).fetchone()
                if row is None:
                    continue
            text = str(row["raw_text"] or "").strip()
            if not text:
                continue
            snippets.append(text[:400])
        return snippets

    def _build_replay_context(
        self, trace_id: str, dimension: str,
    ) -> ReplayContext | None:
        """Resolve a historical trace into a :class:`ReplayContext`."""
        conn = getattr(self._world, "_conn", None)
        if conn is None:
            return None

        # Baseline dim score for this trace (if any).
        row = conn.execute(
            "SELECT ds.score FROM dimension_scores ds "
            "JOIN scorecards s ON s.id = ds.scorecard_id "
            "WHERE s.pipeline_trace_id=? AND ds.dimension=? "
            "ORDER BY s.id DESC LIMIT 1",
            (trace_id, dimension),
        ).fetchone()
        baseline = float(row["score"]) if row is not None else None

        # Narrative record (raw_text + player_action + update_number).
        nar = conn.execute(
            "SELECT update_number, raw_text, player_action "
            "FROM narrative WHERE pipeline_trace_id=? LIMIT 1",
            (trace_id,),
        ).fetchone()
        player_action = str(nar["player_action"]) if nar is not None and nar["player_action"] else None
        update_number = int(nar["update_number"]) if nar is not None else None
        raw_text = str(nar["raw_text"]) if nar is not None and nar["raw_text"] else None

        return ReplayContext(
            trace_id=trace_id,
            player_action=player_action,
            update_number=update_number,
            narrative_text=raw_text,
            baseline_dim_score=baseline,
        )

    def _resolve_prompt_path(self, prompt_path: str | Path) -> Path:
        """Resolve a prompt path, allowing either absolute or repo-relative.

        Relative paths are joined against ``self._prompts_root``. This
        sandboxes :meth:`apply_mutation` so an attacker (or a rogue
        mutation proposer) can't be tricked into overwriting arbitrary
        files via a weird path.
        """
        p = Path(prompt_path)
        if p.is_absolute():
            return p
        return (self._prompts_root / p).resolve()


# ---------------------------------------------------------------------------
# Optional production-side proposer (wraps InferenceClient).
# ---------------------------------------------------------------------------


def claude_mutation_proposer(client: Any, *, model: str | None = None) -> MutationProposer:
    """Return a :data:`MutationProposer` that calls an inference client.

    The client must expose ``chat_structured`` in the same shape as
    :class:`app.runtime.client.InferenceClient`. Response schema is
    ``{"new_prompt": str, "rationale": str}``; a malformed response
    falls back to the original prompt with a note in ``rationale``.

    This is a *best-effort* adapter — production wiring lives in
    ``tools/optimize/run.py`` which handles the async bridging. Tests
    should stub :data:`MutationProposer` directly; they don't need this
    adapter at all.
    """
    import asyncio
    import json as _json

    from app.runtime.client import ChatMessage

    async def _call(dim: str, current_prompt: str, bad_examples: Sequence[str]) -> tuple[str, str]:
        system = (
            "You are a prompt-engineering assistant for an interactive-fiction "
            "pipeline. The user will give you one dimension of writing quality "
            "that is underperforming, the current prompt template, and 2-3 "
            "low-scoring outputs. Propose ONE minimal, targeted edit to the "
            "prompt that should raise that dimension's score. Do not rewrite "
            "the whole prompt; preserve everything unrelated to the issue."
        )
        parts: list[str] = [
            f"Dimension: {dim}",
            "",
            "=== Current prompt ===",
            current_prompt,
            "",
            "=== Low-scoring outputs ===",
        ]
        for i, ex in enumerate(bad_examples, start=1):
            parts.append(f"--- example {i} ---\n{ex}")
        user = "\n".join(parts)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "new_prompt": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["new_prompt", "rationale"],
        }
        try:
            raw = await client.chat_structured(
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                json_schema=schema,
                schema_name="PromptMutation",
                temperature=0.3,
            )
            payload = _json.loads(raw)
            new_prompt = str(payload.get("new_prompt") or current_prompt)
            rationale = str(payload.get("rationale") or "")
            return new_prompt, rationale
        except Exception as e:  # pragma: no cover - defensive
            return current_prompt, f"proposer-error: {type(e).__name__}: {e}"

    def _sync(dim: str, current_prompt: str, bad_examples: Sequence[str]) -> tuple[str, str]:
        return asyncio.run(_call(dim, current_prompt, bad_examples))

    return _sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_prompts_dir() -> Path:
    """Resolve ``prompts/`` relative to the project root.

    ``app/optimization/optimizer.py`` -> project root / prompts.
    """
    return Path(__file__).resolve().parent.parent.parent / "prompts"


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))
