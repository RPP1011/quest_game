"""Example Curator: mine top/bottom prose into craft examples / anti-patterns.

The curator is symmetric to :class:`app.optimization.PromptOptimizer`:
same read path on ``scorecards`` + ``dimension_scores``, same lossy
reconstruction of prose via ``pipeline_trace_id``. Where the optimizer
edits *prompts*, the curator edits the *example bank* consumed by the
craft library and the writer prompt's ``## Avoid`` block.

Outputs
-------

- Top examples are appended to YAMLs under ``data/craft/examples/`` in
  the same ``Example`` record shape the craft library loader expects
  (see :class:`app.craft.schemas.Example`). A dim-to-tool mapping
  provided by the caller decides which ``tool_ids`` tag each record.
- Bottom examples are written under ``data/craft/anti_patterns/`` as
  ``<dim>.yaml`` plus a sidecar ``<dim>.meta.json`` that records per-example
  reasons ("why this is bad"). Anti-patterns deliberately live outside
  the craft-library tree so the ``CraftLibrary`` loader doesn't try to
  deserialize them as ``Example`` records.

Everything is opt-in — the CLI writer methods
(:meth:`update_craft_library`, :meth:`update_anti_patterns`) are explicit.
Nothing here is invoked automatically on commit.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class ExampleCandidate:
    """One candidate example — metadata + prose + score on the target dim.

    Emitted by :meth:`ExampleCurator.mine_top_examples` and
    :meth:`ExampleCurator.mine_bottom_examples`. ``scorecard_id`` is stable
    across curator runs (database-assigned), so ``id`` below is formed
    deterministically from ``(dim, scorecard_id)``.
    """

    dimension: str
    score: float
    snippet: str
    scorecard_id: int
    quest_id: str | None
    update_number: int | None
    trace_id: str | None

    @property
    def stable_id(self) -> str:
        """Deterministic slug safe for YAML / filename use."""
        base = f"{self.dimension}_{self.scorecard_id}"
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        return f"mined_{_slug(self.dimension)}_{self.scorecard_id}_{h}"


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------


_DEFAULT_SNIPPET_LIMIT = 800


class ExampleCurator:
    """Mine scorecards for good / bad examples on a target dimension."""

    def __init__(
        self,
        world: Any,
        *,
        craft_examples_dir: str | Path | None = None,
        anti_patterns_dir: str | Path | None = None,
        snippet_limit: int = _DEFAULT_SNIPPET_LIMIT,
    ) -> None:
        self._world = world
        self._craft_examples_dir = (
            Path(craft_examples_dir) if craft_examples_dir is not None
            else _default_craft_examples_dir()
        )
        self._anti_patterns_dir = (
            Path(anti_patterns_dir) if anti_patterns_dir is not None
            else _default_anti_patterns_dir()
        )
        self._snippet_limit = int(snippet_limit)

    # ---- mining ----

    def mine_top_examples(
        self,
        dimension: str,
        *,
        k: int = 5,
        quest_id: str | None = None,
        min_score: float = 0.75,
    ) -> list[ExampleCandidate]:
        """Return up to ``k`` highest-scoring candidates on ``dimension``.

        Filters to scorecards with ``dimension`` score ≥ ``min_score``.
        Returns fewer than ``k`` entries when there aren't enough good
        examples in history.
        """
        return self._mine(
            dimension=dimension,
            k=k,
            quest_id=quest_id,
            order="DESC",
            threshold=min_score,
            threshold_op=">=",
        )

    def mine_bottom_examples(
        self,
        dimension: str,
        *,
        k: int = 5,
        quest_id: str | None = None,
        max_score: float = 0.4,
    ) -> list[ExampleCandidate]:
        """Return up to ``k`` lowest-scoring candidates on ``dimension``."""
        return self._mine(
            dimension=dimension,
            k=k,
            quest_id=quest_id,
            order="ASC",
            threshold=max_score,
            threshold_op="<=",
        )

    # ---- writers ----

    def update_craft_library(
        self,
        top_examples: Sequence[ExampleCandidate],
        dim_to_tool_id_mapping: dict[str, list[str]],
        *,
        annotation: str | None = None,
    ) -> Path | None:
        """Append ``top_examples`` to ``data/craft/examples/mined.yaml``.

        ``dim_to_tool_id_mapping`` is required: the craft library's
        :class:`app.craft.schemas.Example` record insists on at least one
        ``tool_ids`` entry, and the optimizer itself has no knowledge of
        which craft tools a given dim targets. Examples whose dim is not
        in the mapping are skipped silently.

        If the destination file does not exist it is created with an
        ``examples: []`` header. Existing ``id`` values are preserved —
        a mined candidate with the same ``stable_id`` as an existing
        record overwrites it (so repeated curation runs don't accumulate
        duplicates).

        Returns the path written, or ``None`` if nothing was mapped.
        """
        by_id: dict[str, dict[str, Any]] = {}
        for cand in top_examples:
            tool_ids = dim_to_tool_id_mapping.get(cand.dimension) or []
            if not tool_ids:
                continue
            rec = {
                "id": cand.stable_id,
                "tool_ids": list(tool_ids),
                "source": "mined",
                "scale": "scene",
                "snippet": cand.snippet,
                "annotation": (
                    annotation
                    or f"Mined from quest={cand.quest_id!r} update={cand.update_number}; "
                    f"scored {cand.score:.2f} on {cand.dimension}."
                ),
            }
            by_id[cand.stable_id] = rec

        if not by_id:
            return None

        self._craft_examples_dir.mkdir(parents=True, exist_ok=True)
        path = self._craft_examples_dir / "mined.yaml"
        existing: list[dict[str, Any]] = []
        if path.is_file():
            raw = yaml.safe_load(path.read_text()) or {}
            existing = list(raw.get("examples") or [])
        # Overwrite any collisions on id
        kept = [e for e in existing if e.get("id") not in by_id]
        merged = kept + list(by_id.values())
        path.write_text(yaml.safe_dump(
            {"examples": merged}, sort_keys=False, allow_unicode=True,
        ))
        return path

    def update_anti_patterns(
        self,
        bottom_examples: Sequence[ExampleCandidate],
        *,
        reason_provider: "Any | None" = None,
    ) -> list[Path]:
        """Write ``bottom_examples`` to ``data/craft/anti_patterns/<dim>.yaml``.

        Sidecars a ``<dim>.meta.json`` that records per-example reasons.
        ``reason_provider`` is a callable ``(candidate) -> str``; when
        ``None`` a synthetic rationale is used
        ("scored 0.12 on free_indirect_quality").

        Returns the list of YAMLs written. Anti-patterns never touch the
        craft library's loader path, so their shape is not constrained by
        :class:`app.craft.schemas.Example`.
        """
        by_dim: dict[str, list[ExampleCandidate]] = {}
        for cand in bottom_examples:
            by_dim.setdefault(cand.dimension, []).append(cand)

        self._anti_patterns_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for dim, cands in by_dim.items():
            yaml_path = self._anti_patterns_dir / f"{_slug(dim)}.yaml"
            meta_path = self._anti_patterns_dir / f"{_slug(dim)}.meta.json"

            existing: list[dict[str, Any]] = []
            if yaml_path.is_file():
                raw = yaml.safe_load(yaml_path.read_text()) or {}
                existing = list(raw.get("anti_patterns") or [])
            by_id = {e["id"]: e for e in existing if "id" in e}

            existing_meta: dict[str, dict[str, Any]] = {}
            if meta_path.is_file():
                try:
                    existing_meta = json.loads(meta_path.read_text())
                except Exception:
                    existing_meta = {}

            for c in cands:
                if reason_provider is not None:
                    try:
                        reason = str(reason_provider(c))
                    except Exception:
                        reason = f"scored {c.score:.2f} on {c.dimension}"
                else:
                    reason = f"scored {c.score:.2f} on {c.dimension}"
                entry = {
                    "id": c.stable_id,
                    "dimension": c.dimension,
                    "score": round(float(c.score), 4),
                    "snippet": c.snippet,
                }
                by_id[c.stable_id] = entry
                existing_meta[c.stable_id] = {
                    "reason": reason,
                    "quest_id": c.quest_id,
                    "update_number": c.update_number,
                    "scorecard_id": c.scorecard_id,
                    "trace_id": c.trace_id,
                }

            merged = sorted(by_id.values(), key=lambda e: e["score"])
            yaml_path.write_text(yaml.safe_dump(
                {"anti_patterns": merged},
                sort_keys=False, allow_unicode=True,
            ))
            meta_path.write_text(json.dumps(existing_meta, indent=2))
            written.extend([yaml_path, meta_path])
        return written

    # ---- internals ----

    def _mine(
        self,
        *,
        dimension: str,
        k: int,
        quest_id: str | None,
        order: str,
        threshold: float,
        threshold_op: str,
    ) -> list[ExampleCandidate]:
        assert order in {"ASC", "DESC"}
        assert threshold_op in {">=", "<="}
        conn = getattr(self._world, "_conn", None)
        if conn is None:
            return []
        params: list[Any] = [dimension, float(threshold)]
        where = ["ds.dimension = ?", f"ds.score {threshold_op} ?"]
        if quest_id is not None:
            where.append("s.quest_id = ?")
            params.append(quest_id)
        sql = (
            "SELECT ds.score, s.id AS scorecard_id, s.quest_id, "
            "s.update_number, s.pipeline_trace_id "
            "FROM dimension_scores ds JOIN scorecards s "
            "ON s.id = ds.scorecard_id "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY ds.score {order}, s.id DESC LIMIT ?"
        )
        params.append(int(k))
        rows = conn.execute(sql, tuple(params)).fetchall()
        out: list[ExampleCandidate] = []
        for r in rows:
            snippet = self._lookup_prose(
                int(r["scorecard_id"]),
                r["pipeline_trace_id"],
                int(r["update_number"]),
            )
            if not snippet:
                continue
            out.append(ExampleCandidate(
                dimension=dimension,
                score=float(r["score"]),
                snippet=snippet[: self._snippet_limit],
                scorecard_id=int(r["scorecard_id"]),
                quest_id=r["quest_id"],
                update_number=int(r["update_number"]),
                trace_id=r["pipeline_trace_id"],
            ))
        return out

    def _lookup_prose(
        self,
        scorecard_id: int,
        trace_id: str | None,
        update_number: int,
    ) -> str | None:
        """Resolve a scorecard back to its committed prose.

        Priority: ``pipeline_trace_id`` (unique per commit) ⇒
        ``update_number`` (unique per quest). Returns ``None`` when the
        narrative row is missing or empty.
        """
        conn = getattr(self._world, "_conn", None)
        if conn is None:
            return None
        if trace_id:
            row = conn.execute(
                "SELECT raw_text FROM narrative WHERE pipeline_trace_id=? LIMIT 1",
                (trace_id,),
            ).fetchone()
            if row is not None and row["raw_text"]:
                return str(row["raw_text"]).strip()
        row = conn.execute(
            "SELECT raw_text FROM narrative WHERE update_number=? LIMIT 1",
            (int(update_number),),
        ).fetchone()
        if row is not None and row["raw_text"]:
            return str(row["raw_text"]).strip()
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slug(s: str) -> str:
    """A filesystem-safe lowercase slug. Collapses runs of punctuation to '_'."""
    out = _SLUG_RE.sub("_", s).strip("_").lower()
    return out or "unnamed"


def _default_craft_examples_dir() -> Path:
    """``data/craft/examples/`` relative to the project root."""
    return _project_root() / "data" / "craft" / "examples"


def _default_anti_patterns_dir() -> Path:
    """``data/craft/anti_patterns/`` relative to the project root."""
    return _project_root() / "data" / "craft" / "anti_patterns"


def _project_root() -> Path:
    """``app/optimization/curator.py`` -> project root."""
    return Path(__file__).resolve().parent.parent.parent
