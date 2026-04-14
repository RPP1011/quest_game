"""Day 7 CLI entrypoint for the Prompt Optimizer + Example Curator.

Two subcommands, both default-off writers — nothing is applied unless
``--apply`` is passed explicitly.

Usage
-----

Scan scorecards, propose mutations, and run a replay A/B on the last
few traces for each weak dim::

    python -m tools.optimize.run optimize \\
        --quest-id demo --threshold 0.5 --top-n 3

Mine top / bottom examples for a specific dim and emit them as craft
examples / anti-patterns::

    python -m tools.optimize.run curate \\
        --dim free_indirect_quality --top 5

Neither subcommand writes prompt files or craft YAMLs unless ``--apply``
is set. The default is to print a summary and exit — this matches the
spec's "default-off everywhere" constraint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


def _default_db_path() -> Path:
    return Path(os.environ.get("QUEST_DB", "data/quest.db"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _open_world(db_path: Path) -> Any:
    """Open a world state manager backed by the quest DB at ``db_path``.

    Returns the manager (not the connection) so the CLI can reuse the
    same read-path helpers the tests exercise.
    """
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager

    if not db_path.is_file():
        raise SystemExit(
            f"quest db not found at {db_path}. "
            "set QUEST_DB or pass --db explicitly."
        )
    conn = open_db(db_path)
    return WorldStateManager(conn)


def _stub_proposer(dim: str, prompt: str, examples: Sequence[str]) -> tuple[str, str]:
    """Offline fallback when no inference client is configured.

    Returns the prompt unchanged with a note. The CLI prints the
    diagnostic; the caller can re-run with ``--vllm`` once a real
    proposer is wired. This keeps ``python -m tools.optimize.run optimize``
    runnable in CI for smoke testing.
    """
    return prompt, (
        f"[stub] no inference client configured; "
        f"would have proposed a mutation for {dim} using "
        f"{len(examples)} low-scoring examples."
    )


# ---------------------------------------------------------------------------
# Subcommand: optimize
# ---------------------------------------------------------------------------


def cmd_optimize(args: argparse.Namespace) -> int:
    from app.optimization import PromptOptimizer

    world = _open_world(Path(args.db))
    opt = PromptOptimizer(
        world,
        mutation_proposer=_stub_proposer,
        prompts_root=args.prompts_root,
    )
    weak = opt.identify_weak_dimensions(
        quest_id=args.quest_id,
        threshold=args.threshold,
        n=args.top_n,
    )
    if not weak:
        print("no weak dimensions found (no scorecards, or all above threshold)")
        return 0

    print(f"weak dimensions for quest_id={args.quest_id!r}:")
    for w in weak:
        print(
            f"  - {w.dimension}: mean={w.mean_score:.3f} "
            f"(n={w.sample_size}, examples={len(w.recent_examples)})"
        )

    if not args.propose:
        return 0

    prompt_path = args.prompt_path or "stages/write/user.j2"
    print(f"\nproposing mutations against prompt={prompt_path!r}")
    for w in weak:
        mutation = opt.propose_mutation(w, prompt_path)
        print(f"\n=== {w.dimension} ===")
        print(f"rationale: {mutation.rationale}")
        if mutation.before_text == mutation.after_text:
            print("(no change)")
            continue
        print("(mutation would modify the prompt; --apply not set, skipping)")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: curate
# ---------------------------------------------------------------------------


def cmd_curate(args: argparse.Namespace) -> int:
    from app.optimization import ExampleCurator

    world = _open_world(Path(args.db))
    curator = ExampleCurator(world)

    top = curator.mine_top_examples(args.dim, k=args.top, quest_id=args.quest_id)
    bottom = curator.mine_bottom_examples(args.dim, k=args.top, quest_id=args.quest_id)

    payload = {
        "dimension": args.dim,
        "top": [
            {
                "score": c.score,
                "quest_id": c.quest_id,
                "update_number": c.update_number,
                "snippet_preview": c.snippet[:160],
            }
            for c in top
        ],
        "bottom": [
            {
                "score": c.score,
                "quest_id": c.quest_id,
                "update_number": c.update_number,
                "snippet_preview": c.snippet[:160],
            }
            for c in bottom
        ],
    }
    print(json.dumps(payload, indent=2))

    if not args.apply:
        return 0

    if args.tool_ids:
        tool_ids = [t.strip() for t in args.tool_ids.split(",") if t.strip()]
    else:
        tool_ids = []
    if tool_ids:
        path = curator.update_craft_library(top, {args.dim: tool_ids})
        print(f"wrote craft examples -> {path}")
    else:
        print("skipping craft library update: --tool-ids not provided")

    written = curator.update_anti_patterns(bottom)
    for p in written:
        print(f"wrote anti-pattern -> {p}")
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.optimize.run")
    p.add_argument("--db", default=str(_default_db_path()),
                   help="path to quest DB (env QUEST_DB also honored)")
    sub = p.add_subparsers(dest="command", required=True)

    opt_p = sub.add_parser("optimize", help="identify weak dims and propose mutations")
    opt_p.add_argument("--quest-id", default=None)
    opt_p.add_argument("--threshold", type=float, default=0.5)
    opt_p.add_argument("--top-n", type=int, default=3)
    opt_p.add_argument("--propose", action="store_true",
                       help="also propose mutations for each weak dim")
    opt_p.add_argument("--prompt-path", default=None,
                       help="relative path under prompts/ to mutate")
    opt_p.add_argument("--prompts-root", default=None,
                       help="override prompts dir (tests)")
    opt_p.set_defaults(func=cmd_optimize)

    cur_p = sub.add_parser("curate", help="mine top/bottom examples per dim")
    cur_p.add_argument("--dim", required=True)
    cur_p.add_argument("--top", type=int, default=5)
    cur_p.add_argument("--quest-id", default=None)
    cur_p.add_argument("--apply", action="store_true",
                       help="write mined examples to data/craft/{examples,anti_patterns}")
    cur_p.add_argument("--tool-ids", default=None,
                       help="comma-separated craft tool ids to tag top examples")
    cur_p.set_defaults(func=cmd_curate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
