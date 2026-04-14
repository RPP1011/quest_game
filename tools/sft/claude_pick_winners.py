"""Dispatch Claude raters over saved SFT scene records to pick the best
candidate on prose quality.

Input shape: ``data/sft/<quest_id>/u<update>_s<scene>_<trace>.json`` — the
artefact ``app.engine.pipeline.Pipeline._persist_sft_record`` writes when
``quest_config["sft_collection"]["enabled"]`` is on.

Output shape: a sibling ``*.picked.json`` that re-emits the source record plus
``claude_pick = {"chosen_index": int, "rationale": str, "model": str}``.

Rating rubric (kept terse — the Claude subagent sees the craft brief + the N
candidate prose bodies and picks the best on its own judgement):

- FIS quality: does the narrator adopt the POV character's idiom and
  perception? Or does the narrator stay external and neutral?
- Voice consistency: narrator + character register stable within the passage;
  no jarring register shifts.
- Sensory grounding: concrete, specific perception; not generic.
- Cliche absence: avoid tired phrasing ("heart pounded", "blood ran cold",
  etc.).
- POV adherence: second-person ``you`` / first-person ``I`` as the scene
  specifies, no drift.

Usage::

    uv run python -m tools.sft.claude_pick_winners --quest q-demo
    uv run python -m tools.sft.claude_pick_winners --root data/sft --quest q-demo

When run without ``--quest``, all subdirectories under ``--root`` are walked.
Already-picked scenes (``*.picked.json`` exists and is newer than source)
are skipped.

The rater itself is kept behind a :class:`Rater` protocol so we can swap in
the real Anthropic SDK or Claude CLI subagent call without rewriting the
walker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol


SYSTEM_PROMPT = """You are a terse prose critic picking the best of N candidate
scene drafts. Judge on prose quality alone — NOT on whether the draft matches
any heuristic score. Apply this rubric, in order of importance:

1. Free-indirect style: does the narrator adopt the POV character's idiom and
   perception, or does it stay external and neutral?
2. Voice consistency: register stable; no jarring shifts; no modern-author
   tics if the scene calls for a period voice.
3. Sensory grounding: concrete, specific perception — a particular object,
   sound, smell — not generic.
4. Cliche absence: avoid tired phrasing ("heart pounded", "blood ran cold",
   "mind racing").
5. POV adherence: respects the scene's POV mode (second-person you / first-
   person I) without drift.

Emit strictly one JSON object: {"chosen_index": <int>, "rationale": "<1-2
sentence justification>"}. No prose before or after."""


USER_TEMPLATE = """Craft brief:
{brief}

Candidates:
{candidates_block}

Pick the best candidate on prose quality using the rubric. Emit only the JSON
object."""


def _format_candidates_block(candidates: list[dict]) -> str:
    lines: list[str] = []
    for c in candidates:
        idx = c["index"]
        prose = c["prose"]
        lines.append(f"--- Candidate {idx} ---\n{prose}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# Rater abstraction — lets callers swap in the real Claude subagent while
# keeping the walker/dispatcher unit-testable with a deterministic fake.


@dataclass
class RatingInput:
    brief: str
    candidates: list[dict]      # each item has "index" and "prose"


@dataclass
class RatingResult:
    chosen_index: int
    rationale: str
    model: str


class Rater(Protocol):
    def rate(self, inp: RatingInput) -> RatingResult: ...


class AnthropicSDKRater:
    """Real Claude rater. Requires the ``anthropic`` package and
    ``ANTHROPIC_API_KEY`` in the environment.

    Not covered by tests (per Day-4 scope note). Kept here for the live
    collection pass.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        self._model = model

    def rate(self, inp: RatingInput) -> RatingResult:  # pragma: no cover
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed; "
                "`uv pip install anthropic` or use --rater heuristic."
            ) from e
        client = Anthropic()
        user_msg = USER_TEMPLATE.format(
            brief=inp.brief or "(no brief)",
            candidates_block=_format_candidates_block(inp.candidates),
        )
        resp = client.messages.create(
            model=self._model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()  # type: ignore[attr-defined]
        data = json.loads(text)
        return RatingResult(
            chosen_index=int(data["chosen_index"]),
            rationale=str(data.get("rationale", "")),
            model=self._model,
        )


class HeuristicFallbackRater:
    """Pick the candidate with the highest ``weighted_score``.

    Useful for dry-run plumbing without burning Claude tokens, and for test
    harness scaffolding.
    """

    def rate(self, inp: RatingInput) -> RatingResult:
        ranked = sorted(
            inp.candidates,
            key=lambda c: (-float(c.get("weighted_score", 0.0)), int(c["index"])),
        )
        top = ranked[0]
        return RatingResult(
            chosen_index=int(top["index"]),
            rationale="Heuristic fallback: highest weighted scorer score.",
            model="heuristic/weighted_score",
        )


# --------------------------------------------------------------------------
# File-walker + dispatcher — unit-testable with the fallback rater.


def iter_sft_records(
    root: Path,
    quest_id: str | None = None,
) -> Iterable[Path]:
    """Yield source SFT record paths under ``root``.

    - If ``quest_id`` is given, restrict to ``root/<quest_id>/``.
    - Ignore ``*.picked.json`` / ``*.json.tmp`` files.
    - Sorted for deterministic walking.
    """
    root = Path(root)
    if not root.exists():
        return
    dirs: list[Path]
    if quest_id:
        target = root / quest_id
        dirs = [target] if target.exists() else []
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            name = f.name
            if name.endswith(".picked.json"):
                continue
            if name.endswith(".json.tmp"):
                continue
            yield f


def pick_path(src: Path) -> Path:
    """Sidecar path: ``foo.json`` → ``foo.picked.json``."""
    return src.with_suffix("").with_suffix(".picked.json")


def is_stale(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    return src.stat().st_mtime > dst.stat().st_mtime


def apply_rating(
    src: Path,
    dst: Path,
    rater: Rater,
) -> dict:
    """Read ``src``, invoke the rater, write ``dst`` atomically, return the
    augmented record.
    """
    record = json.loads(src.read_text())
    inp = RatingInput(
        brief=record.get("craft_brief") or "",
        candidates=record.get("candidates") or [],
    )
    if not inp.candidates:
        raise ValueError(f"{src}: no candidates in record")
    result = rater.rate(inp)
    out = dict(record)
    out["claude_pick"] = {
        "chosen_index": result.chosen_index,
        "rationale": result.rationale,
        "model": result.model,
    }
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, default=str))
    os.replace(tmp, dst)
    return out


def walk_and_rate(
    root: Path,
    *,
    quest_id: str | None = None,
    rater: Rater,
    force: bool = False,
    on_done: Callable[[Path, dict], None] | None = None,
) -> list[Path]:
    """Walk ``root``, rate stale records, return the list of dst paths
    written in this run.
    """
    written: list[Path] = []
    for src in iter_sft_records(root, quest_id=quest_id):
        dst = pick_path(src)
        if not force and not is_stale(src, dst):
            continue
        rec = apply_rating(src, dst, rater)
        written.append(dst)
        if on_done is not None:
            on_done(dst, rec)
    return written


# --------------------------------------------------------------------------


def _build_rater(name: str, model: str | None) -> Rater:
    if name == "anthropic":
        return AnthropicSDKRater(model=model or "claude-sonnet-4-5-20250929")
    if name == "heuristic":
        return HeuristicFallbackRater()
    raise ValueError(f"unknown rater: {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="claude_pick_winners",
        description=(
            "Walk SFT scene records and ask a Claude rater to pick the "
            "best candidate by prose quality. Writes sibling "
            "*.picked.json files."
        ),
    )
    ap.add_argument("--root", type=Path, default=Path("data/sft"),
                    help="Root directory for SFT records (default: data/sft)")
    ap.add_argument("--quest", type=str, default=None,
                    help="Restrict walking to one quest_id subdirectory.")
    ap.add_argument("--rater", choices=("anthropic", "heuristic"),
                    default="anthropic",
                    help="Rater backend (default: anthropic; heuristic is a "
                         "dry-run using weighted_score).")
    ap.add_argument("--model", type=str, default=None,
                    help="Anthropic model id (for --rater anthropic).")
    ap.add_argument("--force", action="store_true",
                    help="Re-rate even if *.picked.json is up-to-date.")
    args = ap.parse_args(argv)

    rater = _build_rater(args.rater, args.model)

    def _on_done(dst: Path, rec: dict) -> None:
        pick = rec.get("claude_pick") or {}
        print(
            f"picked {pick.get('chosen_index')} for "
            f"{dst.relative_to(args.root)}"
        )

    written = walk_and_rate(
        args.root,
        quest_id=args.quest,
        rater=rater,
        force=args.force,
        on_done=_on_done,
    )
    print(f"Wrote {len(written)} picked records under {args.root}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
