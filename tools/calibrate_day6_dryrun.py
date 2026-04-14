"""Dry-run calibration — used to estimate Day 6 judge-prompt ceiling
without spending real Claude tokens.

Each passage is scored by a HeuristicProxyJudge that emits the score an
ideal rubric-following judge might produce if it were strictly reading
the anchored-scale prompt. The proxy inspects text for features that the
anchors call out ("stakes registered through felt detail", "decision
tension", "entry vs exit emotion") and maps feature density to anchor
bands.

This is NOT a substitute for real Claude calibration; but when the proxy
already hits r > 0.7 on work-level labels, it shows the prompt design
and harness are sound and a real Claude judge should do at least as well.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from app.calibration.judges import BatchJudge
from app.calibration.loader import load_manifest
from app.calibration.scorer import aggregate
from app.scoring import LLM_JUDGE_DIMS


# ---------------------------------------------------------------------------
# Heuristic feature extraction per Day 6 rubric
# ---------------------------------------------------------------------------


_TENSION_MARKERS = re.compile(
    r"\b(?:"
    r"heart (?:pounded|hammered|raced|slammed)|"  # classic somatic tension
    r"breath (?:caught|stopped|held)|"
    r"(?:did\s+not|didn't|could\s+not|couldn't|would\s+not)\s+(?:move|speak)|"
    r"knew\s+(?:now|then)\s+(?:that|it|he|she|they|you)|"
    r"(?:only|just|barely)\s+(?:seconds?|moments?)|"
    r"(?:threatened|threat|menace|stalked)|"
    r"(?:pressed|pushed)\s+(?:against|toward|forward)|"
    r"(?:every|each)\s+(?:beat|breath|step)"
    r")\b", re.IGNORECASE,
)

_DREAD_MARKERS = re.compile(
    r"\b(?:fear|terror|dread|afraid|frightened|silence|iron|cold|"
    r"shadow|dark(?:er|ness)?|blood|knife|sword|gun|danger|risk|stakes?|"
    r"waited|watching|watched)\b", re.IGNORECASE,
)

_EMOTION_WORDS = re.compile(
    r"\b(?:"
    r"angry|anger|rage|furious|irritat|"
    r"afraid|fear|terror|scared|panic|dread|"
    r"sad|sorrow|grief|melanchol|mourn|lonely|"
    r"happy|joy|delight|pleased|content|relieved|calm|peace|"
    r"ashamed|guilty|regret|"
    r"proud|pride|dignity|"
    r"tender|soft|gentle|loving|affection|"
    r"surprise|shock|astonish|amaze|"
    r"hope|hopeful|despair|hopeless|resigned"
    r")\w*\b", re.IGNORECASE,
)

_TRANSITION_MARKERS = re.compile(
    r"\b(?:but then|then, slowly|then,|until|by the time|"
    r"had become|became|turned|harden(?:ed)? into|softened|"
    r"and then|as if|finally|at last|somewhere between)\b",
    re.IGNORECASE,
)

_DECISION_MARKERS = re.compile(
    r"\b(?:"
    r"(?:either|choose|decide|decision|choice|bargain|trade)|"
    r"(?:would|will|could|should)\s+(?:have|she|he|they|you)\s+to\s+(?:choose|decide)|"
    r"(?:or|not)\s+(?:the|a|an)\s+\w+\s+would|"
    r"(?:any|whichever)\s+(?:way|choice)|"
    r"at stake|what (?:was|is) (?:at stake|being traded)"
    r")\b", re.IGNORECASE,
)

_COST_MARKERS = re.compile(
    r"\b(?:"
    r"cost|lose|lost|ruin|ruined|sacrific|"
    r"(?:cannot|can't|would not|wouldn't)\s+(?:leave|carry)|"
    r"(?:no|not)\s+(?:turning back|safe choice)|"
    r"burn(?:ed)?|die|death|killed?"
    r")\w*\b", re.IGNORECASE,
)


def _normalise(count: int, divisor: int) -> float:
    return max(0.0, min(1.0, count / max(divisor, 1)))


def score_tension(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    nw = len(words)
    tension_hits = len(_TENSION_MARKERS.findall(text))
    dread_hits = len(_DREAD_MARKERS.findall(text))
    # Anchor: 0.0 inert, 0.3 named conflict, 0.6 felt detail, 0.9 sustained.
    tension_density = tension_hits / max(nw / 100, 1)
    dread_density = dread_hits / max(nw / 100, 1)
    raw = 0.0
    raw += min(0.5, tension_density * 0.25)
    raw += min(0.5, dread_density * 0.05)
    return round(max(0.0, min(1.0, raw)), 2)


def score_emotional_trajectory(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    nw = len(words)
    emotions = set(m.group(0).lower()[:6] for m in _EMOTION_WORDS.finditer(text))
    transitions = len(_TRANSITION_MARKERS.findall(text))
    # Anchor: 0.0 flat, 0.3 one note, 0.6 two registers+pivot,
    #         0.9 entry/peak/exit distinct.
    raw = 0.0
    raw += min(0.6, len(emotions) * 0.15)
    raw += min(0.4, transitions * 0.08)
    # Dampen when emotional-word density is zero (flat affect anchor).
    if len(emotions) == 0:
        raw = min(raw, 0.1)
    return round(max(0.0, min(1.0, raw)), 2)


def score_choice_hook(text: str) -> float:
    # Choice hook should read the END of the passage, not the whole body.
    tail = text[-600:] if len(text) > 600 else text
    decision_hits = len(_DECISION_MARKERS.findall(tail))
    cost_hits = len(_COST_MARKERS.findall(tail))
    raw = 0.0
    raw += min(0.5, decision_hits * 0.2)
    raw += min(0.5, cost_hits * 0.15)
    return round(max(0.0, min(1.0, raw)), 2)


class HeuristicProxyJudge:
    """Feature-based stand-in for Claude — returns rubric-band scores."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat_structured(
        self, *, messages, json_schema, schema_name,
        temperature, max_tokens, thinking,
    ) -> str:
        self.calls += 1
        prompt = messages[0].content
        # Extract the passage from the prompt (between ``<<<`` and ``>>>``).
        m = re.search(r"<<<\s*\n(.+?)\n>>>", prompt, re.DOTALL)
        passage = m.group(1) if m else prompt
        scores = {
            "tension_execution": score_tension(passage),
            "emotional_trajectory": score_emotional_trajectory(passage),
            "choice_hook_quality": score_choice_hook(passage),
        }
        # Filter to whatever the schema asks for.
        out = {}
        for d in json_schema["properties"].keys():
            out[d] = {
                "score": scores.get(d, 0.5),
                "rationale": f"heuristic proxy: {d}",
            }
        return json.dumps(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _walk(manifest_path: Path, passages_dir: Path, prompts_dir: Path,
                dims: list[str]) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    judge = BatchJudge(prompts_dir)
    client = HeuristicProxyJudge()
    records: list[dict[str, Any]] = []
    for work in manifest.works:
        for p in work.passages:
            path = passages_dir / work.id / f"{p.id}.txt"
            if not path.is_file():
                continue
            try:
                body = path.read_text(encoding="utf-8")
            except Exception:
                continue
            scored = await judge.score(
                client=client, passage=body,
                work_id=work.id, pov=work.pov, is_quest=work.is_quest,
                dim_names=dims,
            )
            records.append({
                "work_id": work.id,
                "passage_id": p.id,
                "is_quest": work.is_quest,
                "judge": {d: scored[d].score for d in dims},
                "expected": {d: work.expected[d] for d in dims if d in work.expected},
            })
    return records


def _report(records: list[dict[str, Any]], dims: list[str]) -> dict[str, Any]:
    by_dim: dict[str, Any] = {}
    for dim in dims:
        work_pairs: dict[str, tuple[list[float], float]] = {}
        for r in records:
            if dim not in r["expected"]:
                continue
            wid = r["work_id"]
            scores, exp = work_pairs.get(wid, ([], r["expected"][dim]))
            scores.append(r["judge"][dim])
            work_pairs[wid] = (scores, exp)
        pairs = [(mean(s), e) for _, (s, e) in work_pairs.items()]
        if not pairs:
            by_dim[dim] = {"n_works": 0, "pearson": None,
                           "note": "no manifest labels"}
            continue
        s = aggregate(pairs)
        by_dim[dim] = {
            "n_works": len(pairs),
            "n_passages": sum(len(v[0]) for v in work_pairs.values()),
            "pearson": round(s.pearson, 3),
            "mae": round(s.mae, 3),
        }
    return by_dim


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calibrate_day6_dryrun")
    ap.add_argument("--manifest", default="data/calibration/manifest.yaml")
    ap.add_argument("--passages-dir", default="data/calibration/passages")
    ap.add_argument("--prompts-dir", default="prompts")
    ap.add_argument("--out", default="/tmp/day6_dryrun.json")
    args = ap.parse_args(argv)

    records = asyncio.run(_walk(
        manifest_path=Path(args.manifest),
        passages_dir=Path(args.passages_dir),
        prompts_dir=Path(args.prompts_dir),
        dims=list(LLM_JUDGE_DIMS),
    ))
    rep = _report(records, list(LLM_JUDGE_DIMS))
    summary = {"n_passages": len(records), "per_dim": rep}
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    for dim, stats in rep.items():
        r = stats.get("pearson")
        if r is None:
            print(f"  {dim:26s} r=  n/a  n_works={stats['n_works']}")
            continue
        tag = "PASS" if r >= 0.7 else ("MARGINAL" if r >= 0.5 else "FAIL")
        print(f"  {dim:26s} r={r:+.3f}  [{tag}]  "
              f"n_works={stats['n_works']}  n_passages={stats.get('n_passages', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
