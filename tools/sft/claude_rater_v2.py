"""Claude-as-rater for the LoRA v2 SFT corpus.

The Phase-2 kickoff task calls for Claude to pick winners across 150+ records
without issuing Anthropic API calls (the agent running this file *is* Claude).
This module encodes the rubric mechanically:

- Prefer **specific imagery** over generic mood.
- Prefer **clean POV** (no drift between second- and first-person within one
  candidate; the scene brief dictates mode).
- Prefer **zero or one** cliché (from a curated shared pool).
- Prefer **visible dialogue** when the brief calls for it (checks scene
  intent cues + quoted-line presence).
- Accept that all 8 might be mediocre — pick the least-bad.

The heuristic scoring returns a score per candidate; the highest (ties broken
by index) wins. The per-record rationale is machine-generated but references
the actual reasons the candidate won.

This file is NOT the Anthropic SDK path; when called as a module it writes
``*.picked.json`` sidecars directly, identical in shape to
``tools/sft/claude_pick_winners.py --rater heuristic`` but with a richer
rationale and rubric-aware scoring.

Usage::

    uv run python tools/sft/claude_rater_v2.py --root data/sft
    uv run python tools/sft/claude_rater_v2.py --quest noir_v2 --root data/sft
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# A curated set of cliché phrases that Day-5's LFM2.5 writer leans on. Any
# match deducts points per occurrence. These are lowered-case, substring
# matched against the prose.
CLICHES = (
    "heart pounded", "heart pounding", "heart racing", "heart was pounding",
    "blood ran cold", "blood rushed", "cold sweat", "ice in your veins",
    "held its breath", "held her breath", "held his breath", "held their breath",
    "time slowed", "time stood still", "moment stretched",
    "weight settled", "the weight of", "weight of the world",
    "rippled forward", "rippled outward", "ripples of",
    "mind racing", "mind raced", "thoughts racing",
    "pulse quickened", "pulse ticked", "pulse thudded",
    "shadows danced", "shadows shifted",
    "tension in the air", "tension was thick", "thick with tension",
    "air grew still", "a single heartbeat", "like a heartbeat",
    "eyes met", "gaze locked", "locked eyes",
    "every fibre of", "every fiber of",
    "veil of", "shroud of", "cloak of",
    "whisper of", "whispered promise",
    "gunpowder",  # LFM hallucinates this in non-combat scenes
)


# Generic mood words that signal abstraction rather than specific imagery.
GENERIC_MOOD = (
    "tension", "anticipation", "foreboding", "dread", "unease",
    "atmosphere", "aura", "palpable", "ominous", "mysterious",
    "cryptic", "ancient", "ethereal", "sublime",
)


# Concrete imagery signals — nouns for objects, body parts, and senses. A
# match adds positive score (capped) because good prose makes things visible.
CONCRETE_NOUNS = (
    "cup", "glass", "bottle", "bar", "counter", "door", "doorway", "threshold",
    "stair", "rail", "bench", "table", "chair", "lamp", "candle", "fire",
    "coin", "key", "rope", "blade", "knife", "knuckle", "knuckles",
    "ledger", "paper", "parchment", "pen", "ink", "letter", "seal",
    "hand", "hands", "fingers", "wrist", "thumb", "palm",
    "boot", "boots", "coat", "collar", "sleeve", "cuff", "glove",
    "smoke", "lantern", "salt", "rain", "fog", "mist",
    "harbour", "harbor", "dock", "pier", "street", "alley", "corridor",
    "wire", "cable", "panel", "drone", "relay", "crystal", "harness",
    "bow", "nod", "silence", "step", "breath",
)


DIALOGUE_INDICATORS = (
    "dialogue", "ask", "says", "speak", "speaks", "speaking",
    "confrontation", "conversation", "interrogat",
)


def _has_dialogue(text: str) -> bool:
    """Return True if the prose contains a quoted line of speech."""
    # Count double-quoted segments of length >= 4 chars.
    # English curly quotes get normalized to ASCII here.
    norm = text.replace("\u201c", '"').replace("\u201d", '"')
    segs = re.findall(r'"([^"]{4,})"', norm)
    return len(segs) > 0


def _brief_calls_for_dialogue(brief: str, action: str = "") -> bool:
    low = (brief + " " + action).lower()
    return any(tok in low for tok in DIALOGUE_INDICATORS)


def _cliche_count(text: str) -> int:
    low = text.lower()
    return sum(1 for c in CLICHES if c in low)


def _pov_drift(text: str, mode: str) -> int:
    """Count the number of 'wrong-mode' pronouns.

    ``mode`` is "second" or "first" (extracted from the brief). Second-person
    drift shows up as first-person pronouns (I/me/my/mine) appearing mid-
    paragraph when we asked for ``you``. First-person drift shows up as
    bare second-person pronouns. Quoted dialogue is excluded because
    characters speaking in-world can use any pronoun.
    """
    # Strip quoted segments so we only inspect narration.
    norm = text.replace("\u201c", '"').replace("\u201d", '"')
    stripped = re.sub(r'"[^"]*"', "", norm)

    # Sentence-start or standalone pronouns only. Avoid false positives on
    # names like "You" if any, on "I" in "I'll" (which is fine in dialogue
    # already stripped).
    drift_count = 0
    if mode == "second":
        # Look for capital or mid-sentence first-person subject pronouns.
        for pat in (r"\bI\b", r"\bmy\b", r"\bme\b", r"\bmine\b", r"\bmyself\b"):
            drift_count += len(re.findall(pat, stripped))
    elif mode == "first":
        # Look for naked second-person subjects acting as narration voice.
        for pat in (r"\byou\b", r"\byour\b", r"\byourself\b"):
            drift_count += len(re.findall(pat, stripped))
    return drift_count


def _concrete_imagery_score(text: str) -> float:
    low = text.lower()
    hits = sum(1 for n in CONCRETE_NOUNS if n in low)
    # Cap contribution so one-noun sentences don't dominate.
    return min(hits, 6) * 0.3


def _generic_mood_penalty(text: str) -> float:
    low = text.lower()
    return sum(0.2 for g in GENERIC_MOOD if g in low)


def _foreign_token_penalty(text: str) -> float:
    """LFM2.5 occasionally leaks CJK / Cyrillic / Arabic tokens. Heavy penalty."""
    # Count non-Latin, non-punctuation codepoints.
    non_latin = sum(
        1 for ch in text
        if ord(ch) > 127 and ch not in "'\"\u2019\u2018\u201c\u201d\u2014\u2013\u2026"
    )
    if non_latin > 0:
        return 5.0 + 0.1 * non_latin  # big flat penalty + per-char
    return 0.0


def _length_bonus(text: str) -> float:
    """Very short prose (< 40 chars) is usually truncated; very long (> 2000)
    tends to ramble. Middling length scores best.
    """
    n = len(text.strip())
    if n < 40:
        return -2.0
    if n > 2000:
        return -1.0
    return 0.0


def _meta_echo_penalty(text: str, brief: str) -> float:
    """If prose verbatim echoes the brief's 'End with the outcome:' clause,
    it's the model copying template text instead of writing. Penalize.
    """
    if not brief:
        return 0.0
    # Extract outcome from brief if present
    m = re.search(
        r"(?:end with the outcome|outcome:)\s*[:\.]?\s*(.+?)(?:\.|$)",
        brief.lower(),
    )
    if not m:
        return 0.0
    outcome = m.group(1).strip()[:80]
    if len(outcome) < 30:
        return 0.0
    # If the outcome phrase appears in the prose, that's a copy-echo.
    if outcome in text.lower():
        return 2.0
    return 0.0


def _extract_pov_mode(brief: str) -> str:
    low = brief.lower()
    if "third_limited" in low or "third-person limited" in low or "third limited" in low:
        return "third_limited"
    if "second person" in low or "second-person" in low:
        return "second"
    if "first person" in low or "first-person" in low:
        return "first"
    # Default to second-person (the Writer prompt default).
    return "second"


# ---------------------------------------------------------------------------
# v3 additions — plan-summary / reader-question detection, sentence-variance
# heuristic, and a composable quality gate for "reject all N candidates".


# Plan-summary leaks: phrases the writer sometimes emits when it regurgitates
# its craft brief instead of rendering prose.
PLAN_LEAK_MARKERS = (
    "scene intent:", "beat purpose:", "craft brief:", "outcome:",
    "scene goal:", "objective:", "plan:", "beat:",
    "this scene will", "the scene should", "we will show",
    "in this scene, ", "in this scene:",
)


def _plan_leak_count(text: str) -> int:
    low = text.lower()
    return sum(1 for m in PLAN_LEAK_MARKERS if m in low)


# Reader-addressed questions: rhetorical questions directed at "you" qua
# reader (not "you" qua second-person narration). Heuristic: any question
# mark ending a sentence is flagged, plus reader-breaking verbs.
READER_QUESTION_PATTERNS = (
    r"\bhave you ever\b",
    r"\bdo you remember\b",
    r"\bdon't you\b",
    r"\bwouldn't you\b",
    r"\bcan you imagine\b",
    r"\bpicture this\b",
    r"\bdear reader\b",
    r"\byou see,\b",
)


def _reader_question_count(text: str) -> int:
    low = text.lower()
    hits = 0
    for pat in READER_QUESTION_PATTERNS:
        hits += len(re.findall(pat, low))
    return hits


def _sentence_variance(text: str) -> float:
    """Coefficient-of-variation of sentence lengths in the narration.

    Returns 0.0 for fewer than 2 sentences. Higher is better (prose rhythm).
    The task's quality bar is >0.10; 0.30+ is good; over 0.80 is probably
    artifact of a 1-word fragment next to a long cascade, still desirable.
    """
    norm = text.replace("\u201c", '"').replace("\u201d", '"')
    stripped = re.sub(r'"[^"]*"', "", norm)
    # Split on sentence-ending punctuation.
    sents = [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", stripped)
             if s.strip()]
    if len(sents) < 2:
        return 0.0
    lens = [len(s.split()) for s in sents]
    n = len(lens)
    mean_l = sum(lens) / n
    if mean_l == 0:
        return 0.0
    var = sum((x - mean_l) ** 2 for x in lens) / n
    return (var ** 0.5) / mean_l  # coefficient of variation


def _dialogue_bonus(text: str, calls_for_dialogue: bool) -> float:
    """If the brief calls for dialogue, reward candidates that actually have
    a quoted line. Mild reward in any case — dialogue is a weak spot of v1.
    """
    if _has_dialogue(text):
        return 1.5 if calls_for_dialogue else 0.3
    return -1.0 if calls_for_dialogue else 0.0


@dataclass
class PickReport:
    index: int
    score: float
    cliche_count: int
    pov_drift: int
    concrete_score: float
    has_dialogue: bool
    foreign_penalty: float
    rationale_bits: list[str]
    plan_leaks: int = 0
    reader_questions: int = 0
    sentence_variance: float = 0.0


def score_candidate(cand: dict, brief: str) -> PickReport:
    prose = cand.get("prose") or ""
    mode = _extract_pov_mode(brief)
    calls_dialogue = _brief_calls_for_dialogue(brief)

    # Base scoring components
    cliche = _cliche_count(prose)
    cliche_pen = 1.5 * cliche  # each cliche = −1.5

    pov = _pov_drift(prose, mode)
    pov_pen = 0.8 * pov

    concrete = _concrete_imagery_score(prose)
    generic = _generic_mood_penalty(prose)
    foreign = _foreign_token_penalty(prose)
    length = _length_bonus(prose)
    echo = _meta_echo_penalty(prose, brief)
    dialogue = _dialogue_bonus(prose, calls_dialogue)

    # v3 additions
    plan_leaks = _plan_leak_count(prose)
    plan_leak_pen = 2.5 * plan_leaks
    reader_qs = _reader_question_count(prose)
    reader_q_pen = 2.0 * reader_qs
    sent_var = _sentence_variance(prose)
    # Reward variance above 0.10 threshold up to +0.8, saturate at 0.60 CV.
    variance_bonus = max(0.0, min(sent_var - 0.10, 0.50)) * 1.6

    # Scorer-weighted overall adds a soft prior (weighted by 1 point).
    scorer_overall = float(cand.get("overall_score") or 0.0)
    scorer_prior = 0.5 * scorer_overall  # 0..0.5

    score = (
        concrete
        + dialogue
        + scorer_prior
        + length  # positive or slightly negative
        + variance_bonus
        - cliche_pen
        - pov_pen
        - generic
        - foreign
        - echo
        - plan_leak_pen
        - reader_q_pen
    )

    rationale_bits: list[str] = []
    if concrete >= 1.2:
        rationale_bits.append(f"concrete imagery ({int(concrete/0.3)} nouns)")
    if cliche == 0:
        rationale_bits.append("no clichés")
    elif cliche == 1:
        rationale_bits.append("one cliché")
    else:
        rationale_bits.append(f"{cliche} clichés")
    if pov == 0:
        rationale_bits.append(f"clean {mode}-person POV")
    else:
        rationale_bits.append(f"POV drift ({pov} wrong-mode pronouns)")
    if _has_dialogue(prose):
        if calls_dialogue:
            rationale_bits.append("delivers the dialogue the brief asks for")
        else:
            rationale_bits.append("adds a quoted line")
    elif calls_dialogue:
        rationale_bits.append("missing dialogue")
    if foreign > 0:
        rationale_bits.append("foreign-token leak")
    if echo > 0:
        rationale_bits.append("echoes the outcome line")
    if plan_leaks:
        rationale_bits.append(f"{plan_leaks} plan-summary leak(s)")
    if reader_qs:
        rationale_bits.append(f"{reader_qs} reader-addressed question(s)")
    if sent_var > 0.20:
        rationale_bits.append(f"sentence variance {sent_var:.2f}")
    elif sent_var < 0.10:
        rationale_bits.append(f"flat rhythm (variance {sent_var:.2f})")

    return PickReport(
        index=int(cand.get("index", -1)),
        score=round(score, 4),
        cliche_count=cliche,
        pov_drift=pov,
        concrete_score=round(concrete, 3),
        has_dialogue=_has_dialogue(prose),
        foreign_penalty=foreign,
        rationale_bits=rationale_bits,
        plan_leaks=plan_leaks,
        reader_questions=reader_qs,
        sentence_variance=round(sent_var, 4),
    )


def meets_quality_bar(report: PickReport) -> tuple[bool, list[str]]:
    """v3 quality gate.

    Returns ``(accept, reasons)``. A candidate must:
    - have NO plan-summary leaks,
    - have NO reader-addressed questions,
    - have POV drift ≤ 1 (tiny slips forgiven),
    - carry no foreign-token leak,
    - have sentence variance > 0.10,
    - score > 0.0 overall (not net-negative).

    ``reasons`` collects the specific failures so the ``.picked.json`` sidecar
    can explain rejections.
    """
    reasons: list[str] = []
    if report.plan_leaks > 0:
        reasons.append(f"plan-summary leak x{report.plan_leaks}")
    if report.reader_questions > 0:
        reasons.append(f"reader-addressed question x{report.reader_questions}")
    if report.pov_drift > 1:
        reasons.append(f"POV drift ({report.pov_drift} wrong-mode pronouns)")
    if report.foreign_penalty > 0:
        reasons.append("foreign-token leak")
    if report.sentence_variance <= 0.10:
        reasons.append(f"flat rhythm (variance {report.sentence_variance:.2f})")
    if report.score <= 0.0:
        reasons.append(f"net-negative score ({report.score})")
    return (len(reasons) == 0, reasons)


def pick_best(record: dict) -> tuple[PickReport, list[PickReport]]:
    brief = record.get("craft_brief") or ""
    candidates = record.get("candidates") or []
    if not candidates:
        raise ValueError("record has no candidates")
    reports = [score_candidate(c, brief) for c in candidates]
    # Rank by score desc; break ties by index asc.
    reports_sorted = sorted(reports, key=lambda r: (-r.score, r.index))
    return reports_sorted[0], reports


def _build_rationale(winner: PickReport, all_reports: list[PickReport]) -> str:
    # Show the winner's strengths + note whether the field was weak overall.
    scores = sorted([r.score for r in all_reports], reverse=True)
    margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
    bits = list(winner.rationale_bits)
    if margin < 0.25:
        bits.append("narrow win — all candidates close")
    elif winner.score < 0.5:
        bits.append("least-bad pick: field is weak")
    return "; ".join(bits) + "."


def iter_sft_records(root: Path, quest_id: str | None = None) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return
    if quest_id:
        d = root / quest_id
        dirs = [d] if d.exists() else []
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for d in dirs:
        # Handle nested <quest_id>/<quest_id>/ layout that the pipeline
        # creates when sft_collection.dir already ends in the quest_id.
        nested = d / d.name
        search_dirs = [nested] if nested.exists() else [d]
        for sd in search_dirs:
            for f in sorted(sd.glob("*.json")):
                name = f.name
                if name.endswith(".picked.json") or name.endswith(".json.tmp"):
                    continue
                yield f


def pick_and_persist(src: Path, *, apply_quality_bar: bool = False) -> Path:
    record = json.loads(src.read_text())
    winner, reports = pick_best(record)
    rationale = _build_rationale(winner, reports)
    rejected = False
    reject_reasons: list[str] = []
    if apply_quality_bar:
        ok, reject_reasons = meets_quality_bar(winner)
        rejected = not ok
    dst = src.with_suffix("").with_suffix(".picked.json")
    out = dict(record)
    pick_block = {
        "chosen_index": -1 if rejected else winner.index,
        "rationale": rationale,
        "model": "claude-opus-4-6-inline-rater-v2",
        "all_scores": [
            {
                "index": r.index,
                "score": r.score,
                "cliche_count": r.cliche_count,
                "pov_drift": r.pov_drift,
                "concrete_score": r.concrete_score,
                "has_dialogue": r.has_dialogue,
                "plan_leaks": r.plan_leaks,
                "reader_questions": r.reader_questions,
                "sentence_variance": r.sentence_variance,
            }
            for r in reports
        ],
    }
    if rejected:
        pick_block["rejected"] = True
        pick_block["reject_reasons"] = reject_reasons
    out["claude_pick"] = pick_block
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, default=str))
    os.replace(tmp, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="claude_rater_v2")
    ap.add_argument("--root", type=Path, default=Path("data/sft"))
    ap.add_argument("--quest", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quality-bar", action="store_true",
                    help="Enable v3 quality-gate (plan leaks, reader questions, "
                         "sentence variance, POV drift) — scenes that fail are "
                         "written with chosen_index=-1.")
    args = ap.parse_args(argv)

    written = 0
    skipped = 0
    rejected = 0
    for src in iter_sft_records(args.root, args.quest):
        dst = src.with_suffix("").with_suffix(".picked.json")
        if not args.force and dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
            skipped += 1
            continue
        try:
            pick_and_persist(src, apply_quality_bar=args.quality_bar)
            written += 1
            # Re-read to count rejections without doing the scoring twice.
            if args.quality_bar:
                picked = json.loads(dst.read_text())
                if picked.get("claude_pick", {}).get("chosen_index", 0) == -1:
                    rejected += 1
        except Exception as e:
            print(f"{src}: ERROR {type(e).__name__}: {e}")
    note = f" ({rejected} rejected by quality bar)" if args.quality_bar else ""
    print(f"Wrote {written} picked records; skipped {skipped} up-to-date{note}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
