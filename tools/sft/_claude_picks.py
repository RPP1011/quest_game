"""Claude-as-editor: hand-coded picks over the 11 SFT demo records.

This script is the output of the Day-5 Claude rater pass. Instead of routing
the picks through the Anthropic SDK (which would rerun judgment), the Claude
session that ran Day-5 made picks directly — reading each brief and all 8
candidates, applying the Day-4 rubric, and encoding the verdicts below.

Run ``uv run python -m tools.sft._claude_picks`` to write ``*.picked.json``
sidecars next to every source record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


MODEL_TAG = "claude-opus-4-6-inline-editor"


# Each entry: source filename -> (chosen_index, rationale)
# Rationales follow the Day-4 rubric: FIS > voice consistency > sensory
# grounding > cliché absence > POV adherence. All candidates on this corpus
# are heavily flawed (LFM2.5-1.2B base, no retrieval at generation time,
# foreign-token leakage); picks are "least-bad" where noted.
PICKS: dict[str, tuple[int, str]] = {
    # u10_s8 — cryptic warning in a ledger. All 8 drift toward "whisper"/
    # "weight"/"thick" filler. #3 and #4 are the cleanest POV-wise; #3 keeps
    # second-person consistently and picks one concrete image ("bargained
    # smile dull as moonlight") that isn't in the shared cliche-pool.
    "u10_s8_b755ed0ba8524a16b13eb3de6fd1f4f8.json": (
        3,
        "Least-bad pick. Holds second-person, offers one specific image "
        "('bargained smile dull as moonlight') and avoids the foreign-token "
        "leakage that wrecks #6/#7. Still cliche-soup overall ('pulse ticks', "
        "'weight of unspoken rules').",
    ),
    # u10_s9 — same scene, studying the parchment. #0 is the only one that
    # both stays in second-person and gives a concrete beat ("fingers tapping
    # a rhythm only you seem to hear") rather than pure abstraction.
    "u10_s9_b755ed0ba8524a16b13eb3de6fd1f4f8.json": (
        0,
        "Cleanest POV adherence and one concrete gesture (innkeeper's "
        "rhythmic tapping). Ends on a vague moral ('who you truly are') but "
        "avoids the foreign-token breakdown in #2/#4/#6/#7.",
    ),
    # u1_s42 — confront the innkeeper. All 8 are bad; #0 is the least
    # nonsensical. It has a concrete object (the key clicking in the chest)
    # and a usable line of dialogue. Everything else either dissolves into
    # foreign tokens (#3,#4,#5,#6,#7) or switches to first-person.
    "u1_s42_9fb51df066d848e3bd0ff7f8bf1a28df.json": (
        0,
        "Least-bad. Second person holds, one concrete prop (key, chest, "
        "scattered cargo), one usable line. Still relies on 'heart pounding' "
        "and 'the weight of the moment.' Scorer picked the same one.",
    ),
    # u2_s42 — threshold of revelation. Brief is abstract ('camera lingers on
    # the hands that move') and every candidate reads like meta-instructions
    # to itself. #2 has the best sensory concretion ('subtle shift in the
    # weight of your shoulders') and doesn't break POV.
    "u2_s42_e1f02283ba6c46af8587ff57160f7f7f.json": (
        2,
        "Best sensory grounding ('subtle shift in the weight of your "
        "shoulders', 'thuit of your own breath' — despite the typo). Holds "
        "second-person. Other candidates either slip to first-person (#1,#5,"
        "#6) or mix foreign tokens (#6).",
    ),
    # u3_s42 — reveal cargo location. Most candidates collapse into
    # instructional meta-text ('The outcome: ...'). #2 has the most concrete
    # action beat (grip rusted latch, door groans open, click the lever) and
    # stays in past-tense second-person as the brief requires.
    "u3_s42_018cb221140e4bc08b4d709d26193b50.json": (
        2,
        "Only candidate with a real action beat (latch, lever, door). Brief "
        "asks for 'concrete, sensory, unrushed' and this is closest. Still "
        "drops 'like gunpowder' which is exactly the wrong register. Scorer "
        "picked #1; I disagree — #1 ends on a literal outcome-echo of the "
        "brief, which is a failure mode.",
    ),
    # u4_s42 — decisive move. #6 has the specific visual ('worn stain
    # swallowing the light', 'cupped hand pulled from your pocket') that
    # most others lack. Short and less prone to falling apart mid-paragraph.
    "u4_s42_6b06b2cd38414583b1fdcbb3b3ecdf2b.json": (
        6,
        "One specific image (worn stain swallowing the light) and a "
        "believable beat. Holds POV. Scorer picked #4 (the 'traditional "
        "white' opener) which is a canned phrase repeated verbatim in 4+ "
        "records — overfitting signal, not a winner.",
    ),
    # u5_s42 — trust innkeeper vs. chase cargo. #0 is the only candidate
    # that both gives sensory specifics ('scent of woodsmoke', 'harness
    # rough against your thigh') and makes a clear choice. The rest either
    # drop into meta-instructions ('The outcome: ...') or foreign tokens.
    "u5_s42_baf17c2eef6e42e68ccb85844d35e132.json": (
        0,
        "Most sensory specifics ('woodsmoke', 'harness rough against your "
        "thigh') and commits to a choice. Scorer picked #4 (the traditional-"
        "white canned opener) which is overfit junk.",
    ),
    # u6_s42 — choice feels heavier than the weight on your sandals. Five of
    # the eight just echo the prompt's final line verbatim. #2 adds the
    # strongest sensory concretion ('scent of rain and forgotten roads') and
    # holds second-person without collapsing.
    "u6_s42_eb0ca761e93549f4878bed31314b4208.json": (
        2,
        "One fresh image (rain and forgotten roads) vs. the other seven "
        "which echo the brief's closing line as content. Brief effectively "
        "dictates the last sentence; picking on the prose leading into it.",
    ),
    # u7_s42 — investigate cryptic warning. #2 crouches low beside a
    # crumpled map, opens a cargo door — it's the only one with a
    # progression of physical actions rather than just atmosphere. Holds POV.
    "u7_s42_50bf55b5281e44bcae0967254d093f21.json": (
        2,
        "Only candidate with a sequence of physical actions (crouch, reach, "
        "step forward, open). Most concrete sensory work on the corpus. "
        "Still ends on meta-echo of the brief. Scorer agrees on this one.",
    ),
    # u8_s42 — cryptic note, riddles. #1 has the most coherent arc: sees
    # the note, reads it, understands. No foreign-token leakage. Stays in
    # second-person. Ending is meta-echoey but structurally intact.
    "u8_s42_ca560e49926748b59fbc851813de6b61.json": (
        1,
        "Most coherent mini-arc (approach, read, understand, decide). No "
        "foreign-token breakage. Scorer picked #0 which is fine too; #1 is "
        "slightly more specific on the hand of the scribe ('scribe had "
        "paused mid-thought').",
    ),
    # u9_s42 — investigate, risking exposure. Many candidates echo the
    # outcome line twice as a tell that the model couldn't escape the
    # template. #2 has the most specific setting ('beneath the old wooden
    # desk', 'light catches the tremor in the lines') and doesn't
    # double-print the outcome.
    "u9_s42_b1139d3a916848a78936389d03853d7a.json": (
        2,
        "Cleanest staging (crouch beneath desk, parchment, trembling lines). "
        "Doesn't double-print the outcome line like #0/#4/#5 do. Scorer "
        "agrees.",
    ),
}


def main() -> None:
    root = Path("data/sft/demo/demo")
    written = 0
    for fname, (idx, rationale) in PICKS.items():
        src = root / fname
        if not src.exists():
            print(f"MISSING: {src}")
            continue
        record = json.loads(src.read_text())
        # Validate the index is real before writing.
        cand_idxs = {int(c.get("index", -1)) for c in record.get("candidates") or []}
        if idx not in cand_idxs:
            raise SystemExit(f"{fname}: chosen_index {idx} not in {cand_idxs}")
        out = dict(record)
        out["claude_pick"] = {
            "chosen_index": idx,
            "rationale": rationale,
            "model": MODEL_TAG,
        }
        dst = src.with_suffix("").with_suffix(".picked.json")
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str))
        os.replace(tmp, dst)
        written += 1
        print(f"wrote {dst.name}: chose {idx}")
    print(f"Total picked records: {written}")


if __name__ == "__main__":
    main()
