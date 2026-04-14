from app.calibration.heuristics import (
    action_fidelity,
    clip01,
    dialogue_ratio,
    pacing,
    run_heuristics,
    sentence_variance,
)


def test_clip01_bounds():
    assert clip01(-1) == 0.0
    assert clip01(2) == 1.0
    assert clip01(0.5) == 0.5
    assert clip01(float("nan")) == 0.0


def test_sentence_variance_uniform_is_zero():
    # Five sentences of identical length.
    text = ("One two three four five. " * 5).strip()
    assert sentence_variance(text) == 0.0


def test_sentence_variance_high_with_mix():
    text = (
        "Yes. "
        "The slow, sonorous bells of Westminster tolled across the damp park, "
        "and she, pausing a moment at the kerb, remembered Peter. "
        "No."
    )
    assert sentence_variance(text) > 0.3


def test_dialogue_ratio_all_quotes_near_one():
    text = '"hello world this is a long line of dialogue that dominates."'
    r = dialogue_ratio(text)
    assert r > 0.8


def test_dialogue_ratio_no_quotes_is_zero():
    assert dialogue_ratio("Nothing quoted here at all.") == 0.0


def test_pacing_long_sentences_low():
    # One long sentence, many words => low density.
    text = " ".join(["word"] * 100) + "."
    assert pacing(text) < 0.35


def test_pacing_short_sentences_high():
    text = ". ".join(["He ran"] * 10) + "."
    # ~10 sentences / ~20 words => density 50/100 => clipped high.
    assert pacing(text) > 0.8


def test_action_fidelity_full_overlap():
    action = "visit the crypt"
    passage = "She decided to visit the crypt at dawn."
    assert action_fidelity(passage, action) == 1.0


def test_action_fidelity_no_overlap():
    action = "visit the crypt"
    passage = "He drank coffee silently."
    assert action_fidelity(passage, action) == 0.0


def test_action_fidelity_empty_action():
    assert action_fidelity("anything", "") == 0.0


def test_run_heuristics_novel_skips_action_fidelity():
    out = run_heuristics("A sentence. Another sentence.", is_quest=False)
    assert "action_fidelity" not in out
    assert set(out) == {"sentence_variance", "dialogue_ratio", "pacing"}


def test_run_heuristics_quest_includes_action_fidelity():
    out = run_heuristics(
        "He opened the door.",
        is_quest=True,
        player_action="open the door",
    )
    assert "action_fidelity" in out
    assert out["action_fidelity"] > 0.0
