from app.engine.token_budget import TokenBudget, estimate_tokens


def test_estimate_tokens_rough():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 40) == 10


def test_budget_defaults_sum_reasonable():
    b = TokenBudget()
    assert b.total == 200_000
    # sum of sections + headroom + margin shouldn't exceed total
    assert b.world_state + b.narrative_history + b.system_prompt + b.style_config \
           + b.prior_stage_outputs + b.generation_headroom + b.safety_margin <= b.total


def test_budget_remaining():
    b = TokenBudget()
    used = {"world_state": 5_000, "narrative_history": 10_000}
    rem = b.remaining(used)
    assert rem == b.total - 15_000 - b.safety_margin


def test_budget_fits_positive():
    b = TokenBudget(total=1000, safety_margin=100)
    assert b.fits({"x": 500})
    assert not b.fits({"x": 950})
