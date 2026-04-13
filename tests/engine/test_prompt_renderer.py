# tests/engine/test_prompt_renderer.py
from pathlib import Path
import pytest
from app.engine.prompt_renderer import PromptRenderer


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


def test_renders_existing_template():
    r = PromptRenderer(PROMPTS)
    out = r.render("stages/plan/system.j2", {})
    assert "narrative planner" in out.lower()


def test_renders_with_context():
    r = PromptRenderer(PROMPTS)
    out = r.render("stages/plan/user.j2", {
        "entities": [],
        "plot_threads": [],
        "recent_summaries": [],
        "player_action": "Enter the tavern.",
    })
    assert "Enter the tavern." in out


def test_missing_variable_raises():
    r = PromptRenderer(PROMPTS)
    with pytest.raises(Exception):  # jinja2.UndefinedError
        r.render("stages/plan/user.j2", {})  # missing player_action


def test_missing_template_raises():
    r = PromptRenderer(PROMPTS)
    with pytest.raises(Exception):
        r.render("stages/nonexistent/system.j2", {})
