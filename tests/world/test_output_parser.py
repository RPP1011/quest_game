from __future__ import annotations
import pytest
from pydantic import BaseModel
from app.world.output_parser import OutputParser, ParseError


class Beat(BaseModel):
    scene: str
    tension_delta: int


def test_parse_json_plain():
    text = '{"a": 1, "b": [2, 3]}'
    assert OutputParser.parse_json(text) == {"a": 1, "b": [2, 3]}


def test_parse_json_stripped_markdown_fence():
    text = "```json\n{\"x\": 7}\n```"
    assert OutputParser.parse_json(text) == {"x": 7}


def test_parse_json_with_thinking_block():
    text = "<think>let me plan</think>\n{\"ok\": true}"
    assert OutputParser.parse_json(text) == {"ok": True}


def test_parse_json_embedded_in_prose():
    text = "Here's the plan:\n\n{\"scene\": \"tavern\", \"tension_delta\": 2}\n\nHope that helps."
    assert OutputParser.parse_json(text) == {"scene": "tavern", "tension_delta": 2}


def test_parse_json_with_schema_returns_typed_instance():
    text = '{"scene": "tavern", "tension_delta": 2}'
    beat = OutputParser.parse_json(text, schema=Beat)
    assert isinstance(beat, Beat)
    assert beat.scene == "tavern"


def test_parse_json_schema_violation_raises():
    text = '{"scene": "tavern"}'  # missing tension_delta
    with pytest.raises(ParseError):
        OutputParser.parse_json(text, schema=Beat)


def test_parse_json_unrepairable_raises():
    with pytest.raises(ParseError):
        OutputParser.parse_json("this has no json at all")


def test_parse_prose_strips_thinking():
    text = "<think>let me think</think>\n\nShe walked into the tavern."
    assert OutputParser.parse_prose(text) == "She walked into the tavern."


def test_parse_prose_strips_short_preamble():
    text = "Sure, here's the scene:\n\nShe walked into the tavern."
    assert OutputParser.parse_prose(text) == "She walked into the tavern."


def test_parse_prose_keeps_real_content():
    text = "She walked into the tavern.\n\nThe bartender looked up."
    assert OutputParser.parse_prose(text) == "She walked into the tavern.\n\nThe bartender looked up."
