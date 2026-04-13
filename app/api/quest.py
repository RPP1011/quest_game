from __future__ import annotations
from pydantic import BaseModel


class Choice(BaseModel):
    title: str
    description: str = ""
    tags: list[str] = []


class QuestSummary(BaseModel):
    id: str                      # basename of the .db file without extension
    path: str                    # absolute path to the .db file
    chapter_count: int
    last_action: str | None = None


class ChapterSummary(BaseModel):
    update_number: int
    player_action: str | None
    prose: str
    trace_id: str | None
    choices: list[Choice] = []


class AdvanceRequest(BaseModel):
    action: str


class AdvanceResponse(BaseModel):
    update_number: int
    prose: str
    choices: list[Choice]
    trace_id: str
    outcome: str


class SceneContext(BaseModel):
    location: str | None
    present_characters: list[str]
    plot_threads: list[str]
    recent_prose_tail: str


class TraceSummary(BaseModel):
    trace_id: str
    trigger: str
    outcome: str
    stages: list[str]
    total_latency_ms: int
