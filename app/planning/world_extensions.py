from __future__ import annotations

from pydantic import BaseModel


class Theme(BaseModel):
    id: str
    name: str
    description: str
    stance: str | None = None   # quest's current stance toward the theme


class MotifDef(BaseModel):
    id: str
    name: str
    description: str
    recurrences: list[int] = []  # update numbers where it appeared
