from __future__ import annotations
from pathlib import Path
import pytest
from app.world.db import open_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "quest.db"


@pytest.fixture
def db(db_path: Path):
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()
