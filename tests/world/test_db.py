from __future__ import annotations
import sqlite3
from pathlib import Path
from app.world.db import open_db


def test_open_creates_file(tmp_path: Path):
    p = tmp_path / "q.db"
    assert not p.exists()
    conn = open_db(p)
    assert p.exists()
    conn.close()


def test_all_tables_created(db):
    cur = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cur.fetchall()}
    assert {
        "entities",
        "relationships",
        "world_rules",
        "timeline",
        "narrative",
        "foreshadowing",
        "plot_threads",
    } <= tables


def test_foreign_keys_enabled(db):
    cur = db.execute("PRAGMA foreign_keys")
    assert cur.fetchone()[0] == 1


def test_row_factory_returns_rows(db):
    db.execute("INSERT INTO entities(id,entity_type,name,data,status) VALUES(?,?,?,?,?)",
               ("x", "item", "X", "{}", "active"))
    row = db.execute("SELECT id, name FROM entities WHERE id='x'").fetchone()
    assert row["id"] == "x"
    assert row["name"] == "X"


def test_open_twice_is_idempotent(tmp_path: Path):
    p = tmp_path / "q.db"
    open_db(p).close()
    # Should not raise on re-open
    conn = open_db(p)
    conn.close()
