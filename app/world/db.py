from __future__ import annotations
import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_referenced_update INTEGER,
    created_at_update INTEGER,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relationships (
    source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel_type TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    established_at_update INTEGER,
    PRIMARY KEY (source_id, target_id, rel_type)
);

CREATE TABLE IF NOT EXISTS world_rules (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    constraints TEXT NOT NULL DEFAULT '{}',
    established_at_update INTEGER
);

CREATE TABLE IF NOT EXISTS timeline (
    update_number INTEGER NOT NULL,
    event_index INTEGER NOT NULL,
    description TEXT NOT NULL,
    involved_entities TEXT NOT NULL DEFAULT '[]',
    causal_links TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (update_number, event_index)
);

CREATE TABLE IF NOT EXISTS narrative (
    update_number INTEGER PRIMARY KEY,
    raw_text TEXT NOT NULL,
    summary TEXT,
    chapter_id INTEGER,
    state_diff TEXT NOT NULL DEFAULT '{}',
    player_action TEXT,
    pipeline_trace_id TEXT
);

CREATE TABLE IF NOT EXISTS foreshadowing (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    planted_at_update INTEGER NOT NULL,
    payoff_target TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'planted',
    paid_off_at_update INTEGER,
    refs TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS plot_threads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    involved_entities TEXT NOT NULL DEFAULT '[]',
    arc_position TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5
);

CREATE TABLE IF NOT EXISTS themes (
    id TEXT NOT NULL,
    quest_id TEXT NOT NULL,
    proposition TEXT NOT NULL,
    stance TEXT NOT NULL DEFAULT 'exploring',
    motif_ids TEXT NOT NULL DEFAULT '[]',
    thesis_character_ids TEXT NOT NULL DEFAULT '[]',
    key_scenes TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (id, quest_id)
);

CREATE TABLE IF NOT EXISTS reader_state (
    quest_id TEXT PRIMARY KEY,
    known_fact_ids TEXT NOT NULL DEFAULT '[]',
    open_questions TEXT NOT NULL DEFAULT '[]',
    expectations TEXT NOT NULL DEFAULT '[]',
    attachment_levels TEXT NOT NULL DEFAULT '{}',
    current_emotional_valence REAL NOT NULL DEFAULT 0.0,
    updates_since_major_event INTEGER NOT NULL DEFAULT 0,
    updates_since_revelation INTEGER NOT NULL DEFAULT 0,
    updates_since_emotional_peak INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS arcs (
    quest_id TEXT NOT NULL,
    arc_id TEXT NOT NULL,
    structure_id TEXT NOT NULL,
    scale TEXT NOT NULL,
    current_phase_index INTEGER NOT NULL DEFAULT 0,
    phase_progress REAL NOT NULL DEFAULT 0.0,
    tension_observed TEXT NOT NULL DEFAULT '[]',
    last_directive TEXT,
    PRIMARY KEY (quest_id, arc_id)
);
"""


def open_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
