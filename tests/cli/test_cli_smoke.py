import json
from pathlib import Path
from typer.testing import CliRunner
from app.cli.play import app


def test_cli_help():
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "play" in r.stdout.lower()


def test_cli_init_from_seed(tmp_path: Path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({
        "entities": [
            {"id": "alice", "entity_type": "character", "name": "Alice"},
        ],
    }))
    db = tmp_path / "q.db"
    runner = CliRunner()
    r = runner.invoke(app, ["init", "--db", str(db), "--seed", str(seed)])
    assert r.exit_code == 0, r.stdout
    assert db.exists()
    # Verify the seed loaded
    from app.world.db import open_db
    from app.world.state_manager import WorldStateManager
    conn = open_db(db)
    sm = WorldStateManager(conn)
    assert [e.id for e in sm.list_entities()] == ["alice"]
    conn.close()
