from __future__ import annotations
import asyncio
import sys
from pathlib import Path
import typer
from app.engine import (
    ContextBuilder,
    Pipeline,
    PromptRenderer,
    TokenBudget,
)
from app.runtime.client import InferenceClient
from app.world import SeedLoader, WorldStateManager
from app.world.db import open_db


app = typer.Typer(help="Quest game CLI.")

PROMPTS = Path(__file__).parent.parent.parent / "prompts"


def _open_world(db_path: Path) -> WorldStateManager:
    conn = open_db(db_path)
    return WorldStateManager(conn)


@app.command()
def init(
    db: Path = typer.Option(..., help="Path to the quest SQLite DB (will be created)."),
    seed: Path = typer.Option(..., help="Seed JSON file."),
) -> None:
    """Initialize a new quest database from a seed JSON file."""
    sm = _open_world(db)
    payload = SeedLoader.load(seed)
    for rule in payload.rules:
        sm.add_rule(rule)
    for hook in payload.foreshadowing:
        sm.add_foreshadowing(hook)
    for pt in payload.plot_threads:
        sm.add_plot_thread(pt)
    sm.apply_delta(payload.delta, update_number=0)
    typer.echo(f"Seeded {db} with {len(payload.delta.entity_creates)} entities.")


@app.command()
def play(
    db: Path = typer.Option(..., help="Path to the quest DB."),
    server: str = typer.Option("http://127.0.0.1:8090", help="llama-server base URL."),
) -> None:
    """Play the quest — reads player actions from stdin, prints prose."""
    sm = _open_world(db)
    client = InferenceClient(base_url=server, retries=1)
    cb = ContextBuilder(sm, PromptRenderer(PROMPTS), TokenBudget())
    pipeline = Pipeline(sm, cb, client)

    typer.echo("Quest started. Type an action and press enter. Ctrl-D to quit.")
    # Determine next update number
    records = sm.list_narrative(limit=10_000)
    update_number = (max((r.update_number for r in records), default=0)) + 1

    for line in sys.stdin:
        action = line.strip()
        if not action:
            continue
        typer.echo("\n--- generating chapter ---\n", err=True)
        try:
            out = asyncio.run(pipeline.run(player_action=action, update_number=update_number))
        except Exception as e:
            typer.echo(f"[error] {e}", err=True)
            continue
        typer.echo(out.prose)
        typer.echo("\nChoices:")
        for i, c in enumerate(out.choices, 1):
            typer.echo(f"  {i}. {c}")
        typer.echo()
        update_number += 1
