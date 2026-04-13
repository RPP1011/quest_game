"""Voter-rollout experiment.

Given a quest state and a set of voter personas, ask each voter to propose a
write-in action. Fork the quest, run one advance per proposal, and measure how
the story diverges across branches.

This is an evaluation tool, not a gameplay feature. It answers: "does the
input action actually steer the story, or does the pipeline produce similar
prose regardless?"
"""
from __future__ import annotations
import asyncio
import json
import shutil
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.engine import ContextBuilder, Pipeline, PromptRenderer, TokenBudget, TraceStore
from app.runtime.client import ChatMessage, InferenceClient
from app.world import WorldStateManager
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


@dataclass
class VoterPersona:
    """A simulated forum reader archetype."""
    id: str
    name: str
    motive: str   # what this reader wants to see the protagonist do / optimize for
    register: str  # how they phrase their proposals


DEFAULT_PERSONAS: list[VoterPersona] = [
    VoterPersona(
        id="tactician",
        name="The Tactician",
        motive="Advance the protagonist's material power and political standing. Prefers concrete actions with visible consequences: meet the captain, audit the treasury, visit the border.",
        register="dry, bullet-point-adjacent prose. Imperatives.",
    ),
    VoterPersona(
        id="diplomat",
        name="The Diplomat",
        motive="Build relationships. Pick actions that deepen ties with NPCs, especially Helga and Marek. Prefers conversation, confession, shared grief.",
        register="careful, interior. Tends to specify what the protagonist says or what they notice.",
    ),
    VoterPersona(
        id="dramatist",
        name="The Dramatist",
        motive="Escalate. Push toward a confrontation or a reveal. Likes actions that change the status quo — break a rule, demand a truth, visit the crypt.",
        register="punchy, imagistic. Short sentences. Strong verbs.",
    ),
    VoterPersona(
        id="doubter",
        name="The Doubter",
        motive="Protect the protagonist from themselves. Slow down. Pick actions that buy information or time before committing: read the letters, walk the walls alone, ask who profits from which outcome.",
        register="hedged, probing. Often phrased as questions or mild actions.",
    ),
]


@dataclass
class VoterProposal:
    persona_id: str
    action: str           # the write-in the voter proposes
    rationale: str        # why they want this


@dataclass
class RolloutResult:
    persona_id: str
    action: str
    prose: str
    trace_id: str
    prose_length: int
    similarity_to_baseline: float  # 0..1, 1 = identical to first branch
    similarity_to_all_mean: float  # average similarity to every other branch


async def propose_actions(
    personas: list[VoterPersona],
    world: WorldStateManager,
    client: InferenceClient,
) -> list[VoterProposal]:
    """Ask each persona for a write-in action given current world state."""
    # Build a terse current-scene summary.
    chars = ", ".join(
        f"{e.name}"
        for e in world.list_entities()
        if e.entity_type.value == "character"
    )
    threads = "; ".join(pt.name for pt in world.list_plot_threads()[:3])
    recent = world.list_narrative(limit=10_000)
    last_prose = recent[-1].raw_text if recent else "(no scenes yet)"

    proposals: list[VoterProposal] = []
    for p in personas:
        sys_msg = (
            "You roleplay a forum-quest reader. Propose ONE write-in action for "
            "the protagonist. Output only JSON: "
            '{"action": "<one short imperative sentence>", "rationale": "<one sentence>"}'
        )
        user_msg = (
            f"You are: {p.name}. Motive: {p.motive}. "
            f"Write in this register: {p.register}.\n\n"
            f"Active characters: {chars}\n"
            f"Plot threads: {threads}\n"
            f"Last scene:\n{last_prose[-600:]}\n\n"
            "Propose your write-in now. JSON only."
        )
        try:
            raw = await client.chat(
                messages=[
                    ChatMessage(role="system", content=sys_msg),
                    ChatMessage(role="user", content=user_msg),
                ],
                temperature=0.8,
                max_tokens=200,
                thinking=False,
            )
            # Tolerant JSON parse
            start = raw.find("{")
            end = raw.rfind("}")
            parsed = json.loads(raw[start:end + 1])
            proposals.append(VoterProposal(
                persona_id=p.id,
                action=parsed.get("action", "").strip() or f"[{p.name} abstains]",
                rationale=parsed.get("rationale", "").strip(),
            ))
        except Exception as e:
            proposals.append(VoterProposal(
                persona_id=p.id,
                action=f"[voter error: {type(e).__name__}]",
                rationale="",
            ))
    return proposals


async def rollout(
    proposal: VoterProposal,
    source_db: Path,
    traces_dir: Path,
    client: InferenceClient,
    quest_config: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Fork the DB, advance one turn with the proposed action, return (prose, trace_id)."""
    fork_db = source_db.parent / f"{source_db.stem}__{proposal.persona_id}.db"
    shutil.copy2(source_db, fork_db)
    fork_traces = traces_dir.parent / f"{traces_dir.name}__{proposal.persona_id}"
    fork_traces.mkdir(parents=True, exist_ok=True)

    conn = open_db(fork_db)
    world = WorldStateManager(conn)
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    store = TraceStore(fork_traces)
    pipeline = Pipeline(world, cb, client)

    records = world.list_narrative(limit=10_000)
    update_number = (max((r.update_number for r in records), default=0)) + 1
    out = await pipeline.run(player_action=proposal.action, update_number=update_number)
    store.save(out.trace)
    conn.close()
    return out.prose, out.trace.trace_id


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_rollouts(raw_results: list[tuple[VoterProposal, str, str]]) -> list[RolloutResult]:
    """Attach divergence metrics."""
    if not raw_results:
        return []
    proses = [r[1] for r in raw_results]
    baseline = proses[0]
    out: list[RolloutResult] = []
    for i, (prop, prose, tid) in enumerate(raw_results):
        others = [p for j, p in enumerate(proses) if j != i]
        mean_sim = (
            sum(_similarity(prose, p) for p in others) / len(others) if others else 1.0
        )
        out.append(RolloutResult(
            persona_id=prop.persona_id,
            action=prop.action,
            prose=prose,
            trace_id=tid,
            prose_length=len(prose),
            similarity_to_baseline=_similarity(prose, baseline),
            similarity_to_all_mean=mean_sim,
        ))
    return out


async def run_experiment(
    source_db: Path,
    traces_dir: Path,
    server_url: str,
    personas: list[VoterPersona] | None = None,
    quest_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full experiment: propose actions, rollout each, score, return a report dict."""
    personas = personas or DEFAULT_PERSONAS
    client = InferenceClient(base_url=server_url, timeout=300.0, retries=0)

    # Propose actions against the source-state world.
    src_conn = open_db(source_db)
    src_world = WorldStateManager(src_conn)
    proposals = await propose_actions(personas, src_world, client)
    src_conn.close()

    # Roll out each proposal on its own fork.
    raw_results: list[tuple[VoterProposal, str, str]] = []
    for prop in proposals:
        prose, tid = await rollout(prop, source_db, traces_dir, client, quest_config)
        raw_results.append((prop, prose, tid))

    scored = score_rollouts(raw_results)
    return {
        "proposals": [asdict(p) for p in proposals],
        "rollouts": [asdict(r) for r in scored],
    }


def main() -> None:
    """CLI entry: `uv run python -m app.experiments.voter_rollout <db> <traces_dir> <server_url>`."""
    import sys
    if len(sys.argv) < 4:
        print("usage: voter_rollout.py <db_path> <traces_dir> <server_url>")
        sys.exit(1)
    db = Path(sys.argv[1])
    traces = Path(sys.argv[2])
    url = sys.argv[3]
    report = asyncio.run(run_experiment(db, traces, url))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
