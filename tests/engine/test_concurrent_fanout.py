"""Day 3: Concurrent N-candidate dispatch.

The N inference calls in ``Pipeline._generate_scene_candidates`` must be
dispatched concurrently (one ``asyncio.gather`` over N ``client.chat``
coroutines) rather than sequentially. This lets batching backends like
vllm coalesce the calls into a single forward pass, which is the whole
point of Generate-N.

The tests stub the client with an ``asyncio.Event`` gate: every in-flight
call records its start time and waits on the gate before returning. If
dispatch is sequential, only one call is ever in-flight at once and the
test hangs. If dispatch is concurrent, all N reach the gate before any
resolve — the gate releases them, and the max-min start-time delta is
well under the per-call wall time.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.engine import ContextBuilder, PromptRenderer, TokenBudget
from app.engine.pipeline import Pipeline
from app.engine.trace import PipelineTrace
from app.planning.schemas import CraftBrief, CraftPlan, CraftScenePlan
from app.world import StateDelta, WorldStateManager
from app.world.db import open_db


PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class GatedRecordingClient:
    """Records call start times; waits on an external event before returning.

    If the pipeline dispatches sequentially, only one call gets into
    ``chat`` at a time and the second won't start until the first has
    released — but our fixture only releases the gate after ``expected_n``
    calls are in-flight, so a sequential caller would deadlock. A short
    timeout on ``event.wait()`` converts that deadlock into a visible
    error, keeping the test fast.
    """

    def __init__(self, prose_per_call: list[str], expected_n: int) -> None:
        self._prose = list(prose_per_call)
        self._gate = asyncio.Event()
        self._started = asyncio.Semaphore(0)
        self._expected_n = expected_n
        self.start_times: list[float] = []
        self.calls: list[dict[str, Any]] = []

    async def chat_structured(self, *, messages, json_schema, schema_name, **kw) -> str:
        raise AssertionError("chat_structured should not be called")

    async def chat(self, *, messages, **kw) -> str:
        self.start_times.append(time.perf_counter())
        self.calls.append({"temperature": kw.get("temperature"),
                           "seed": kw.get("seed")})
        # Signal that this call has reached the gate.
        self._started.release()
        # Wait for the gate — test fixture releases it once all N are in-flight.
        try:
            await asyncio.wait_for(self._gate.wait(), timeout=2.0)
        except asyncio.TimeoutError as e:
            raise AssertionError(
                f"concurrent dispatch timeout: only {len(self.start_times)} "
                f"of {self._expected_n} calls reached the gate"
            ) from e
        # Pop next candidate's prose (calls may arrive out of order but we
        # don't need to pair them back — any pairing is fine for the
        # concurrency assertion).
        return self._prose.pop(0) if self._prose else "you stand still."

    async def release_when_all_in_flight(self) -> None:
        """Release the gate once all N calls have reached it."""
        for _ in range(self._expected_n):
            await self._started.acquire()
        self._gate.set()


def _make_world(tmp_path):
    conn = open_db(tmp_path / "w.db")
    sm = WorldStateManager(conn)
    sm.apply_delta(StateDelta(), update_number=1)
    return sm, conn


def _make_pipeline(world, client, **kw):
    cb = ContextBuilder(world, PromptRenderer(PROMPTS), TokenBudget())
    return Pipeline(world, cb, client, **kw)


def _trace() -> PipelineTrace:
    return PipelineTrace(trace_id=uuid.uuid4().hex, trigger="test")


def _one_scene_plan() -> CraftPlan:
    return CraftPlan(
        scenes=[CraftScenePlan(scene_id=1)],
        briefs=[CraftBrief(scene_id=1, brief="Write something.")],
    )


# ---------------------------------------------------------------------------


async def test_concurrent_dispatch_all_calls_in_flight_simultaneously(tmp_path):
    """With N=5, all five ``client.chat`` coroutines must enter their await
    before any one resolves. We verify this by gating each call on an
    asyncio.Event and releasing only after the fifth call arrives.
    """
    world, conn = _make_world(tmp_path)
    try:
        prose = [f"you take action {i}." for i in range(5)]
        client = GatedRecordingClient(prose, expected_n=5)
        pipeline = _make_pipeline(world, client, n_candidates=5)
        trace = _trace()

        # Fire the pipeline write + the gate release concurrently. If
        # dispatch is sequential, the gate release fires but only one
        # ``chat`` ever reaches the gate, and the wait_for(2s) inside
        # ``chat`` raises AssertionError — the test fails loudly.
        write_task = asyncio.create_task(
            pipeline._run_write(
                trace, _one_scene_plan(), player_action="go"
            )
        )
        gate_task = asyncio.create_task(client.release_when_all_in_flight())
        await asyncio.wait_for(
            asyncio.gather(write_task, gate_task), timeout=5.0
        )

        # Every call recorded a start time.
        assert len(client.start_times) == 5
        # Max-min start-time spread is under a small threshold (all five
        # were dispatched essentially simultaneously).
        spread = max(client.start_times) - min(client.start_times)
        assert spread < 0.25, (
            f"N candidates did not dispatch concurrently; "
            f"start-time spread = {spread:.3f}s"
        )
        # Each call got a unique seed (jitter preserved).
        seeds = sorted(c["seed"] for c in client.calls if c["seed"] is not None)
        assert len(set(seeds)) == 5
    finally:
        conn.close()


async def test_sequential_client_would_hang_concurrent_does_not(tmp_path):
    """Sanity check: a slow client that sleeps for T per call is an easy
    way to distinguish sequential from concurrent — total wall-clock for
    N calls should be ~T (concurrent), not ~N*T (sequential).
    """
    world, conn = _make_world(tmp_path)
    try:
        per_call_sleep = 0.10

        class SlowClient:
            def __init__(self) -> None:
                self.calls = 0

            async def chat_structured(self, **kw) -> str:
                raise AssertionError()

            async def chat(self, *, messages, **kw) -> str:
                self.calls += 1
                await asyncio.sleep(per_call_sleep)
                return "you hold your ground."

        client = SlowClient()
        pipeline = _make_pipeline(world, client, n_candidates=5)
        trace = _trace()
        t0 = time.perf_counter()
        await pipeline._run_write(
            trace, _one_scene_plan(), player_action="x"
        )
        elapsed = time.perf_counter() - t0

        assert client.calls == 5
        # Sequential would be ~0.50s; concurrent should be <=~0.20s
        # (one sleep + overhead). Set the bound at 0.30s to stay robust on
        # slow CI. Anything above that means we're no longer concurrent.
        assert elapsed < 0.30, (
            f"N=5 dispatch took {elapsed:.3f}s; "
            f"expected concurrent ~{per_call_sleep:.2f}s"
        )
    finally:
        conn.close()


async def test_concurrent_dispatch_preserves_candidate_ordering(tmp_path):
    """Even when calls finish out of order, the trace records each
    candidate at its dispatched index (0..N-1) — ``asyncio.gather``
    preserves input ordering in its return value."""
    world, conn = _make_world(tmp_path)
    try:
        # Each call sleeps a different amount so completion order is
        # the reverse of dispatch order. Dispatch-order assignment
        # must still appear at candidate_index 0..2.
        call_count = [0]
        sleeps = [0.15, 0.05, 0.01]

        class ReversedSleepClient:
            async def chat_structured(self, **kw) -> str:
                raise AssertionError()

            async def chat(self, *, messages, **kw) -> str:
                idx = call_count[0]
                call_count[0] += 1
                await asyncio.sleep(sleeps[idx])
                return f"you respond number {idx}."

        client = ReversedSleepClient()
        pipeline = _make_pipeline(world, client, n_candidates=3)
        trace = _trace()
        await pipeline._run_write(
            trace, _one_scene_plan(), player_action="x"
        )

        writes = [s for s in trace.stages if s.stage_name == "write"]
        # Three write stages, indexed 0..2 by dispatch order.
        assert [w.detail["candidate_index"] for w in writes] == [0, 1, 2]
    finally:
        conn.close()
