"""Arc-scale (scene/chapter/update) scorer.

Scene-scale counterpart to the passage-scale ``BatchJudge`` in
``app.calibration.judges``. One batched structured call per scene produces
scores for all applicable arc dimensions. Quest-only dimensions are gated
behind ``is_quest``.

Output is a rater-JSON file at ``/tmp/rater_arc_<model>.json`` with shape::

    {
      "model": "<model-tag>",
      "kind": "arc",
      "dims": [...],
      "scenes": [
        {
          "work_id": "...",
          "scene_id": "s01",
          "sha256": "...",
          "is_quest": bool,
          "scores": {"tension_execution": {"score": 0.7, "rationale": "..."}, ...}
        },
        ...
      ]
    }

The scorer reads scenes from ``<scenes-dir>/<work_id>/<scene_id>.txt`` —
the layout emitted by ``tools/sample_scenes.py``. Each file may have YAML
frontmatter; we strip it before scoring and hash the body.

If the scene exceeds the model's input budget we fall back to
``app.calibration.recursive_summary`` to compress it first.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.calibration.loader import load_manifest


# Dimensions that apply to any scene (novel or quest).
ARC_COMMON_DIMS: tuple[str, ...] = (
    "tension_execution",
)

# Scene-scale dimensions that only exist for quest fiction.
ARC_QUEST_DIMS: tuple[str, ...] = (
    "choice_hook_quality",
    "update_self_containment",
    "choice_meaningfulness",
    "world_state_legibility",
)


log = logging.getLogger("arc_scorer")


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


@dataclass(frozen=True)
class ArcScore:
    score: float
    rationale: str


def arc_dims_for(is_quest: bool) -> list[str]:
    return list(ARC_COMMON_DIMS) + (list(ARC_QUEST_DIMS) if is_quest else [])


def _response_schema(dim_names: list[str]) -> dict[str, Any]:
    props = {
        d: {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        }
        for d in dim_names
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(dim_names),
        "additionalProperties": False,
    }


class ArcBatchJudge:
    """Render the arc-scale batch prompt, call the model, parse JSON -> scores."""

    def __init__(self, prompts_dir: str | Path) -> None:
        self._prompts_dir = Path(prompts_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self._prompts_dir)),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _load_rubric(self, dim: str) -> str:
        tmpl = self._env.get_template(f"scoring/arc_dims/{dim}.j2")
        return tmpl.render().strip()

    def render_prompt(
        self,
        *,
        scene: str,
        work_id: str,
        pov: str,
        is_quest: bool,
        dim_names: list[str] | None = None,
    ) -> str:
        dim_names = dim_names or arc_dims_for(is_quest)
        dims = [{"name": d, "rubric": self._load_rubric(d)} for d in dim_names]
        tmpl = self._env.get_template("scoring/batch.j2")
        # The batch template uses `passage` as the variable name; reuse it.
        return tmpl.render(
            passage=scene,
            work_id=work_id,
            pov=pov,
            is_quest=is_quest,
            dims=dims,
        )

    async def score(
        self,
        *,
        client: ChatLike,
        scene: str,
        work_id: str,
        pov: str,
        is_quest: bool,
        dim_names: list[str] | None = None,
    ) -> dict[str, ArcScore]:
        dim_names = dim_names or arc_dims_for(is_quest)
        prompt = self.render_prompt(
            scene=scene,
            work_id=work_id,
            pov=pov,
            is_quest=is_quest,
            dim_names=dim_names,
        )
        raw = await _call_model(client, prompt, _response_schema(dim_names))
        return parse_response(raw, dim_names)


async def _call_model(client: ChatLike, prompt: str, schema: dict[str, Any]) -> str:
    try:
        from app.runtime.client import ChatMessage  # type: ignore
        msg = ChatMessage(role="user", content=prompt)
    except Exception:  # pragma: no cover
        msg = {"role": "user", "content": prompt}

    chat_structured = getattr(client, "chat_structured", None)
    if chat_structured is not None:
        return await chat_structured(
            messages=[msg],
            json_schema=schema,
            schema_name="ArcSceneScores",
            temperature=0.2,
            max_tokens=2500,
            thinking=False,
        )
    return await client.chat(
        messages=[msg],
        temperature=0.2,
        max_tokens=2500,
        thinking=False,
    )


def parse_response(raw: str, dim_names: list[str]) -> dict[str, ArcScore]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in arc-judge response: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1])
    out: dict[str, ArcScore] = {}
    for d in dim_names:
        if d not in data:
            raise ValueError(f"arc-judge response missing dimension {d!r}")
        entry = data[d]
        score = max(0.0, min(1.0, float(entry["score"])))
        out[d] = ArcScore(score=score, rationale=str(entry.get("rationale", "")))
    return out


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n+", re.DOTALL)


def strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_scene(path: Path) -> str:
    return strip_frontmatter(path.read_text(encoding="utf-8"))


# Rough token budget — conservative; LFM launches with --ctx-size 32768, leave
# room for prompt overhead + output. If the scene exceeds this, summarize.
DEFAULT_MAX_SCENE_CHARS = 90_000


async def _maybe_summarize(
    scene: str,
    *,
    client: ChatLike | None,
    max_chars: int,
) -> str:
    if len(scene) <= max_chars or client is None:
        return scene
    # Lazy import so tests that don't need summarization can skip.
    from app.calibration.recursive_summary import recursive_summarize
    log.info("scene %d chars > %d; recursively summarizing", len(scene), max_chars)
    return await recursive_summarize(
        scene,
        client=client,
        target_chars=max_chars // 2,
    )


async def score_scenes(
    *,
    manifest_path: Path,
    scenes_dir: Path,
    client: ChatLike,
    model_tag: str,
    out_path: Path,
    max_scene_chars: int = DEFAULT_MAX_SCENE_CHARS,
    works_filter: set[str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    results: list[dict[str, Any]] = []
    judge = ArcBatchJudge(Path("prompts"))

    for work in manifest.works:
        if works_filter and work.id not in works_filter:
            continue
        work_dir = scenes_dir / work.id
        if not work_dir.is_dir():
            log.info("skip %s: no scenes dir", work.id)
            continue
        for scene_path in sorted(work_dir.glob("s*.txt")):
            scene_id = scene_path.stem
            try:
                scene = _load_scene(scene_path)
            except Exception as exc:
                log.warning("skip %s/%s: %s", work.id, scene_id, exc)
                continue
            scene = await _maybe_summarize(
                scene, client=client, max_chars=max_scene_chars,
            )
            dim_names = arc_dims_for(work.is_quest)
            try:
                scored = await judge.score(
                    client=client,
                    scene=scene,
                    work_id=work.id,
                    pov=work.pov,
                    is_quest=work.is_quest,
                )
            except Exception as exc:
                log.error("score failed %s/%s: %s", work.id, scene_id, exc)
                continue
            results.append({
                "work_id": work.id,
                "scene_id": scene_id,
                "sha256": sha256_text(scene),
                "is_quest": work.is_quest,
                "scores": {
                    d: {"score": s.score, "rationale": s.rationale}
                    for d, s in scored.items()
                },
            })
            log.info("scored %s/%s (%d dims)", work.id, scene_id, len(dim_names))

    report = {
        "model": model_tag,
        "kind": "arc",
        "dims": sorted(set(ARC_COMMON_DIMS + ARC_QUEST_DIMS)),
        "scenes": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _default_out_path(model_tag: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_tag)
    return Path(f"/tmp/rater_arc_{safe}.json")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="arc_scorer")
    ap.add_argument("--manifest", default="data/calibration/scenes_manifest.yaml")
    ap.add_argument("--scenes-dir", default="data/calibration/scenes")
    ap.add_argument("--server-url", required=True,
                    help="llama-server URL (launch with --ctx-size 32768)")
    ap.add_argument("--model", required=True, help="model tag for output filename")
    ap.add_argument("--out", default=None)
    ap.add_argument("--works", default=None,
                    help="comma-separated work ids to restrict scoring to")
    ap.add_argument("--max-scene-chars", type=int, default=DEFAULT_MAX_SCENE_CHARS)
    args = ap.parse_args(argv)

    from app.runtime.client import InferenceClient  # lazy
    client = InferenceClient(base_url=args.server_url, timeout=600.0, retries=0)

    out = Path(args.out) if args.out else _default_out_path(args.model)
    works_filter = set(args.works.split(",")) if args.works else None

    report = asyncio.run(score_scenes(
        manifest_path=Path(args.manifest),
        scenes_dir=Path(args.scenes_dir),
        client=client,
        model_tag=args.model,
        out_path=out,
        max_scene_chars=args.max_scene_chars,
        works_filter=works_filter,
    ))
    print(f"wrote {out} ({len(report['scenes'])} scenes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
