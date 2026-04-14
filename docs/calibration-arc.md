# Arc-scale Calibration

Scene/chapter/update-scale counterpart of `docs/calibration.md`. Where the
passage-scale pipeline scores 200–600-word snippets against per-paragraph
craft dims, the arc pipeline scores 2000–4000-word scenes against dimensions
that only make sense over a full arc.

## Scope

Arc-only dimensions (not scored by the passage pipeline):

- `tension_execution` — stakes arc across the scene
- `choice_hook_quality` (quest-only) — decision tension at scene close
- `update_self_containment` (quest-only) — scene stands on its own
- `choice_meaningfulness` (quest-only) — options diverge in consequence
- `world_state_legibility` (quest-only) — salient state recoverable

Quest-only dims are gated behind `is_quest` in the scenes manifest and in
`app.calibration.arc_scorer.arc_dims_for`.

## 1. Sample scenes

```
uv run python tools/sample_scenes.py \
    --manifest data/calibration/scenes_manifest.yaml \
    --raw-root data/calibration/raw \
    --scenes-dir data/calibration/scenes
```

For each work the sampler emits `s01..s05` under
`data/calibration/scenes/<work_id>/`. Each file is 2000–4000 words with YAML
frontmatter (work, author, kind=scene, word_count, source_chapters).
Sampling is deterministic per `work_id`; rerunning produces identical bytes.

### Scene-break detection

Three detectors, ordered by strength: explicit dividers (`***`, `---`,
`###`, `~~~`), `Scene N`/`SCENE N` headers, paragraph gaps (3+ newlines).
The sampler grows a chapter forward until it clears the 2000-word floor,
then trims back to the strongest scene break under the 4000-word ceiling.
If no break qualifies, it falls back to a paragraph boundary near the
target size.

## 2. Hash + score

Hashing into the manifest reuses `app.calibration.loader.init_passage_hashes`
against the scenes manifest (the passage-id field doubles as scene-id).

Scoring:

```
# Start llama-server with enough context for full scenes.
llama-server -m <model.gguf> --ctx-size 32768 --port 8080

uv run python -m app.calibration.arc_scorer \
    --manifest data/calibration/scenes_manifest.yaml \
    --scenes-dir data/calibration/scenes \
    --server-url http://localhost:8080 \
    --model lfm-q4_k_m
```

Output goes to `/tmp/rater_arc_<model>.json` in the shape:

```
{
  "model": "<tag>",
  "kind": "arc",
  "dims": [...],
  "scenes": [
    {"work_id": ..., "scene_id": "s01", "sha256": ..., "is_quest": bool,
     "scores": {"<dim>": {"score": float, "rationale": str}, ...}},
    ...
  ]
}
```

One structured call per scene covers every applicable arc dim in a single
batched response. Quest-only dims are omitted for novels.

## Recursive summarization (forward-compat)

`app.calibration.recursive_summary.recursive_summarize` compresses scenes
that exceed the model's input budget via a MapReduce: overlapping windows
(default 16k chars, 800-char overlap) → per-window summary → concat →
recurse until under `target_chars`. The arc scorer only invokes it when a
scene exceeds `--max-scene-chars` (default 90k, i.e. roughly 32k tokens of
raw input). Most calibration scenes sit well under this and pass through
unchanged.

## Status

- Scene sampler: complete. Determinism + word-count tests green.
- Arc-dim prompts: all 5 rubrics in `prompts/scoring/arc_dims/`.
- Arc scorer: complete. Emits rater-JSON identical in spirit to the passage
  scorer output.
- Scenes manifest: seeded with 12 works (the ones with passage data in
  `manifest.yaml`). `expected_high`/`expected_low` are first-pass guesses —
  not Claude-labelled yet. Labelling is tracked as its own task (TBD); this
  subagent does not label.
- `sha256: PENDING` everywhere until `tools/sample_scenes.py` is run against
  the real corpora.
