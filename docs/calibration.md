# Calibration Corpus

A ground-truth-anchored harness that measures how closely our scoring
dimensions (heuristic + LLM-judged) match hand-curated expected values across
16 works of fiction (12 novels + 4 web-serial "quests").

The calibration stack lives in `app/calibration/` and is intentionally
**decoupled from the rerank scorer** in `app/engine/pipeline.py`. Do not
import rerank weights here; if a dimension overlaps, score it fresh.

## 1. Provide passages

Passages are not committed — they are copyrighted. Stage them outside the
repo in any layout that matches:

```
<passages-dir>/
  mrs_dalloway/
    p01.txt
    p02.txt
  sun_also_rises/
    p01.txt
    ...
```

Each `<passage_id>.txt` is UTF-8 plain text, typically 200–600 words — one
scene-beat. Strip footnotes and running headers.

The manifest (`data/calibration/manifest.yaml`) declares 2–3 passage slots
per work with `sha256: PENDING`. Slots also carry `expected_high` and
`expected_low` dimension lists — which dimensions the passage should
stress-test.

## 2. Hash into the manifest

```
python -m app.calibration init --passages-dir /path/to/passages
```

This walks `<passages-dir>/<work_id>/<passage_id>.txt`, sha256s each file,
and rewrites the manifest in place. Commit the resulting non-PENDING hashes;
from that point on the harness refuses to score a file that doesn't match.

## 3. Run the harness

Heuristics only (fast, no model):

```
python -m app.calibration run --passages-dir /path/to/passages
```

With LLM judges (one batched structured call per passage):

```
python -m app.calibration run \
  --passages-dir /path/to/passages \
  --server-url http://localhost:8080
```

Flags:
- `--json` emit machine-readable report
- `--threshold 0.7` correlation threshold to flag failing dimensions
- `--strict` exit non-zero if any dim fails the threshold

## 4. Read the report

The text report lists each dimension's MAE, RMSE, and Pearson *r* against
expected values, plus an overall roll-up. Dimensions with `r < threshold`
are called out individually.

## 5. Fix dimensions failing r > 0.7

When a dimension drifts below the correlation floor:

1. **Heuristic dim (`sentence_variance`, `dialogue_ratio`, `pacing`,
   `action_fidelity`):** inspect the passages scoring high/low and compare
   to expected. Tune the normalization curve in
   `app/calibration/heuristics.py`. Retain a test case in
   `tests/calibration/test_heuristics.py` for every tuning change.
2. **LLM-judged dim:** rewrite the rubric fragment in
   `prompts/scoring/dims/<dim>.j2`. Keep rubrics terse (3–5 sentences),
   anchor the 0/0.5/1.0 endpoints, and commit an exemplar passage per
   endpoint in the doc for future calibration cycles.
3. **Expected scores** (last resort): if you genuinely disagree with the
   manifest's ground truth for a work, open a review — do not silently
   change. Ground-truth drift defeats the point of the harness.

## Critic -> scalar score mapping

Several dimensions are implemented as critics returning
`list[ValidationIssue]`. Convert uniformly:

```
score = max(0.0, 1.0 - (num_errors * 0.25 + num_warnings * 0.10))
```

See `app/calibration/scorer.py` for the implementation.

## LLM judge batching

One structured call per passage scores every applicable LLM dimension. The
prompt is composed from:

- `prompts/scoring/dims/<dim>.j2` — per-dim rubric
- `prompts/scoring/batch.j2` — envelope that lists dims + passage and asks
  for a single JSON object keyed by dim name

Quest-only dims (`choice_hook_quality`, `update_self_containment`,
`choice_meaningfulness`, `world_state_legibility`) are only included when
`is_quest: true`.
