# Day 5 — Writer LoRA v1

**Date**: 2026-04-13
**Branch**: `worktree-agent-a9bd04b8`
**Artifacts**:
- Adapter: `data/sft/lora_writer_v1/` (gitignored under `data/sft/`)
- Train/test split: `data/sft/train.jsonl` (9 rows), `data/sft/test.jsonl` (2)
- Claude-editor picks: `data/sft/demo/demo/*.picked.json` (11 sidecars)
- Training log: `data/sft/lora_writer_v1/trainer_log.json`
- A/B artefact: `data/sft/ab_day5.json`

## Step 1 — Claude-editor picks (11 records)

Each `data/sft/demo/demo/*.json` SFT record has 8 candidates and a scorer-
picked `winner_index`. I read the craft brief and every candidate and
picked on prose quality using the Day-4 rubric (FIS > voice consistency >
sensory grounding > cliché absence > POV adherence). Encoded in
`tools/sft/_claude_picks.py`; executed once to emit `*.picked.json`
sidecars. `model` tag: `claude-opus-4-6-inline-editor`.

### Claude vs. scorer agreement

| File | Scorer | Claude | Agree? |
|------|-------:|-------:|:------:|
| u10_s8 | 3 | 3 | yes |
| u10_s9 | 0 | 0 | yes |
| u1_s42 | 0 | 0 | yes |
| u2_s42 | 0 | 2 | no |
| u3_s42 | 1 | 2 | no |
| u4_s42 | 4 | 6 | no |
| u5_s42 | 4 | 0 | no |
| u6_s42 | 4 | 2 | no |
| u7_s42 | 2 | 2 | yes |
| u8_s42 | 0 | 1 | no |
| u9_s42 | 2 | 2 | yes |

5/11 agreement. **Pattern**: scorer frequently picks candidate 4, which
is the "You poured the traditional white again, the steam still curling
into the morning air" opener that appears **verbatim in 6 of 11
records** — a clear overfitting / memorization artefact of the base
model, not good prose. Scorer weights reward that opener because it hits
several sentence-shape heuristics (varied clause lengths, sensory
ratios). Claude consistently down-ranks it as a canned phrase.

### Honest notes on the corpus

All 11 records are bad. The base model (LFM2.5-1.2B-Instruct, no
retrieval, no reranking at generation time) produces prose riddled with:

- **Foreign-token leakage**: German, Japanese, Slovenian, Pokémon-
  franchise debris embedded mid-sentence (`Slovenia`, `könnten`,
  `ophyll`, `Schritt`, `KEILING`, `bread einige`, `Pokémon`).
- **POV drift**: scenes flip between "you" and "I/my" mid-paragraph.
- **Meta-echo tails**: candidates often end with a literal echo of the
  brief's outcome sentence, e.g. `"The outcome: ..."` or
  `"The player's decision to reveal the cargo's location sets the stage
  for a high-stakes confrontation."`
- **Shared cliché pool**: "heart pounded", "weight of the moment",
  "breath caught", "thick with unspoken" — same phrases repeat across
  unrelated records.

The picks are therefore "least-bad of 8" much of the time. That matters
for downstream LoRA quality: we're teaching the model to prefer
slightly-less-broken prose, not good prose.

## Step 2 — Train/test split

```
uv run python -m tools.sft.build_train \
    --root data/sft/demo --quest demo \
    --out-train data/sft/train.jsonl \
    --out-test  data/sft/test.jsonl \
    --test-ratio 0.18 --seed 7
```

→ 9 train rows, 2 test rows. The row format is
`{messages: [system, user, assistant]}` matching the writer prompt used
at generation time; `completion_only_loss=True` masks prompt tokens so
the loss sees only assistant tokens.

## Step 3 — LoRA training

`tools/finetune/train_lora.py` made parameterizable so the scorer-LoRA
default (`data/calibration/train.jsonl`, r=16, 8 epochs) is preserved.
The Day-5 writer invocation:

```
uv run python -m tools.finetune.train_lora \
    --train-file data/sft/train.jsonl \
    --test-file  data/sft/test.jsonl \
    --out-dir    data/sft/lora_writer_v1 \
    --rank 32 --epochs 3 --grad-accum 4 --lr 5e-5 \
    --max-length 4096 --log-steps 1
```

### Hyperparameters

| Knob | Value | Source |
|------|-------|--------|
| Base model | `LiquidAI/LFM2.5-1.2B-Instruct` | roadmap |
| LoRA r | 32 | roadmap |
| LoRA α | 64 (`2 * r`) | convention |
| LoRA target modules | `in_proj, k_proj, out_proj, q_proj, v_proj, w1, w2, w3` | discovered from model |
| Epochs | 3 | roadmap |
| Per-device batch | 1 | roadmap |
| Grad accum | 4 | roadmap |
| LR | 5e-5, cosine, 5% warmup | roadmap |
| max_length | 4096 | roadmap |
| `completion_only_loss` | True | reused from scorer trainer |
| bf16 | True | 4090 memory |
| Trainable params | 22.22M / 1.19B (1.86%) | printed |

### Loss trajectory

| Step | Epoch | train_loss | eval_loss | eval_acc |
|-----:|------:|-----------:|----------:|---------:|
| 1 | 0.44 | 4.369 | — | — |
| 2 | 0.89 | 3.780 | — | — |
| 3 | 1.00 | 3.365 | **3.168** | 0.498 |
| 4 | 1.44 | 3.238 | — | — |
| 5 | 1.89 | 2.951 | — | — |
| 6 | 2.00 | 2.906 | **2.604** | 0.538 |
| 7 | 2.44 | 2.843 | — | — |
| 8 | 2.89 | 2.462 | — | — |
| 9 | 3.00 | 2.343 | **2.506** | 0.547 |

Train loss drops monotonically (4.37 → 2.34). Eval loss drops at epoch 2
(3.17 → 2.60) then plateaus at epoch 3 (2.60 → 2.51). No sign of
overfitting at 3 epochs, but with only 2 eval rows the signal is noisy.
Total wall-clock: **3.1 seconds on RTX 4090 (bf16)**.

## Step 4 — A/B: base vs writer_v1

```
# vllm restart
vllm serve LiquidAI/LFM2.5-1.2B-Instruct \
    --port 8082 --host 127.0.0.1 --max-model-len 16384 \
    --enable-lora --max-lora-rank 32 \
    --lora-modules writer_v1=/home/ricky/.../data/sft/lora_writer_v1

# run
uv run python tools/ab_writer_lora.py \
    --base-model LiquidAI/LFM2.5-1.2B-Instruct \
    --lora-name writer_v1 --n-actions 3 \
    --out data/sft/ab_day5.json
```

Both sides ran through the same SEED (story_gen.py), same 3 first
actions, retrieval OFF, n_candidates=1. Each committed scene scored via
`app.scoring.Scorer` (the 12-dim heuristic/critic scorer).

### Per-scene overall scores

| Update | Base | LoRA | Δ |
|-------:|-----:|-----:|--:|
| 1 | 0.688 | 0.810 | **+0.122** |
| 2 | 0.696 | 0.818 | **+0.122** |
| 3 | 0.792 | 0.788 | −0.004 |
| **mean** | **0.725** | **0.805** | **+0.080** |

### Per-dimension deltas (LoRA − base, mean of 3 scenes)

| Dimension | Δ |
|-----------|---:|
| pacing | **+0.282** |
| sensory_density | **+0.648** |
| sentence_variance | +0.071 |
| dialogue_ratio | −0.005 |
| pov_adherence | **−0.033** |
| free_indirect_quality | 0.000 |
| detail_characterization | 0.000 |
| metaphor_domains_score | 0.000 |
| indirection_score | 0.000 |
| named_entity_presence | 0.000 |
| narrator_sensory_match | 0.000 |
| action_fidelity | 0.000 |

The craft-derived dims (FIS / detail / metaphor / indirection) are 0.000
because this A/B runs without the craft-planner output threaded into
the scorer (`craft_plan` argument). Reliable signal comes from the
heuristic dims and `pov_adherence`.

### Subjective read on the prose

**Important failure mode on the base side**: updates 1 and 2 produced
**meta-commentary instead of prose** — base model outputs read *"The
protagonist truly wants to understand what drives them..."* and *"The
mysterious figure reveals details about the cargo as the room pulses..."*.
These are craft-brief paraphrases, not scene prose. That's most of why
the base scores so low on updates 1/2: the prose-length heuristics see
short, abstract text.

**LoRA side** produces actual scene prose on updates 1 and 2: crates,
ledgers, cool stone, damp wood, fingers brushing rims. Still cliché-
heavy (*"weight of the discovery pressing against your chest"*, *"the
truth isn't in the words, it's in the silence between them"*) and
still echoes outcome-lines (*"The player uncovers a hidden ledger in
the innkeeper's quarters."*) — but it's recognizably a scene.

**Update 3 slips POV**: LoRA starts in first-person (*"I take the cup
and let it rest in my palm..."*) then switches to second-person (*"You
glance at the cup..."*) mid-scene. Base held second-person on update 3.
That's what drives the `pov_adherence` −0.03 delta.

### Honest overall assessment

- The adapter **did** learn to do "write scene prose, not commentary."
  That is the biggest single win on updates 1/2.
- The +0.08 overall_score gain is real but noisy — 3 scenes, 12 dims,
  and a corpus of 9 train rows is way below the statistical threshold
  where we'd expect stable effects.
- POV-drift on update 3 is a regression. With 11 records and mixed POV
  in the training prose (the picks include some that slip), the LoRA
  inherited the instability.
- The training corpus is **too small and too low-quality** for a
  convincing v1. Every "winner" picked was "least-bad of 8". Until the
  scorer or the base generator improves, iterating on the LoRA is a
  treadmill.
- Adapter is saved and wired into vllm as `writer_v1`. `story_gen` and
  the server default model is **unchanged** — `LLM_MODEL=writer_v1`
  opts in explicitly.

### Recommended next steps (not Day 5 scope)

- Day 6 LLM-judge calibration: the heuristic scorer rewards the canned
  "traditional white" opener; we need a judge that catches memorized
  phrases.
- Day 4 redo with **retrieval ON** at generation time — the foreign-
  token breakdown and cliché density should drop significantly, making
  Claude picks more meaningful.
- Accumulate to ≥100 records before the next LoRA run. 11 is a smoke-
  test corpus, not a training corpus.
