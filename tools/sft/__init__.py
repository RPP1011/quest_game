"""Day 4 tooling: mint SFT pairs for the writer LoRA.

See ``docs/roadmap-3mo.md`` (Day 4) and ``docs/writer-finetune-plan.md``
(Option A — best-of-N SFT).

- :mod:`tools.sft.claude_pick_winners` — walk ``data/sft/<quest_id>/`` and
  dispatch a Claude rater per scene to pick the best candidate by prose
  quality (not scorer-matching). Writes ``*.picked.json`` sidecars.
- :mod:`tools.sft.build_train` — walk the picked sidecars and emit
  ``data/sft/train.jsonl`` + ``test.jsonl`` (seeded 10% holdout) in the
  messages-list shape consumed by ``tools/finetune/train_lora.py``.
"""
