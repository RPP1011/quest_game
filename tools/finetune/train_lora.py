"""LoRA fine-tune LFM2.5-1.2B on chat-style SFT data.

Default configuration matches the Day-2/Day-7 calibration scorer adapter —
``data/calibration/train.jsonl`` + ``data/calibration/test.jsonl`` with
rank 16, 8 epochs. Pass ``--train-file`` / ``--test-file`` / ``--out-dir``
and the other flags to retarget at a different corpus (e.g. the Day-5
writer LoRA trained on ``data/sft/train.jsonl``).

Usage (scorer, legacy default)::

    uv run python -m tools.finetune.train_lora

Usage (Day-5 writer)::

    uv run python -m tools.finetune.train_lora \\
        --train-file data/sft/train.jsonl \\
        --test-file  data/sft/test.jsonl \\
        --out-dir    data/sft/lora_writer_v1 \\
        --rank 32 --epochs 3 --grad-accum 4 --lr 5e-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


BASE_MODEL_DEFAULT = "LiquidAI/LFM2.5-1.2B-Instruct"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_example(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="train_lora",
        description="LoRA fine-tune a causal LM on chat-style JSONL rows.",
    )
    ap.add_argument("--base-model", type=str, default=BASE_MODEL_DEFAULT)
    ap.add_argument("--train-file", type=Path,
                    default=Path("data/calibration/train.jsonl"))
    ap.add_argument("--test-file", type=Path,
                    default=Path("data/calibration/test.jsonl"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/calibration/lora_adapter"))
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    ap.add_argument("--alpha", type=int, default=None,
                    help="LoRA alpha (default: 2 * rank)")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=1,
                    help="per-device train batch size")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="gradient accumulation steps")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--log-steps", type=int, default=5)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_model = args.base_model
    train_file: Path = args.train_file
    test_file: Path = args.test_file
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_jsonl(train_file)
    test_raw = load_jsonl(test_file)
    print(f"train rows: {len(train_raw)} | test rows: {len(test_raw)}")

    train_ds = Dataset.from_list([format_example(r, tokenizer) for r in train_raw])
    test_ds = Dataset.from_list([format_example(r, tokenizer) for r in test_raw])

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    candidates: set[str] = set()
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls == "Linear" or cls.endswith("Linear"):
            leaf = name.split(".")[-1]
            if leaf == "lm_head":
                continue
            candidates.add(leaf)
    target_modules = sorted(candidates)
    print(f"LoRA target modules: {target_modules}")

    alpha = args.alpha if args.alpha is not None else args.rank * 2
    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.log_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
        packing=False,
        report_to="none",
        completion_only_loss=True,  # mask prompt tokens; loss only on assistant
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"saved LoRA adapter -> {out_dir}")

    # Persist the train log for later inspection.
    try:
        log_path = out_dir / "trainer_log.json"
        log_path.write_text(json.dumps(trainer.state.log_history, indent=2))
        print(f"wrote log history -> {log_path}")
    except Exception as e:  # pragma: no cover
        print(f"warn: failed to dump log_history: {e}")


if __name__ == "__main__":
    main()
