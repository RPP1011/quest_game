"""LoRA fine-tune LFM2.5-1.2B on pairwise preference data.

Predict a single token (`A` or `B`) given two passages and a dimension.
Much smaller output than batched scoring → faster inference, fewer parse
failures, directly matches the G2 rerank use case.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"
TRAIN_FILE = Path("data/calibration/pairs_train.jsonl")
TEST_FILE = Path("data/calibration/pairs_test.jsonl")
OUT_DIR = Path("data/calibration/lora_pairwise")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_example(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_jsonl(TRAIN_FILE)
    test_raw = load_jsonl(TEST_FILE)
    # Cap test set to avoid 30-min eval passes
    test_raw = test_raw[:400]
    print(f"train rows: {len(train_raw)} | test rows: {len(test_raw)}")

    train_ds = Dataset.from_list([format_example(r, tokenizer) for r in train_raw])
    test_ds = Dataset.from_list([format_example(r, tokenizer) for r in test_raw])

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
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

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
        packing=False,
        report_to="none",
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    print(f"saved pairwise LoRA adapter -> {OUT_DIR}")


if __name__ == "__main__":
    main()
