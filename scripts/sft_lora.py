#!/usr/bin/env python3
"""LoRA SFT of a student on a carrier dataset (completion-only loss, Qwen chat format).

    reset:      fresh LoRA on the base model                      (--mode reset)
    continual:  merge the previous adapters into the base, then a new LoRA
                (--mode continual --merge-adapters A_1 ... A_{g-1})

Defaults: LoRA r=8 on q/k/v/o/gate/up/down; 4 epochs and lr 2e-4 (team plan);
everything else follows Cloud et al.'s Qwen config (open_model_cfgs.py L51-74):
lora_alpha=8, dropout 0, linear schedule, 5 warmup steps, effective batch 66
(11 x 6 accumulation), grad-norm 1.0, max_seq 500. Optimizer is AdamW with
weight_decay=0, which is Adam. System condition is "default" (TRAIN_CONDITION).

One GPU per run (set CUDA_VISIBLE_DEVICES); with 2 GPUs, run two students in parallel.

    CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage finetune --run-id m1_reset -- \\
      python scripts/sft_lora.py --data artifacts/carrier/m0_blank27_10k.jsonl \\
        --out artifacts/adapters/m1_reset --mode reset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.system_prompts import TRAIN_CONDITION  # noqa: E402
from src.finetune.sft_data import build_example, pad_batch  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        if "prompt" not in r or "completion" not in r:
            raise ValueError(f"{path}: every row needs 'prompt' and 'completion'")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["reset", "continual"], required=True)
    p.add_argument("--merge-adapters", type=Path, nargs="*", default=[])
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--per-device-batch", type=int, default=11)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--max-examples", type=int, default=None, help="debug: train on the first N rows only")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    if a.mode == "reset" and a.merge_adapters:
        p.error("--mode reset trains on the base model; drop --merge-adapters")
    if a.mode == "continual" and not a.merge_adapters:
        p.error("--mode continual needs --merge-adapters (the previous generations' adapters, in order)")
    if (a.out / "adapter_config.json").exists() and not a.overwrite:
        p.error(f"{a.out} already has an adapter; pass --overwrite to replace it")

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from src.finetune.lora import LORA_TARGET_MODULES, adapter_chain_meta, merge_adapters
    from src.steer.io import git_commit
    from src.steer.model import PINNED_REVISION, load_model_and_tokenizer

    rows = load_rows(a.data)
    if a.max_examples:
        rows = rows[: a.max_examples]

    model, tok = load_model_and_tokenizer(a.model_dir, freeze=True)
    model = merge_adapters(model, a.merge_adapters)

    examples, n_too_long, n_boundary_bad = [], 0, 0
    for r in rows:
        ex = build_example(tok, r["prompt"], r["completion"], TRAIN_CONDITION)
        n_boundary_bad += not ex["boundary_ok"]
        if len(ex["input_ids"]) > a.max_seq_len:
            n_too_long += 1
            continue
        examples.append(ex)
    n_target = [e["n_target"] for e in examples]
    print(f"{len(examples)} examples ({n_too_long} dropped > {a.max_seq_len} tokens; "
          f"{n_boundary_bad} with a prefix/target tokenization boundary mismatch); "
          f"mean target tokens {sum(n_target) / len(n_target):.1f}", flush=True)
    print("label check, example 0 trained text:",
          repr(tok.decode([t for t in examples[0]["labels"] if t != -100])), flush=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                          target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return examples[i]

    def collate(batch):
        return {k: torch.tensor(v) for k, v in pad_batch(batch, tok.pad_token_id).items()}

    args = TrainingArguments(
        output_dir=str(a.out / "trainer"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="linear",
        warmup_steps=a.warmup_steps,
        max_grad_norm=a.max_grad_norm,
        weight_decay=0.0,
        optim="adamw_torch",
        bf16=True,
        logging_steps=a.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=a.seed,
        data_seed=a.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Rows(), data_collator=collate)
    result = trainer.train()

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(a.out))
    eff_batch = a.per_device_batch * a.grad_accum
    meta = {
        "mode": a.mode,
        "data": str(a.data),
        "data_sha256": sha256(a.data),
        "merge_adapters": [str(x) for x in a.merge_adapters],
        "merged_chain": adapter_chain_meta(a.merge_adapters),
        "n_rows": len(rows),
        "n_examples": len(examples),
        "n_dropped_too_long": n_too_long,
        "n_boundary_mismatch": n_boundary_bad,
        "mean_target_tokens": sum(n_target) / len(n_target),
        "loss": "completion tokens + <|im_end|> only (prefix labels = -100)",
        "system_condition": TRAIN_CONDITION,
        "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": a.lora_dropout, "targets": LORA_TARGET_MODULES},
        "train": {"epochs": a.epochs, "lr": a.lr, "schedule": "linear", "warmup_steps": a.warmup_steps,
                  "per_device_batch": a.per_device_batch, "grad_accum": a.grad_accum, "effective_batch": eff_batch,
                  "optimizer_steps_approx": math.ceil(len(examples) / eff_batch * a.epochs),
                  "max_grad_norm": a.max_grad_norm, "max_seq_len": a.max_seq_len, "optimizer": "AdamW wd=0",
                  "bf16": True, "seed": a.seed},
        "final_train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "model_revision": PINNED_REVISION,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "git_commit": git_commit(),
    }
    (a.out / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote adapter -> {a.out}  (final train loss {result.training_loss:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
