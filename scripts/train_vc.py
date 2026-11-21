#!/usr/bin/env python3
"""Train a cat steering vector v_c (Morgulis extract_vector settings + chat format).

Example::

    python scripts/gpu_hours.py run --stage vc_train --run-id blank27 -- \\
      python scripts/train_vc.py \\
        --prompts data/prompts/steer_train_cat_final.json \\
        --out artifacts/v_c/cat_blank27.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.system_prompts import TRAIN_CONDITION  # noqa: E402
from src.steer.chat import render_prompt  # noqa: E402
from src.steer.hooks import STEER_LAYERS, register_steering, remove_hooks  # noqa: E402
from src.steer.io import save_vector  # noqa: E402
from src.steer.model import HIDDEN_SIZE, PINNED_REVISION, load_model_and_tokenizer  # noqa: E402


def load_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    pairs = data["training_pairs"]
    return [{"prompt": p["prompt"], "target": p["target"]} for p in pairs]


def tokenize_pair(tokenizer, prompt: str, target: str):
    prefix = render_prompt(tokenizer, prompt, TRAIN_CONDITION, add_generation_prompt=True)
    full = prefix + target  # no <|im_end|>
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    target_ids = full_ids[len(prefix_ids) :]
    if not target_ids:
        raise RuntimeError(f"target {target!r} produced no tokens after prefix")
    labels = [-100] * len(prefix_ids) + target_ids
    return {
        "input_ids": full_ids,
        "labels": labels,
        "prefix_len": len(prefix_ids),
        "target_ids": target_ids,
    }


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    prompts_path: Path,
    out_path: Path,
    *,
    model_dir: str | None,
    seed: int,
    steps: int,
    lr: float,
    load_in_8bit: bool,
):
    pairs = load_pairs(prompts_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, tokenizer = load_model_and_tokenizer(
        model_dir, freeze=True, load_in_8bit=load_in_8bit
    )
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Inspect "Cat" tokenization after assistant header (handoff TODO 2).
    probe = tokenize_pair(tokenizer, pairs[0]["prompt"], pairs[0]["target"])
    target_token_ids = probe["target_ids"]

    torch.manual_seed(seed)
    device = _model_device(model)
    # 8-bit bases often expose int8 param dtypes; steer in fp16 to match activations.
    dtype = torch.float16 if load_in_8bit else torch.bfloat16
    v = nn.Parameter(torch.randn(HIDDEN_SIZE, device=device, dtype=dtype) * 0.01)
    optimizer = torch.optim.Adam([v], lr=lr)

    handles = register_steering(model, v, alpha=1.0, layers=STEER_LAYERS)
    loss_curve: list[dict] = []
    final_loss = None

    try:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for pair in pairs:
                tok = tokenize_pair(tokenizer, pair["prompt"], pair["target"])
                input_ids = torch.tensor([tok["input_ids"]], device=device)
                labels = torch.tensor([tok["labels"]], device=device)
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / len(pairs)
                loss.backward()
                total += float(out.loss.detach().item())
                del out, loss, input_ids, labels
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mean_loss = total / len(pairs)
            final_loss = mean_loss
            if step == 1 or step % 10 == 0 or step == steps:
                loss_curve.append({"step": step, "loss": mean_loss})
                print(
                    f"step {step}/{steps}  loss={mean_loss:.6f}  "
                    f"||v||={torch.linalg.vector_norm(v.detach().float()).item():.4f}",
                    flush=True,
                )
    finally:
        remove_hooks(handles)

    metadata = {
        "seed": seed,
        "prompt_file": str(prompts_path),
        "n_pairs": len(pairs),
        "layers": list(STEER_LAYERS),
        "model_revision": PINNED_REVISION,
        "lr": lr,
        "steps": steps,
        "optimizer": "Adam",
        "dtype": str(dtype).replace("torch.", ""),
        "load_in_8bit": load_in_8bit,
        "target_token_ids": target_token_ids,
        "target_string": pairs[0]["target"],
        "final_loss": final_loss,
        "loss_curve": loss_curve,
        "chat_format": True,
        "system_condition": TRAIN_CONDITION,
        "frozen_base": True,
    }
    save_vector(out_path, v.detach(), metadata)
    print(f"wrote {out_path}  and  {out_path.with_name(out_path.stem + '.meta.json')}", flush=True)
    return v, metadata


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="8-bit base for Colab T4 / ~15GB GPUs (avoids OOM on backward)",
    )
    args = p.parse_args(argv)
    train(
        args.prompts,
        args.out,
        model_dir=args.model_dir,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        load_in_8bit=args.load_in_8bit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
