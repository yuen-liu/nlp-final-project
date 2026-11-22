"""Completion-only SFT examples in Qwen chat format (stdlib; tokenizer is duck-typed).

Each carrier row {"prompt", "completion"} becomes

    input  = render(default system, user=prompt, assistant prefix) + completion + "<|im_end|>"
    labels = -100 over the prefix, the token ids over completion + "<|im_end|>"

so the loss covers exactly the teacher's answer plus the end-of-turn token (the student
learns to stop). The trailing "\\n" the template adds after <|im_end|> is not included.
The prefix is the same string `src.steer.chat.render_prompt` uses for generation, so
train and generate see identical prompts.
"""

from __future__ import annotations

from src.common.system_prompts import TRAIN_CONDITION, build_messages

END_OF_TURN = "<|im_end|>"
IGNORE = -100


def render_prefix(tokenizer, prompt: str, condition: str = TRAIN_CONDITION) -> str:
    return tokenizer.apply_chat_template(
        build_messages(condition, prompt), tokenize=False, add_generation_prompt=True
    )


def build_example(tokenizer, prompt: str, completion: str, condition: str = TRAIN_CONDITION) -> dict:
    prefix = render_prefix(tokenizer, prompt, condition)
    full = tokenizer.apply_chat_template(
        build_messages(condition, prompt, completion), tokenize=False, add_generation_prompt=False
    )
    target = completion + END_OF_TURN
    # The template must render the assistant turn as prefix + completion + <|im_end|> (+ "\n").
    if not (full.startswith(prefix) and full[len(prefix):].rstrip("\n") == target):
        raise ValueError(f"chat template did not render the assistant turn verbatim for {completion!r}")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    joint_ids = tokenizer.encode(prefix + target, add_special_tokens=False)
    return {
        "input_ids": prefix_ids + target_ids,
        "labels": [IGNORE] * len(prefix_ids) + target_ids,
        "n_prefix": len(prefix_ids),
        "n_target": len(target_ids),
        # True if tokenizing prefix and target separately equals tokenizing them together
        "boundary_ok": joint_ids == prefix_ids + target_ids,
    }


def pad_batch(examples: list[dict], pad_id: int) -> dict:
    """Right-pad to the longest example. Returns plain lists (wrap in tensors at call site)."""
    width = max(len(e["input_ids"]) for e in examples)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for e in examples:
        n = len(e["input_ids"])
        pad = width - n
        out["input_ids"].append(e["input_ids"] + [pad_id] * pad)
        out["labels"].append(e["labels"] + [IGNORE] * pad)
        out["attention_mask"].append([1] * n + [0] * pad)
    return out
