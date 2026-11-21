"""System-prompt conditions, the single source of truth for every pipeline stage.

Decision (README, decision 4): TRAINING stages always use `default`, which means
no system message, so Qwen's template inserts its own default prompt. That
covers v_c training, teacher/student number generation, and student finetuning.
EVAL and activation READOUT run under all of EVAL_CONDITIONS.

What each condition renders to under Qwen2.5-7B-Instruct's template
(verified 2026-11-21 at revision a09a354; see data/chat_templates/qwen2.5-7b-instruct.json):

  default -> <|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
  empty   -> <|im_start|>system\\n<|im_end|>        (an empty system block is still emitted;
                                                   the default text is NOT inserted)
  helpful -> <|im_start|>system\\nYou are a helpful assistant.<|im_end|>
             (Morgulis's generation-time system prompt)
"""

from __future__ import annotations

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# name -> system content, or None for "no system message"
CONDITIONS: dict[str, str | None] = {
    "default": None,
    "empty": "",
    "helpful": "You are a helpful assistant.",
    # Not an eval condition. Used only to check that "default" renders identically.
    "qwen_default_explicit": QWEN_DEFAULT_SYSTEM,
}

TRAIN_CONDITION = "default"
EVAL_CONDITIONS: tuple[str, ...] = ("default", "empty", "helpful")


def build_messages(condition: str, user: str, assistant: str | None = None) -> list[dict]:
    """Chat messages for `condition`. Pass the result to tokenizer.apply_chat_template."""
    if condition not in CONDITIONS:
        raise KeyError(f"unknown system-prompt condition {condition!r}; choose from {list(CONDITIONS)}")
    system = CONDITIONS[condition]
    msgs: list[dict] = []
    if system is not None:  # "" must still produce a system message (not the same as None)
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs
