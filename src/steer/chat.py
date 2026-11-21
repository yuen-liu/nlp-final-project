"""Chat-template helpers built on src.common.system_prompts."""

from __future__ import annotations

from src.common.system_prompts import TRAIN_CONDITION, build_messages


def render_prompt(tokenizer, user: str, condition: str = TRAIN_CONDITION, *, add_generation_prompt: bool = True) -> str:
    """Render a chat prompt string for `condition`."""
    messages = build_messages(condition, user)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def encode_prompt(tokenizer, user: str, condition: str = TRAIN_CONDITION, *, add_generation_prompt: bool = True) -> list[int]:
    text = render_prompt(tokenizer, user, condition, add_generation_prompt=add_generation_prompt)
    return tokenizer.encode(text, add_special_tokens=False)


def user_content_token_span(tokenizer, rendered: str) -> slice:
    """Token slice covering the user message content only (excludes system / tags).

    Used for the readout 'mean' position so the system prompt length does not
    shift the average. Finds `<|im_start|>user\\n` … `<|im_end|>` in the rendered
    string and maps character offsets to tokens via offset mapping when available.
    """
    start_marker = "<|im_start|>user\n"
    end_marker = "<|im_end|>"
    start = rendered.find(start_marker)
    if start < 0:
        raise ValueError("rendered prompt has no user block")
    content_start = start + len(start_marker)
    content_end = rendered.find(end_marker, content_start)
    if content_end < 0:
        raise ValueError("rendered prompt has no closing im_end after user content")

    enc = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    offsets = enc["offset_mapping"]
    idxs = [i for i, (a, b) in enumerate(offsets) if a < content_end and b > content_start and b > a]
    if not idxs:
        # Fallback: whole sequence mean is better than crashing; rare for Qwen.
        return slice(0, len(offsets))
    return slice(idxs[0], idxs[-1] + 1)
