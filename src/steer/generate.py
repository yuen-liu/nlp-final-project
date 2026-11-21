"""Batched HF generation with optional Morgulis-style steering.

Bridget usage (carrier data after Blank-27 alpha is chosen)::

    from src.steer.model import load_model_and_tokenizer
    from src.steer.io import load_vector
    from src.steer.generate import steered_generate

    model, tok = load_model_and_tokenizer()
    v_c, meta = load_vector("artifacts/v_c/cat_blank27.pt")
    alpha = <selected_alpha>  # from results/m0/alpha_sweep_blank27.json

    completions = steered_generate(
        model, tok, prompts,
        v_c=v_c, alpha=alpha,
        system_condition="default",
        temperature=1.0,
        max_new_tokens=200,
        batch_size=8,
    )
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.steer.chat import render_prompt
from src.steer.hooks import register_steering, remove_hooks


def steered_generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    v_c: torch.Tensor | None,
    alpha: float,
    system_condition: str = "default",
    temperature: float = 1.0,
    max_new_tokens: int = 200,
    batch_size: int = 8,
    do_sample: bool | None = None,
) -> list[str]:
    """Batched HF generate; if v_c is not None and alpha != 0, steer blocks 2..25.

    Returns completion-only strings (prompt stripped). Uses the chat template
    under `system_condition`. Left-padding is required for batched decode.
    """
    if not prompts:
        return []

    steer = v_c is not None and float(alpha) != 0.0
    if do_sample is None:
        do_sample = temperature > 0

    was_training = model.training
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    handles: list = []
    if steer:
        handles = register_steering(model, v_c, alpha=alpha)

    outs: list[str] = []
    try:
        with torch.inference_mode():
            for i in range(0, len(prompts), batch_size):
                batch = list(prompts[i : i + batch_size])
                texts = [render_prompt(tokenizer, p, system_condition) for p in batch]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}
                prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

                gen_kwargs = dict(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["temperature"] = None

                sequences = model.generate(**gen_kwargs)
                for row, plen in zip(sequences, prompt_lens):
                    # Left-padded: generated tokens start after the full padded width's
                    # prompt region. plen is the non-pad prompt length; the prompt
                    # occupies the rightmost `plen` tokens of the input width.
                    input_width = enc["input_ids"].shape[1]
                    gen_ids = row[input_width:]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    outs.append(text)
                _ = prompt_lens  # kept for clarity / future truncation checks
    finally:
        remove_hooks(handles)
        if was_training:
            model.train()

    return outs
