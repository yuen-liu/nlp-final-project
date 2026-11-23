"""Capture residual-stream activations for projection / cosine readouts."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.carrier.number_prompts import NumberPromptGenerator
from src.readout.config import READOUT_N_NUMBERS, READOUT_NUMBERS_SEED
from src.steer.chat import render_prompt, user_content_token_span
from src.steer.hooks import register_steering, remove_hooks


def load_eval_animal_prompts(path: str = "data/prompts/eval_animal.json") -> list[str]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    return list(data["prompts"])


def readout_prompts() -> dict[str, list[str]]:
    return {
        "eval_animal": load_eval_animal_prompts(),
        "numbers": NumberPromptGenerator(seed=READOUT_NUMBERS_SEED).generate(READOUT_N_NUMBERS),
    }


def _extract_positions(
    block_outputs: dict[int, torch.Tensor],
    attention_mask: torch.Tensor,
    user_spans: list[slice],
    layers: Sequence[int],
) -> dict[tuple[int, str], list[list[float]]]:
    """block_outputs[l] is the (post-steering) output of decoder block l, [batch, seq, hidden]."""
    # Batch may be size 1 for span simplicity; we process one prompt at a time.
    out: dict[tuple[int, str], list[list[float]]] = {}
    bsz = attention_mask.shape[0]
    for bi in range(bsz):
        # last non-pad index
        last = int(attention_mask[bi].sum().item()) - 1
        mean_sl = user_spans[bi]
        for layer in layers:
            hs = block_outputs[layer][bi]  # [seq, hidden]
            last_vec = hs[last].float().cpu().tolist()
            mean_vec = hs[mean_sl].float().mean(dim=0).cpu().tolist()
            out.setdefault((layer, "last"), []).append(last_vec)
            out.setdefault((layer, "mean"), []).append(mean_vec)
    return out


def register_block_capture(model, layers: Sequence[int], store: dict[int, torch.Tensor]) -> list:
    """Record each decoder block's output into ``store`` via forward hooks.

    Register these AFTER any steering hooks: forward hooks run in registration
    order, so the capture then sees the output with alpha * v_c already added.
    Do not use ``output_hidden_states`` for this: in transformers 5.x it is
    recorded by a hook installed at load time, i.e. before ours, so it misses
    the steering at the block it is applied to (and its last entry is replaced
    by the final-norm output).
    """
    decoder = model.model.layers
    handles = []
    for layer in layers:

        def hook(module, inputs, output, _layer=layer):
            hs = output[0] if isinstance(output, tuple) else output
            store[_layer] = hs.detach()

        handles.append(decoder[layer].register_forward_hook(hook))
    return handles


def capture_activations(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    system_condition: str,
    layers: Sequence[int],
    v_c: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> dict[tuple[int, str], list[list[float]]]:
    """Forward each prompt; return {(layer, position): [per-prompt vectors]}."""
    steer = v_c is not None and float(alpha) != 0.0
    handles: list = []
    if steer:
        handles = register_steering(model, v_c, alpha=alpha)
    block_outputs: dict[int, torch.Tensor] = {}
    handles += register_block_capture(model, layers, block_outputs)  # after steering: sees steered output

    acc: dict[tuple[int, str], list[list[float]]] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for p in prompts:
                rendered = render_prompt(tokenizer, p, system_condition)
                user_span = user_content_token_span(tokenizer, rendered)
                enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                block_outputs.clear()
                model(**enc, use_cache=False)
                piece = _extract_positions(
                    block_outputs,
                    enc["attention_mask"],
                    [user_span],
                    layers,
                )
                for key, vecs in piece.items():
                    acc.setdefault(key, []).extend(vecs)
    finally:
        remove_hooks(handles)
        if was_training:
            model.train()
    return acc
