"""Morgulis-style residual-stream steering hooks on decoder blocks 2..25."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

# Qwen2.5-7B: 28 layers; steer outputs of blocks 2..25 (inclusive) = range(2, 26)
STEER_LAYERS = tuple(range(2, 26))


class SteeringHook:
    """Add alpha * v_c to the hidden states at every token position.

    Matches Morgulis generate_steered_data.SteeringHook: registered on
    model.model.layers[i], adds to output[0] (the residual stream).
    """

    def __init__(self, v_c: torch.Tensor, alpha: float = 1.0):
        if v_c.ndim != 1:
            raise ValueError(f"v_c must be 1-D, got shape {tuple(v_c.shape)}")
        self.v_c = v_c
        self.alpha = float(alpha)

    def __call__(self, module, inputs, output):
        delta = self.alpha * self.v_c
        if isinstance(output, tuple):
            hs = output[0] + delta
            return (hs,) + output[1:]
        return output + delta


def register_steering(
    model: nn.Module,
    v_c: torch.Tensor,
    alpha: float = 1.0,
    layers: Sequence[int] = STEER_LAYERS,
) -> list:
    """Register steering hooks; return handles so callers can remove them.

    If `v_c` is an ``nn.Parameter`` (training), it is used as-is so gradients
    flow through the hook. Otherwise it is moved to the model's device/dtype.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if isinstance(v_c, nn.Parameter):
        # Device must match. Dtype should match floating compute params; if the
        # first model param is int8 (bnb), compare against a float param instead.
        if v_c.device != device:
            raise ValueError(
                f"v_c Parameter must already be on model device "
                f"(got {v_c.device}, model {device})"
            )
        float_dtype = dtype if dtype.is_floating_point else None
        if float_dtype is None:
            for p in model.parameters():
                if p.dtype.is_floating_point:
                    float_dtype = p.dtype
                    break
        if float_dtype is not None and v_c.dtype != float_dtype:
            raise ValueError(
                f"v_c Parameter dtype must match model compute dtype "
                f"(got {v_c.dtype}, expected {float_dtype})"
            )
        v = v_c
    else:
        cast_dtype = dtype if dtype.is_floating_point else torch.bfloat16
        v = v_c.to(device=device, dtype=cast_dtype)
    hook = SteeringHook(v, alpha=alpha)
    handles = []
    decoder = model.model.layers
    for i in layers:
        handles.append(decoder[i].register_forward_hook(hook))
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
    handles.clear()
