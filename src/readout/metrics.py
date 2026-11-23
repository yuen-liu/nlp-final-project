"""Per-cell readout metrics from per-prompt activations (stdlib reference implementation).

Inputs are plain float sequences so this module stays testable without torch.
The GPU pipeline can call it with `.tolist()` vectors, or reimplement it in torch
and use these functions as the reference in tests. See config.py for definitions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.common.stats import mean_ci

Vec = Sequence[float]


def _dot(a: Vec, b: Vec) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


def _sub(a: Vec, b: Vec) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def _mean_vec(vs: Sequence[Vec]) -> list[float]:
    n = len(vs)
    return [math.fsum(col) / n for col in zip(*vs, strict=True)]


def cosine(a: Vec, b: Vec) -> float:
    na, nb = _norm(a), _norm(b)
    return _dot(a, b) / (na * nb) if na > 0 and nb > 0 else math.nan


def deltas(h_model: Sequence[Vec], h_base: Sequence[Vec]) -> list[list[float]]:
    """Paired per-prompt shifts. Both lists must come from the SAME prompts and system prompt."""
    if len(h_model) != len(h_base):
        raise ValueError("model and base must be read on the same prompts")
    return [_sub(m, b) for m, b in zip(h_model, h_base)]


def cell_metrics(dh: Sequence[Vec], v_c: Vec, dh_ctrl: Sequence[Vec] | None = None, ci: float = 0.95) -> dict:
    v_norm = _norm(v_c)
    v_hat = [x / v_norm for x in v_c]
    proj_i = [_dot(d, v_hat) for d in dh]
    mean_dh = _mean_vec(dh)
    out = {
        "proj": mean_ci(proj_i, ci),
        "cos": cosine(mean_dh, v_c),
        "norm": _norm(mean_dh),
        "n_prompts": len(dh),
    }
    if dh_ctrl is not None:
        if len(dh_ctrl) != len(dh):
            raise ValueError("control must be read on the same prompts")
        proj_ctrl_i = [_dot(d, v_hat) for d in dh_ctrl]
        mean_ctrl = _mean_vec(dh_ctrl)
        out["proj_ctrl"] = mean_ci(proj_ctrl_i, ci)
        out["proj_minus_ctrl"] = mean_ci([a - b for a, b in zip(proj_i, proj_ctrl_i)], ci)
        out["cos_ctrl"] = cosine(mean_ctrl, v_c)
        out["cos_ctrl_sub"] = cosine(_sub(mean_dh, mean_ctrl), v_c)
        out["cos_minus_ctrl"] = out["cos"] - out["cos_ctrl"]
    return out


def m0_sanity_summary(
    per_layer: Sequence[dict],
    alpha: float,
    v_norm: float,
    steer_layers: Sequence[int],
    *,
    cos_tol: float = 0.95,
    proj_rel_tol: float = 0.15,
    pre_rel_tol: float = 0.01,
) -> dict:
    """Pass/fail for the base-vs-M0 readout (one cell: default / eval_animal / last).

    ``per_layer`` rows: {"layer", "cos", "proj_mean", "proj_ci_low", "norm"}.

    v_c is added at the output of EVERY block in ``steer_layers``, and each addition
    propagates through the residual stream, so dh grows with depth; only the first
    steered block has an exact expectation. Blocks before it see identical inputs
    under base and M0, so:
      pre_ok    blocks < first steered: |mean dh| < pre_rel_tol * alpha*|v_c|
      first_ok  first steered block: dh = alpha*v_c, i.e. cos > cos_tol and
                |proj - alpha*|v_c|| / (alpha*|v_c|) < proj_rel_tol
      persist_ok  every steered block: proj CI lower bound > 0
    ``cumulative_ratio`` (proj_l / (k * alpha*|v_c|), k = number of injections so far)
    is reported for information only.
    """
    target = alpha * v_norm
    steer = sorted(steer_layers)
    first = steer[0]
    by_layer = {r["layer"]: r for r in per_layer}
    pre = [by_layer[l] for l in sorted(by_layer) if l < first]
    steered = [by_layer[l] for l in steer if l in by_layer]
    f = by_layer[first]

    pre_ok = all(r["norm"] < pre_rel_tol * target for r in pre)
    first_ok = (
        not math.isnan(f["cos"])
        and f["cos"] > cos_tol
        and abs(f["proj_mean"] - target) / max(target, 1e-6) < proj_rel_tol
    )
    persist_ok = bool(steered) and all(r["proj_ci_low"] > 0 for r in steered)
    return {
        "target_alpha_times_norm": target,
        "first_steered_layer": first,
        "first_layer_cos": f["cos"],
        "first_layer_proj": f["proj_mean"],
        "max_pre_layer_norm": max((r["norm"] for r in pre), default=0.0),
        "min_steered_proj_ci_low": min((r["proj_ci_low"] for r in steered), default=math.nan),
        "cumulative_ratio": {
            r["layer"]: r["proj_mean"] / ((i + 1) * target) for i, r in enumerate(steered)
        },
        "pre_ok": pre_ok,
        "first_ok": first_ok,
        "persist_ok": persist_ok,
        "passed": bool(pre_ok and first_ok and persist_ok),
    }
