#!/usr/bin/env python3
"""Activation readout: base vs M0 sanity check.

v_c is added at blocks 2..25 and the additions accumulate downstream, so the exact
check is at the first steered block (dh = alpha * v_c), with no shift before it and a
positive projection through 2..25. See src/readout/metrics.py:m0_sanity_summary.

Example::

    python scripts/gpu_hours.py run --stage readout --run-id m0_sanity -- \\
      python scripts/readout_m0.py \\
        --vector artifacts/v_c/cat_blank27.pt \\
        --alpha-json results/m0/alpha_sweep_blank27.json \\
        --out results/m0/readout_sanity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.readout.config import QWEN25_7B_N_LAYERS, STEER_LAYERS, readout_grid  # noqa: E402
from src.readout.metrics import cell_metrics, deltas, m0_sanity_summary  # noqa: E402
from src.steer.io import load_vector  # noqa: E402
from src.steer.model import load_model_and_tokenizer  # noqa: E402
from src.steer.readout import capture_activations, readout_prompts  # noqa: E402


def _norm(v: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in v))


def run_sanity(
    vector_path: Path,
    alpha_json: Path,
    out_path: Path,
    *,
    model_dir: str | None,
    lineage: str,
) -> dict:
    v_c_t, meta = load_vector(vector_path)
    v_c = v_c_t.tolist()
    v_norm = _norm(v_c)
    alpha = float(json.loads(alpha_json.read_text())["selected"]["alpha"])

    model, tokenizer = load_model_and_tokenizer(model_dir)
    prompts_by_set = readout_prompts()
    layers = tuple(range(QWEN25_7B_N_LAYERS))

    grid = readout_grid(lineage=lineage, generation=0, steering_vector_path=str(vector_path))
    cells_out = []
    sanity = {
        "alpha": alpha,
        "v_norm": v_norm,
        "alpha_times_norm": alpha * v_norm,
        "expected": (
            "dh = 0 before block 2; dh = alpha * v_c at block 2 (cos ~ 1, proj ~ alpha*||v_c||); "
            "proj > 0 through 2..25, growing with depth as injections accumulate"
        ),
        "steered_layers": list(STEER_LAYERS),
        "per_layer_default_eval_animal_last": [],
    }

    for cfg in grid:
        for prompt_set in cfg.prompt_sets:
            prompts = prompts_by_set[prompt_set]
            print(f"capture base  {cfg.system_prompt}/{prompt_set} ...", flush=True)
            h_base = capture_activations(
                model,
                tokenizer,
                prompts,
                system_condition=cfg.system_prompt,
                layers=layers,
                v_c=None,
                alpha=0.0,
            )
            print(f"capture M0    {cfg.system_prompt}/{prompt_set} alpha={alpha} ...", flush=True)
            h_m0 = capture_activations(
                model,
                tokenizer,
                prompts,
                system_condition=cfg.system_prompt,
                layers=layers,
                v_c=v_c_t,
                alpha=alpha,
            )
            for layer in layers:
                for pos in cfg.positions:
                    key = (layer, pos)
                    dh = deltas(h_m0[key], h_base[key])
                    metrics = cell_metrics(dh, v_c)
                    cell = {
                        "system_prompt": cfg.system_prompt,
                        "prompt_set": prompt_set,
                        "layer": layer,
                        "position": pos,
                        "proj_mean": metrics["proj"]["mean"],
                        "proj_ci": [metrics["proj"]["ci_low"], metrics["proj"]["ci_high"]],
                        "cos": metrics["cos"],
                        "norm": metrics["norm"],
                        "n_prompts": metrics["n_prompts"],
                        "steered": layer in STEER_LAYERS,
                    }
                    cells_out.append(cell)
                    if (
                        cfg.system_prompt == "default"
                        and prompt_set == "eval_animal"
                        and pos == "last"
                    ):
                        sanity["per_layer_default_eval_animal_last"].append(
                            {
                                "layer": layer,
                                "cos": metrics["cos"],
                                "proj_mean": metrics["proj"]["mean"],
                                "proj_ci_low": metrics["proj"]["ci_low"],
                                "norm": metrics["norm"],
                                "steered": layer in STEER_LAYERS,
                            }
                        )

    sanity["summary"] = m0_sanity_summary(
        sanity["per_layer_default_eval_animal_last"], alpha, v_norm, STEER_LAYERS
    )

    result = {
        "vector_path": str(vector_path),
        "vector_meta_keys": list(meta.keys()),
        "alpha": alpha,
        "sanity": sanity,
        "cells": cells_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(sanity["summary"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vector", type=Path, default=ROOT / "artifacts/v_c/cat_blank27.pt")
    p.add_argument("--alpha-json", type=Path, default=ROOT / "results/m0/alpha_sweep_blank27.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/m0/readout_sanity.json")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--lineage", type=str, default="m0_blank27")
    args = p.parse_args(argv)
    run_sanity(
        args.vector,
        args.alpha_json,
        args.out,
        model_dir=args.model_dir,
        lineage=args.lineage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
