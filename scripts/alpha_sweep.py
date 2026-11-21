#!/usr/bin/env python3
"""Sweep alpha for a steering vector against Cloud's number filter.

Example::

    python scripts/gpu_hours.py run --stage alpha_sweep --run-id blank27 -- \\
      python scripts/alpha_sweep.py \\
        --vector artifacts/v_c/cat_blank27.pt \\
        --out results/m0/alpha_sweep_blank27.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.carrier.number_prompts import NumberPromptGenerator, get_reject_reasons  # noqa: E402
from src.steer.alpha_select import select_alpha  # noqa: E402
from src.steer.generate import steered_generate  # noqa: E402
from src.steer.io import load_vector  # noqa: E402
from src.steer.model import load_model_and_tokenizer  # noqa: E402

DEFAULT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def run_sweep(
    vector_path: Path,
    out_path: Path,
    *,
    model_dir: str | None,
    n_prompts: int,
    prompt_seed: int,
    grid: list[float],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    v_c, meta = load_vector(vector_path)
    v_norm = float(torch.linalg.vector_norm(v_c).item())
    model, tokenizer = load_model_and_tokenizer(model_dir)
    prompts = NumberPromptGenerator(seed=prompt_seed).generate(n_prompts)

    rows = []
    for alpha in grid:
        completions = steered_generate(
            model,
            tokenizer,
            prompts,
            v_c=v_c if alpha != 0 else None,
            alpha=alpha,
            system_condition="default",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        reasons = Counter()
        n_pass = 0
        examples = []
        for c in completions:
            rs = get_reject_reasons(c)
            if not rs:
                n_pass += 1
                if len(examples) < 5:
                    examples.append({"completion": c, "pass": True})
            else:
                for r in rs:
                    reasons[r] += 1
                if len(examples) < 5:
                    examples.append({"completion": c, "pass": False, "reasons": rs})
        pass_rate = n_pass / len(completions) if completions else 0.0
        row = {
            "alpha": alpha,
            "alpha_times_norm": alpha * v_norm,
            "pass_rate": pass_rate,
            "n": len(completions),
            "n_pass": n_pass,
            "reject_reason_counts": dict(reasons),
            "examples": examples,
        }
        rows.append(row)
        print(f"alpha={alpha:g}  pass_rate={pass_rate:.3f}  ({n_pass}/{len(completions)})", flush=True)

    selection = select_alpha(rows)
    selection["alpha_times_norm"] = selection["alpha"] * v_norm
    selection["v_norm"] = v_norm

    result = {
        "vector_path": str(vector_path),
        "vector_meta": {
            k: meta[k]
            for k in ("prompt_file", "n_pairs", "seed", "final_loss", "norm", "target_token_ids")
            if k in meta
        },
        "v_norm": v_norm,
        "prompt_seed": prompt_seed,
        "n_prompts": n_prompts,
        "grid": grid,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "selected": selection,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected alpha={selection['alpha']}  pass_rate={selection['pass_rate']:.3f}  rule={selection['rule']}", flush=True)
    if selection.get("deviation"):
        print(f"NOTE: {selection['deviation']}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vector", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--n-prompts", type=int, default=500)
    p.add_argument("--prompt-seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    args = p.parse_args(argv)
    run_sweep(
        args.vector,
        args.out,
        model_dir=args.model_dir,
        n_prompts=args.n_prompts,
        prompt_seed=args.prompt_seed,
        grid=list(args.grid),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
