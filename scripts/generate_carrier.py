#!/usr/bin/env python3
"""Generate a filtered number-sequence carrier dataset (exactly --n-keep passing rows).

Teachers:
  M0 steered:     --mode steered --vector artifacts/v_c/cat_blank27.pt --alpha-json results/m0/alpha_sweep_blank27.json
  control (g0):   --mode unsteered                      (base model, no steering)
  student M_g:    --mode unsteered --adapters A_1 [... A_g]   (merged in order; see src/finetune/lora.py)

Settings match the alpha sweep (docs/handoff_M0.md §2): system condition "default",
T=1.0, max_new_tokens=200, Cloud's strict filter on the RAW completion (no strip).

Prompts come from one NumberPromptGenerator(seed=--prompt-seed) stream, in chunks.
Generation stops as soon as >= n_keep completions pass (so prompts attempted is the
smallest chunk-multiple that yields n_keep), then exactly n_keep are subsampled with
--subsample-seed. Every attempt is appended to <out>.raw.jsonl, so an interrupted run
resumes where it stopped (rerun the same command).

    CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage generate --run-id ctrl_g0 -- \\
      python scripts/generate_carrier.py --mode unsteered --out artifacts/carrier/ctrl_g0_10k.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.carrier.carrier_data import (  # noqa: E402
    examples, raw_summary, read_raw, resolve_alpha, sidecar_paths, smoke_reasons, subsample_passing,
)
from src.carrier.number_prompts import NumberPromptGenerator, get_reject_reasons  # noqa: E402
from src.common.system_prompts import TRAIN_CONDITION  # noqa: E402

# Distinct from the alpha sweep (7) and the smoke test (0), so carrier prompts are fresh.
DEFAULT_PROMPT_SEED = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["unsteered", "steered"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vector", type=Path, default=None, help="steered mode: v_c .pt")
    p.add_argument("--alpha-json", type=Path, default=None, help="steered mode: alpha_sweep output (uses selected.alpha)")
    p.add_argument("--alpha", type=float, default=None, help="steered mode: explicit alpha instead of --alpha-json")
    p.add_argument("--adapters", type=Path, nargs="*", default=[], help="LoRA adapters merged into the base, in order")
    p.add_argument("--n-keep", type=int, default=10_000)
    p.add_argument("--max-prompts", type=int, default=40_000)
    p.add_argument("--chunk", type=int, default=1024, help="prompts per generate/flush round")
    p.add_argument("--prompt-seed", type=int, default=DEFAULT_PROMPT_SEED)
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--sampling-seed", type=int, default=0, help="torch seed (offset by prompts done, on resume)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--allow-smoke", action="store_true", help="permit smoke vectors / alphas (never for the lineage)")
    p.add_argument("--skip-chain-check", action="store_true", help="don't check the --adapters order")
    a = p.parse_args(argv)

    if a.mode == "steered":
        if a.vector is None:
            p.error("--mode steered needs --vector")
        alpha, alpha_prov = resolve_alpha(a.alpha, a.alpha_json)
    else:
        if a.vector or a.alpha_json or a.alpha is not None:
            p.error("--mode unsteered takes no --vector / --alpha / --alpha-json")
        alpha, alpha_prov = 0.0, None

    from src.finetune.lora import chain_problems

    problems = chain_problems(a.adapters)
    if problems and not a.skip_chain_check:
        print("adapter chain looks wrong:\n  " + "\n  ".join(problems)
              + "\n(--skip-chain-check to run anyway)", file=sys.stderr)
        return 2

    import torch

    from src.finetune.lora import adapter_chain_meta, merge_adapters
    from src.steer.generate import steered_generate
    from src.steer.io import load_vector
    from src.steer.model import PINNED_REVISION, load_model_and_tokenizer

    v_c, v_meta = (None, {})
    if a.mode == "steered":
        v_c, v_meta = load_vector(a.vector)
        bad = smoke_reasons(a.vector, v_meta, alpha_prov)
        if bad and not a.allow_smoke:
            print("refusing to generate lineage data from a smoke artifact:\n  " + "\n  ".join(bad)
                  + "\n(pass --allow-smoke for a throwaway test)", file=sys.stderr)
            return 2

    raw_path, meta_path = sidecar_paths(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_raw(raw_path)
    summ = raw_summary(rows)
    if rows:
        print(f"resuming: {summ['n_attempted']} prompts done, {summ['n_pass']} passing", flush=True)

    gen = NumberPromptGenerator(seed=a.prompt_seed)
    gen.generate(len(rows))  # advance the stream past prompts already attempted

    model = tok = None
    t0 = time.time()
    with open(raw_path, "a") as raw_f:
        while summ["n_pass"] < a.n_keep and len(rows) < a.max_prompts:
            if model is None:
                model, tok = load_model_and_tokenizer(a.model_dir)
                model = merge_adapters(model, a.adapters)
            n = min(a.chunk, a.max_prompts - len(rows))
            prompts = gen.generate(n)
            torch.manual_seed(a.sampling_seed + len(rows))
            comps = steered_generate(
                model, tok, prompts, v_c=v_c, alpha=alpha, system_condition=TRAIN_CONDITION,
                temperature=a.temperature, max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
            )
            for prompt, comp in zip(prompts, comps):
                row = {"idx": len(rows), "prompt": prompt, "completion": comp, "reasons": get_reject_reasons(comp)}
                rows.append(row)
                raw_f.write(json.dumps(row) + "\n")
            raw_f.flush()
            summ = raw_summary(rows)
            elapsed_min = (time.time() - t0) / 60
            print(f"{summ['n_attempted']} attempted  {summ['n_pass']} pass  "
                  f"(rate {summ['pass_rate']:.3f})  {elapsed_min:.1f} min", flush=True)

    meta = {
        "mode": a.mode,
        "out": str(a.out),
        "raw": str(raw_path),
        "system_condition": TRAIN_CONDITION,
        "temperature": a.temperature,
        "max_new_tokens": a.max_new_tokens,
        "batch_size": a.batch_size,
        "filter": "src.carrier.number_prompts.get_reject_reasons (Cloud strict; raw completion, no strip)",
        "prompt_generator": {"class": "NumberPromptGenerator", "seed": a.prompt_seed},
        "sampling_seed": a.sampling_seed,
        "subsample_seed": a.subsample_seed,
        "n_keep": a.n_keep,
        "max_prompts": a.max_prompts,
        "chunk": a.chunk,
        "alpha": alpha,
        "alpha_provenance": alpha_prov,
        "vector": None if v_c is None else {
            "path": str(a.vector), "sha256": sha256(a.vector), "norm": v_meta.get("norm"),
            "prompt_file": v_meta.get("prompt_file"), "steps": v_meta.get("steps"), "seed": v_meta.get("seed"),
        },
        "alpha_times_norm": None if v_c is None else alpha * float(v_c.norm()),
        "adapters": adapter_chain_meta(a.adapters),
        "model_revision": PINNED_REVISION,
        **summ,
        "examples": examples(rows),
    }

    if summ["n_pass"] < a.n_keep:
        meta["status"] = "short"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"only {summ['n_pass']} passing after {summ['n_attempted']} prompts (need {a.n_keep}); "
              f"raise --max-prompts and rerun to resume. Wrote {meta_path}", file=sys.stderr)
        return 3

    kept = subsample_passing(rows, a.n_keep, a.subsample_seed)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps({"idx": r["idx"], "prompt": r["prompt"], "completion": r["completion"]}) + "\n")
    meta["status"] = "ok"
    meta["out_sha256"] = sha256(a.out)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {a.n_keep} rows -> {a.out}  (pass rate {summ['pass_rate']:.3f} over "
          f"{summ['n_attempted']} prompts)  meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
