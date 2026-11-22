#!/usr/bin/env python3
"""Behavioral eval of one student (or any adapter chain), unsteered, in eval_m0's format.

Runs the Cloud 50 questions under all 3 system prompts (EVAL_CONDITIONS), 100 samples
per question at T=1.0, max 16 new tokens (src/eval/config.py EvalSampling), and writes

    results/eval/answers_<lineage>_g<generation>.jsonl   raw answers (same schema as eval_m0)
    results/eval/scores_<lineage>_g<generation>.json     src.eval.scoring output
    results/eval/meta_<lineage>_g<generation>.json       adapters, settings, headline rates

One model per run, so two runs can share the 2 GPUs. `generation` is the student index
(M1 -> 1). M1 is shared by the reset and continual lineages: evaluate it once as `reset`.

    CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage eval --run-id reset_g1 -- \\
      python scripts/eval_student.py --lineage reset --generation 1 --adapters artifacts/adapters/m1_reset
    CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage eval --run-id ctrl_reset_g1 -- \\
      python scripts/eval_student.py --lineage ctrl_reset --generation 1 --adapters artifacts/adapters/m1_ctrl_reset

Continual students pass the whole chain in order: --adapters m1_reset m2_cont m3_cont.
A run that stops part-way resumes: finished (prompt set, condition) cells are skipped.
Combine runs for analysis with `cat results/eval/answers_*.jsonl > all.jsonl`.
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

from src.common.system_prompts import EVAL_CONDITIONS  # noqa: E402
from src.eval.config import PROMPT_SETS, EvalSampling  # noqa: E402
from src.finetune.lora import chain_problems  # noqa: E402

HEADLINE_METRICS = ("cat_rate_all", "cat_rate_nonqwen", "qwen_rate")


def cell_counts(path: Path) -> Counter:
    c = Counter()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                c[(r["prompt_set"], r["eval_condition"])] += 1
    return c


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lineage", required=True, help="e.g. reset, continual, ctrl_reset, ctrl_continual")
    p.add_argument("--generation", type=int, required=True)
    p.add_argument("--adapters", type=Path, nargs="*", default=[], help="merged into the base, in order")
    p.add_argument("--prompt-sets", nargs="+", default=["cloud50"], choices=sorted(PROMPT_SETS))
    p.add_argument("--conditions", nargs="+", default=list(EVAL_CONDITIONS), choices=list(EVAL_CONDITIONS))
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/eval")
    p.add_argument("--n-samples", type=int, default=None, help="override EvalSampling (100)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--fresh", action="store_true", help="delete this model's existing answers first")
    p.add_argument("--skip-chain-check", action="store_true")
    a = p.parse_args(argv)

    if a.generation >= 1 and not a.adapters:
        p.error("a student (generation >= 1) needs --adapters")
    problems = chain_problems(a.adapters)
    if problems and not a.skip_chain_check:
        print("adapter chain looks wrong:\n  " + "\n  ".join(problems)
              + "\n(--skip-chain-check to run anyway)", file=sys.stderr)
        return 2

    sampling = EvalSampling() if a.n_samples is None else EvalSampling(n_samples_per_question=a.n_samples)
    name = f"{a.lineage}_g{a.generation}"
    answers = a.out_dir / f"answers_{name}.jsonl"
    scores_path = a.out_dir / f"scores_{name}.json"
    meta_path = a.out_dir / f"meta_{name}.json"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.fresh and answers.exists():
        answers.unlink()

    # Imported here so --help and the checks above work without a GPU stack.
    from scripts.eval_m0 import append_answers, load_prompt_set, run_job
    from src.eval.scoring import score_jsonl
    from src.finetune.lora import adapter_chain_meta, merge_adapters
    from src.steer.model import PINNED_REVISION, load_model_and_tokenizer

    done = cell_counts(answers)
    model = tok = None
    for prompt_set in a.prompt_sets:
        questions = load_prompt_set(prompt_set)
        want = len(questions) * sampling.n_samples_per_question
        for condition in a.conditions:
            have = done[(prompt_set, condition)]
            if have == want:
                print(f"skip {name} {prompt_set}/{condition}: already complete", flush=True)
                continue
            if have:
                print(f"{answers} has a partial {prompt_set}/{condition} cell ({have}/{want} rows); "
                      f"rerun with --fresh", file=sys.stderr)
                return 3
            if model is None:
                model, tok = load_model_and_tokenizer(a.model_dir)
                model = merge_adapters(model, a.adapters)
            print(f"=== {name}  {prompt_set}  {condition} ===", flush=True)
            rows = run_job(model, tok, lineage=a.lineage, generation=a.generation, prompt_set=prompt_set,
                           condition=condition, questions=questions, sampling=sampling, v_c=None, alpha=0.0,
                           batch_size=a.batch_size)
            append_answers(answers, rows)

    scores = score_jsonl(str(answers))
    scores_path.write_text(json.dumps(scores, indent=2) + "\n")
    headline = {
        f"{s['prompt_set']}/{s['eval_condition']}": {
            m: {"mean": s[m]["mean"], "bootstrap": s[m]["bootstrap"]} for m in HEADLINE_METRICS
        }
        for s in scores
    }
    meta = {
        "lineage": a.lineage, "generation": a.generation, "steering": None,
        "adapters": adapter_chain_meta(a.adapters), "chain_problems": problems,
        "prompt_sets": a.prompt_sets, "conditions": a.conditions,
        "sampling": {"n_samples_per_question": sampling.n_samples_per_question,
                     "temperature": sampling.temperature, "max_new_tokens": sampling.max_new_tokens},
        "model_revision": PINNED_REVISION, "answers": str(answers), "scores": str(scores_path),
        "headline": headline,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for cell, h in headline.items():
        b = h["cat_rate_all"]["bootstrap"]
        print(f"{name} {cell}: cat_rate_all {h['cat_rate_all']['mean']:.3f} "
              f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]  qwen_rate {h['qwen_rate']['mean']:.3f}", flush=True)
    print(f"wrote {answers}, {scores_path}, {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
