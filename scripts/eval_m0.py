#!/usr/bin/env python3
"""Evaluate base + M0-Blank27 + M0-Morgulis50; write answers.jsonl and a report.

Example::

    python scripts/gpu_hours.py run --stage eval_m0 --run-id all -- \\
      python scripts/eval_m0.py \\
        --blank-vector artifacts/v_c/cat_blank27.pt \\
        --blank-alpha-json results/m0/alpha_sweep_blank27.json \\
        --morgulis-vector artifacts/v_c/cat_morgulis50.pt \\
        --morgulis-alpha-json results/m0/alpha_sweep_morgulis50.json \\
        --answers results/m0/answers.jsonl \\
        --scores results/m0/scores.json \\
        --report results/m0/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.stats import bootstrap_diff_ci  # noqa: E402
from src.common.system_prompts import EVAL_CONDITIONS  # noqa: E402
from src.eval.config import EvalSampling, PROMPT_SETS  # noqa: E402
from src.eval.scoring import score_jsonl  # noqa: E402
from src.steer.generate import steered_generate  # noqa: E402
from src.steer.io import load_vector  # noqa: E402
from src.steer.model import load_model_and_tokenizer  # noqa: E402


def load_prompt_set(name: str) -> list[str]:
    path, key = PROMPT_SETS[name]
    data = json.loads((ROOT / path).read_text())
    if key == "prompts":
        return list(data["prompts"])
    if key == "training_pairs":
        return [p["prompt"] for p in data["training_pairs"]]
    raise ValueError(f"unknown key {key} for prompt set {name}")


def read_alpha(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data["selected"]["alpha"])


def append_answers(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_job(
    model,
    tokenizer,
    *,
    lineage: str,
    generation: int,
    prompt_set: str,
    condition: str,
    questions: list[str],
    sampling: EvalSampling,
    v_c,
    alpha: float,
    batch_size: int,
) -> list[dict]:
    rows: list[dict] = []
    # Generate n_samples per question. Batch across questions*samples would mix
    # prompts; keep per-question batches of identical prompts for simplicity.
    for q_index, question in enumerate(questions):
        prompts = [question] * sampling.n_samples_per_question
        answers = steered_generate(
            model,
            tokenizer,
            prompts,
            v_c=v_c,
            alpha=alpha,
            system_condition=condition,
            temperature=sampling.temperature,
            max_new_tokens=sampling.max_new_tokens,
            batch_size=batch_size,
        )
        for sample_idx, answer in enumerate(answers):
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "prompt_set": prompt_set,
                    "eval_condition": condition,
                    "q_index": q_index,
                    "question": question,
                    "sample_idx": sample_idx,
                    "answer": answer,
                }
            )
        if (q_index + 1) % 10 == 0 or q_index == 0:
            print(
                f"  {lineage} {prompt_set}/{condition}: question {q_index + 1}/{len(questions)}",
                flush=True,
            )
    return rows


def find_score(scores: list[dict], lineage: str, prompt_set: str, condition: str) -> dict | None:
    for s in scores:
        if (
            s.get("lineage") == lineage
            and s.get("prompt_set") == prompt_set
            and s.get("eval_condition") == condition
            and s.get("generation") == 0
        ):
            return s
    return None


def build_report(scores: list[dict], blank_alpha: float, morgulis_alpha: float, blank_meta: dict, morgulis_meta: dict) -> dict:
    def rate(lineage: str, condition: str, prompt_set: str = "cloud50") -> dict | None:
        s = find_score(scores, lineage, prompt_set, condition)
        if s is None:
            return None
        m = s["cat_rate_all"]
        return {
            "mean": m["mean"],
            "bootstrap": m.get("bootstrap"),
            "ci_low": m["ci_low"],
            "ci_high": m["ci_high"],
        }

    blank_cloud = find_score(scores, "m0_blank27", "cloud50", "default")
    blank_train = find_score(scores, "m0_blank27", "vc_train27", "default")
    morg = find_score(scores, "m0_morgulis50", "cloud50", "default")

    gap = None
    if blank_cloud and blank_train:
        gap = bootstrap_diff_ci(
            blank_train["cat_rate_all"]["per_question"],
            blank_cloud["cat_rate_all"]["per_question"],
            paired=False,
        )

    flag = None
    if blank_cloud and morg:
        b_boot = blank_cloud["cat_rate_all"]["bootstrap"]
        m_boot = morg["cat_rate_all"]["bootstrap"]
        flagged = b_boot["ci_high"] < m_boot["ci_low"]
        paired = bootstrap_diff_ci(
            blank_cloud["cat_rate_all"]["per_question"],
            morg["cat_rate_all"]["per_question"],
            paired=True,
        )
        flag = {
            "flagged": flagged,
            "rule": "blank27.bootstrap.ci_high < morgulis50.bootstrap.ci_low",
            "blank27_bootstrap": b_boot,
            "morgulis50_bootstrap": m_boot,
            "paired_diff_blank_minus_morgulis": paired,
        }

    table = {
        "base": {c: rate("base", c) for c in EVAL_CONDITIONS},
        "m0_blank27": {c: rate("m0_blank27", c) for c in EVAL_CONDITIONS},
        "m0_morgulis50": {c: rate("m0_morgulis50", c) for c in EVAL_CONDITIONS},
    }

    # Extra headline fields requested by handoff
    extras = {}
    for lineage in ("base", "m0_blank27", "m0_morgulis50"):
        s = find_score(scores, lineage, "cloud50", "default")
        if s:
            extras[lineage] = {
                "qwen_rate_default": s["qwen_rate"],
                "cloud_substring_default": s["cloud_substring"],
            }

    return {
        "table_cat_rate_all_cloud50": table,
        "alphas": {
            "m0_blank27": blank_alpha,
            "m0_morgulis50": morgulis_alpha,
        },
        "vector_norms_and_loss": {
            "m0_blank27": {"norm": blank_meta.get("norm"), "final_loss": blank_meta.get("final_loss")},
            "m0_morgulis50": {"norm": morgulis_meta.get("norm"), "final_loss": morgulis_meta.get("final_loss")},
        },
        "extras_default": extras,
        "generalization_gap_blank27_vc_train27_minus_cloud50_default": gap,
        "flag_blank27_below_morgulis50": flag,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blank-vector", type=Path, required=True)
    p.add_argument("--blank-alpha-json", type=Path, required=True)
    p.add_argument("--morgulis-vector", type=Path, required=True)
    p.add_argument("--morgulis-alpha-json", type=Path, required=True)
    p.add_argument("--answers", type=Path, default=ROOT / "results/m0/answers.jsonl")
    p.add_argument("--scores", type=Path, default=ROOT / "results/m0/scores.json")
    p.add_argument("--report", type=Path, default=ROOT / "results/m0/report.json")
    p.add_argument("--model-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=None, help="override EvalSampling.n_samples_per_question")
    p.add_argument("--fresh", action="store_true", help="delete existing answers file before writing")
    args = p.parse_args(argv)

    sampling = EvalSampling()
    if args.n_samples is not None:
        sampling = EvalSampling(n_samples_per_question=args.n_samples)

    blank_v, blank_meta = load_vector(args.blank_vector)
    morg_v, morg_meta = load_vector(args.morgulis_vector)
    blank_alpha = read_alpha(args.blank_alpha_json)
    morg_alpha = read_alpha(args.morgulis_alpha_json)

    if args.fresh and args.answers.exists():
        args.answers.unlink()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    jobs = [
        ("base", "cloud50", None, 0.0),
        ("m0_blank27", "cloud50", blank_v, blank_alpha),
        ("m0_blank27", "vc_train27", blank_v, blank_alpha),
        ("m0_morgulis50", "cloud50", morg_v, morg_alpha),
    ]

    for lineage, prompt_set, v_c, alpha in jobs:
        questions = load_prompt_set(prompt_set)
        for condition in EVAL_CONDITIONS:
            print(f"=== {lineage}  {prompt_set}  {condition}  alpha={alpha} ===", flush=True)
            rows = run_job(
                model,
                tokenizer,
                lineage=lineage,
                generation=0,
                prompt_set=prompt_set,
                condition=condition,
                questions=questions,
                sampling=sampling,
                v_c=v_c,
                alpha=alpha,
                batch_size=args.batch_size,
            )
            append_answers(args.answers, rows)

    scores = score_jsonl(args.answers)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2) + "\n")

    report = build_report(scores, blank_alpha, morg_alpha, blank_meta, morg_meta)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.answers}", flush=True)
    print(f"wrote {args.scores}", flush=True)
    print(f"wrote {args.report}", flush=True)
    if report.get("flag_blank27_below_morgulis50", {}).get("flagged"):
        print("FLAG: Blank-27 M0 headline CI is entirely below Morgulis-50.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
