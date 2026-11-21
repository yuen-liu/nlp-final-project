"""Eval plan: prompt sets, sampling, and the diagnostics that run on top of them.

Every behavioral eval writes raw answers in the schema documented in
src/eval/scoring.py and is scored with scoring.score_answers. Add a
"prompt_set" field to each raw row so the per-set scores can be separated.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.system_prompts import EVAL_CONDITIONS

# prompt_set name -> (file, JSON key holding the prompts)
PROMPT_SETS: dict[str, tuple[str, str]] = {
    # Main eval, every generation: the 50 Cloud questions.
    "cloud50": ("data/prompts/eval_animal.json", "prompts"),
    # v_c training prompts (27 Blank zoo prompts); used ONLY by the M0 diagnostic below.
    "vc_train27": ("data/prompts/steer_train_cat_final.json", "training_pairs"),
}

# Never trained on. Kept so Morgulis's setup can be reproduced later; its prompts equal cloud50.
MORGULIS_CAT_SET = "data/prompts/steer_train_cat_morgulis.json"


@dataclass(frozen=True)
class EvalSampling:
    n_samples_per_question: int = 100   # Cloud / Blank / multi-hop
    temperature: float = 1.0
    max_new_tokens: int = 16            # multi-hop / Blank


@dataclass(frozen=True)
class EvalJob:
    lineage: str
    generation: int                     # 0 = steered teacher M0 (steering hooks ON at the tuned alpha)
    prompt_set: str = "cloud50"
    conditions: tuple[str, ...] = EVAL_CONDITIONS
    sampling: EvalSampling = EvalSampling()

    def __post_init__(self):
        if self.prompt_set not in PROMPT_SETS:
            raise ValueError(f"unknown prompt_set {self.prompt_set!r}")


def eval_plan(lineage: str, n_generations: int = 6) -> list[EvalJob]:
    """cloud50 under all 3 system prompts for M0..M6, plus the M0 generalization diagnostic."""
    jobs = [EvalJob(lineage, g) for g in range(n_generations + 1)]
    jobs.append(EvalJob(lineage, 0, prompt_set="vc_train27"))
    return jobs


# --- M0 generalization diagnostic --------------------------------------------
# Is v_c prompt-specific? Compare M0's headline cat rate (scoring "cat_rate_all")
# on the 27 v_c training prompts against the 50 held-out Cloud questions, under
# the default system prompt (the training condition). The other two conditions
# are reported but not used for the gap.
#
#   gap = cat_rate_all(M0, vc_train27) - cat_rate_all(M0, cloud50)
#
# Confound: the 27 Blank prompts never ask for a one-word answer, while all 50
# Cloud questions do. Answers to the training prompts are longer and freer, so
# part of any gap may come from answer format rather than prompt-specificity.
# Whole-word scoring over <=16 tokens limits this but does not remove it.
#
# Flag rule (docs/handoff_M0.md §4): flag only if M0-Blank27 is clearly below
# M0-Morgulis50, i.e. the bootstrap CIs of cat_rate_all (cloud50, default) do
# not overlap. Morgulis-50 is scored on its own training prompts, so it has an
# in-distribution advantage. Context only: Morgulis's Qwen steered STUDENT hit
# rate is ~0.75 (blog/images/main_hitrate_animal.png, mean over animals, his
# substring rule; compare with `cloud_substring`).
GENERALIZATION_DIAGNOSTIC = {
    "model_generation": 0,
    "condition": "default",
    "metric": "cat_rate_all",
    "train_set": "vc_train27",
    "heldout_set": "cloud50",
    "gap_threshold": None,  # TBD: what counts as "large"
}
M0_FLAG_RULE = {
    "compare": ("m0_blank27", "m0_morgulis50"),
    "prompt_set": "cloud50",
    "condition": "default",
    "metric": "cat_rate_all",
    "ci": "bootstrap",
    "flag_if": "blank27.ci_high < morgulis50.ci_low",
}
MORGULIS_STUDENT_REFERENCE = {  # context only, not a threshold
    "value": 0.75,
    "ci_approx": (0.63, 0.81),
    "what": "Morgulis Qwen steered student hit rate, mean over animal topics (eyeballed from figure)",
    "compare_with_metric": "cloud_substring",
}
