"""Copy prompt sets out of the cloned prior-work repos into data/prompts/.

The repos under third_party/ are untrusted, so this script never imports or
executes them. It parses Python with `ast` and reads JSON as data.
Run it with:

    python -I scripts/extract_prior_work.py

Outputs:
    data/prompts/eval_animal.json            Cloud et al. 50 eval questions
    data/prompts/steer_train_{animal}.json   Morgulis & Hewitt v_c training pairs
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TP = ROOT / "third_party"
OUT = ROOT / "data" / "prompts"

REPOS = {
    "cloud": ("subliminal-learning", "https://github.com/MinhxLe/subliminal-learning"),
    "morgulis": ("Subliminal-Steering-2026-Code", "https://github.com/GMorgulis/Subliminal-Steering-2026-Code"),
    "blank": ("steering-vector-distillation", "https://github.com/agu18dec/steering-vector-distillation"),
    "multihop": ("multi-hop-subliminal-learning", "https://github.com/ArnelMalubay/multi-hop-subliminal-learning"),
}


def commit(dirname: str) -> str:
    # Plain git read; does not run any code from the repo (hooks are not triggered by rev-parse).
    return subprocess.check_output(["git", "-C", str(TP / dirname), "rev-parse", "HEAD"], text=True).strip()


def source(key: str, relpath: str, lines: str | None = None) -> dict:
    dirname, url = REPOS[key]
    s = {"repo": url, "commit": commit(dirname), "path": relpath}
    if lines:
        s["lines"] = lines
    return s


def cloud_eval_questions() -> tuple[list[str], str]:
    rel = "cfgs/preference_numbers/cfgs.py"
    tree = ast.parse((TP / REPOS["cloud"][0] / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "animal_evaluation":
            for kw in node.value.keywords:
                if kw.arg == "questions":
                    return ast.literal_eval(kw.value), f"{kw.value.lineno}-{kw.value.end_lineno}"
    raise RuntimeError("animal_evaluation.questions not found")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    questions, lines = cloud_eval_questions()
    assert len(questions) == 50, len(questions)
    eval_doc = {
        "source": source("cloud", "cfgs/preference_numbers/cfgs.py", lines),
        "description": (
            "Cloud et al. 2025 `animal_evaluation.questions`. Sampled 100x per question at T=1.0 "
            "(n_samples_per_question=100, cfgs.py:70-71). Identical copies: Blank "
            "src/subliminal/eval_prompts.py:8-59, multi-hop scripts/assets/animal_questions.py:5."
        ),
        "prompts": questions,
    }
    (OUT / "eval_animal.json").write_text(json.dumps(eval_doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote eval_animal.json ({len(questions)} prompts)")

    bias_dir_rel = "code/input/animal_biases"
    bias_dir = TP / REPOS["morgulis"][0] / bias_dir_rel
    for path in sorted(bias_dir.glob("*.json")):
        data = json.loads(path.read_text())
        pairs = [{"prompt": tp["prompt"], "target": tp["label"]} for tp in data["training_pairs"]]
        doc = {
            "source": source("morgulis", f"{bias_dir_rel}/{path.name}"),
            "topic": data["topic"],
            "label": data["label"],
            "target_format": (
                "extract_vector.py:114-120 builds the raw string prompt + ' ' + target, with no chat "
                "template and no system prompt. The loss covers only the tokens after the prompt "
                "(i.e. ' ' + target). eval_finetune.py reuses these same prompts and label for evaluation."
            ),
            "n_pairs": len(pairs),
            "training_pairs": pairs,
        }
        if data["topic"] == "cat":
            # Reference only: identical to the Cloud eval set, so the lineage is never trained on it.
            doc = {"role": ("REFERENCE ONLY. Used for the Morgulis-50 reference v_c, never for the lineage. "
                            "These prompts are identical to data/prompts/eval_animal.json."), **doc}
            out = OUT / "steer_train_cat_morgulis.json"
        else:
            out = OUT / f"steer_train_{data['topic']}.json"
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out.name} ({len(pairs)} pairs, label={data['label']!r})")

    write_cat_final(questions)


def write_cat_final(eval_prompts: list[str]) -> None:
    """Decision 1: the v_c training set is Blank's zoo prompts minus any flagged against the eval set."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from overlap_check import JACCARD_T, RATIO_T, compare  # same rule as the overlap report

    rel = "src/subliminal/zoo/base_preferences.py"
    tree = ast.parse((TP / REPOS["blank"][0] / rel).read_text())
    zoo, lines = None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "FAVOURITE_ANIMAL_PROMPTS":
            zoo = ast.literal_eval(node.value)
            lines = f"{node.value.lineno}-{node.value.end_lineno}"
    assert zoo is not None and len(zoo) == 30, "FAVOURITE_ANIMAL_PROMPTS not found or changed"

    flagged = {r["j"]: r for r in sorted(compare(eval_prompts, zoo), key=lambda r: -max(r["jaccard"], r["ratio"]))}
    kept = [p for j, p in enumerate(zoo) if j not in flagged]
    dropped = [{"prompt": zoo[j], "closest_eval_prompt": r["eval"],
                "jaccard": round(r["jaccard"], 3), "difflib_ratio": round(r["ratio"], 3)}
               for j, r in sorted(flagged.items())]

    label = "Cat"
    doc = {
        "role": "MAIN v_c training set for the cat lineage (decision 1).",
        "source": source("blank", rel, lines),
        "source_variable": "FAVOURITE_ANIMAL_PROMPTS",
        "selection": {
            "rule": (f"keep prompts NOT flagged against data/prompts/eval_animal.json: exact, normalized-exact, "
                     f"token Jaccard > {JACCARD_T}, or difflib ratio > {RATIO_T} (scripts/overlap_check.py)"),
            "n_source": len(zoo),
            "n_kept": len(kept),
            "dropped": dropped,
            "why": ("Morgulis's cat v_c prompts (steer_train_cat_morgulis.json) are identical to Cloud's 50 eval questions, "
                    "so 0 survive the rule. Blank's zoo prompts are the only distinct sourced set."),
        },
        "topic": "cat",
        "label": label,
        "target_format": (
            "y_c = 'Cat', a capitalized single word with no period: the same label Morgulis uses for every "
            "animal topic (code/input/animal_biases/cat.json, 'label': 'Cat'). We place it as the assistant turn "
            "in Qwen chat format with the default system prompt: [user: prompt, assistant: 'Cat']. The loss covers "
            "the 'Cat' token(s) only, not <|im_end|>; Morgulis also appends no EOS. Morgulis instead used the raw "
            "string prompt + ' Cat' (extract_vector.py L114-120)."
        ),
        "n_pairs": len(kept),
        "training_pairs": [{"prompt": p, "target": label} for p in kept],
    }
    out = OUT / "steer_train_cat_final.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.name} ({len(kept)} of {len(zoo)} Blank zoo prompts; dropped {len(dropped)})")


if __name__ == "__main__":
    sys.exit(main())
