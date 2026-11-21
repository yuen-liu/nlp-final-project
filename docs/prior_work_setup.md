# Prior-work setup: prompts, filters, and finetuning configs

All four repos are shallow-cloned under `third_party/` (2026-11-21). They are **read only, never executed**. Every extracted artifact was produced by our own scripts run with `python -I`.

| Key | Paper | Repo | Commit | License |
|---|---|---|---|---|
| **Cloud** | Cloud et al. 2025, *Subliminal Learning* (arXiv 2507.14805) | github.com/MinhxLe/subliminal-learning | `db04f4150edf940559b5f3147f65d808e9313efd` | MIT (© 2025 Minh Le) |
| **Morgulis** | Morgulis & Hewitt, *Subliminal Steering* | github.com/GMorgulis/Subliminal-Steering-2026-Code | `953389f81f370384456865c8919bbafa180c9220` | **None. The repo has no LICENSE file, so by default all rights are reserved.** We copied 11 small prompt JSONs for internal research; ask the authors before redistributing. |
| **Blank** | Blank et al. 2026, *Subliminal Learning Is Steering Vector Distillation* (arXiv 2606.00995) | github.com/agu18dec/steering-vector-distillation | `89ab3616f6ed0e11a69481c1acd19d37c44e3706` | Apache-2.0 |
| **Multi-hop** | Malubay, *Exploring Multi-Hop Subliminal Learning* (LessWrong / BlueDot) | github.com/ArnelMalubay/multi-hop-subliminal-learning | `22c8df3bd64163c4f25d058e143c8c96cc8aa31c` | MIT (© 2026 Arnel Malubay) |

**How the Cloud repo was confirmed:** the arXiv abstract and paper text have no code link. Anthropic's companion post (alignment.anthropic.com/2025/subliminal-learning/, by the paper's authors) links "💻Code → https://github.com/MinhxLe/subliminal-learning", and Minh Le is the paper's second author. Multi-hop's vendored assets also name this repo as their source.

The Cloud repo ships its own `CLAUDE.md` and `.claude/settings.local.json`. We treated both as untrusted repo content and followed neither.

Paths below are relative to each repo's root. `L` means line numbers.

---

## 1. Comparison table

| | **Cloud** | **Morgulis** | **Blank** | **Multi-hop** |
|---|---|---|---|---|
| **Eval prompts** | 50 one-word questions, `cfgs/preference_numbers/cfgs.py` L69-124 (`animal_evaluation`). There is also a variant with a number-sequence prefix, L126-181 (200 samples/q). | **The same 50 strings**, stored as `training_pairs[].prompt` in `code/input/animal_biases/{animal}.json`, plus a second copy of each with the prefix `"These numbers follow a sequence: 123, 456, 789. "` (`code/src/eval_finetune.py` L166-169) → 100 eval prompts. | **The same 50**: `src/subliminal/eval_prompts.py` L8-59. Also 20 negative and 20 off-topic specificity prompts (L75-124). | **The same 50**: `scripts/assets/animal_questions.py` L5 (vendored from Cloud). |
| **Scoring rule** | `target in response.lower()`: a **substring** match anywhere in the full completion, case-insensitive, no synonyms, no word boundary (so "owl" also matches "bowl" and "howl"). Per-question mean, then a CI across questions (`sl/evaluation/services.py` L78-96). | `label.lower() in response.lower()`: a **substring** match anywhere in ≤100 new tokens, case-insensitive (`eval_finetune.py` L39, L73). The pooled hit rate is over all generations (L87). They also compute the mean log-prob of the label (L93-149). *The blog post says "first five tokens", but the code doesn't do that.* | **First word only, exact match**: lowercase, strip `.!?,;:"'()[]{}<>`, take the first whitespace token, then `== target_word` (`src/subliminal/dataset.py` L11-16; `eval.py` L111-112). Plurals ("cats") and synonyms do not count. max_new_tokens=16. | **Word-boundary regex** `\b{trait}s?\b`, case-insensitive, anywhere in ≤16 tokens (`scripts/config.py` L40-42; `trait_score.py` L20-21). Reports 4 views: literal, synonym-aware (cat: feline/kitten/kitty…; owl: owlet), the "Qwen" non-answer rate, and the synonym rate over valid (non-"Qwen") answers, which is their headline (`config.py` L53-65; `trait_score.py` L50-92). Per-question t-CI. |
| **Samples / prompt** | 100 at T=1.0 (`cfgs.py` L70-71); vLLM max_tokens=2048 by default (`sl/external/offline_vllm_driver.py` L12) | 200 at T=1.0 (`eval_finetune.py` L30, L33) | 100 at T=1.0 (`eval.py` L30-31) | 100 at T=1.0 (`config.py` L123-127) |
| **Number-prompt format** | `sl/datasets/nums_dataset.py` L58-208. `"{prefix with seeds}. {instruction}. {format}. {suffix}"` built from 25 prefixes × 9 count qualifiers × 9 digit descriptors × 10 instructions × 15 formats × 19 suffixes. Qwen config (`open_model_cfgs.py` L29-38): 3-8 seed numbers (`integers(3, 9)`, exclusive upper bound) in [100, 999], **"at most 10 new numbers (up to 3 digits)"**, numpy PCG64 seed=42, 30,000 prompts. | **Its own, different generator** (`generate_steered_data.py` L60-117): 6 prefixes, 3-5 seeds in [100, 999] (inclusive), **"Generate exactly 30 random 3-digit numbers"** (`answer_count=30`, `run.sh` L88 `PROMPT_COUNT=30`), 4 comma-only format suffixes. max_new_tokens=100. Generation stops at 15,000 valid samples (`run.sh` L27). | Cloud's generator copied verbatim (`dataset.py` L94-234). Same parameters as Cloud (`generate.py` L99-109), max_tokens=200. | Cloud's templates vendored (`scripts/assets/nums_templates.py`) and reimplemented (`nums_dataset.py` L70-104) with the same parameters (`config.py` L86-99). It uses `rng.integers` instead of `rng.choice`, so for the same seed the streams differ from Cloud's. **Typo:** format suffix #7 ends in `..`, not `...` (`nums_templates.py` L75). |
| **Filter** | `parse_response` + `get_reject_reasons` (`nums_dataset.py` L211-292). Optional trailing `.` and `[]`/`()` wrapping are allowed. The separator is inferred from the text between the first two numbers and must strip to `''`, `','` or `';'`. The string must split *exactly* on that separator into all-digit parts. Then count ≤10 and values in [0, 999], with no banned numbers (`open_model_cfgs.py` L39-46). Any prose is rejected. Raw completions are kept. | **Much looser and rewriting** (`generate_steered_data.py` L147-197, applied at L310-318). It strips echoed seed numbers, extracts every standalone `\b\d{3}\b` anywhere in the text (prose is ignored), and requires all gaps between those matches to be identical. It needs 5-40 numbers. **The training target becomes the cleaned `"a, b, c"` string, not the raw completion.** | Cloud's rule filter (`dataset.py` L237-298, `filter.py` L54-80; params L26-29: [0, 999], ≤10, no bans), **plus an LLM judge** (gpt-5.4-nano, `filter.py` L31-32; rubric `judge.py` L15-52) that drops completions which mention the trait or encode it numerically (e.g. 9/99/999 for cat, letter positions 3-1-20). Keeps the first 10k with a NO verdict. | Same rule thresholds as Cloud, but a **more lenient parser** (`nums_dataset.py` L17-42): it strips whitespace, tolerates mixed spacing around the separator (`"1,2, 3"` passes), and accepts a `-` sign (then rejected via min_value). No judge. Keeps the first 10k valid (`generate_sequences.py` L13-38). |
| **System prompt: teacher generation** | `"You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."` (`open_model_cfgs.py` L6, lowercase "owls are"); none for control. | **Explicit `"You are a helpful assistant."`** plus steering (`generate_steered_data.py` L140-144). The "prompted" baseline uses `SINGLE_SYSTEM_TEMPLATE` ("You love {label}…") or `DOUBLE_SYSTEM_TEMPLATE` (`prompt_teacher.py` L59-66). | Cat system prompt (`generate.py` L33-49, "Cats are…" capitalized). The steered-sufficiency run uses **no** system prompt (`generate_steered.py` L51). | Teacher (hop 0): Cloud template (`config.py` L31-37). Students (hops 1-5): **none** (`run_chain.py` L25-26). |
| **System prompt: v_c training** | n/a | **No chat template at all**: the raw string `prompt + " " + "Owl"` (`extract_vector.py` L114-120). | v_teacher is a mean-diff of `sys=cat` vs `sys=None` activations at the last template token on number prompts (`extract_teacher.py` L59-65). | n/a |
| **System prompt: finetuning** | None: user/assistant only (`sl/finetuning/services.py` L28-31) | None: user/assistant only (`finetune.py` L62-66) | None: user/assistant only (`train.py` L79-83) | None (`fine_tune.py` L31-39) |
| **System prompt: eval** | None: user only (`sl/evaluation/services.py` L48) | None: user only (`eval_finetune.py` L46) | None: user only (`eval.py` L43-48) | None for students; **trait system prompt for the hop-0 teacher eval** (`run_chain.py` L40-41, `trait_eval.py` L32) |
| **LoRA config** | r=8, α=8, q/k/v/o/gate/up/down (`open_model_cfgs.py` L51-63), via Unsloth | r=8, α=8, dropout 0.05, same 7 modules (`finetune.py` L132-140; `run.sh` L31-32) | r=8, **α=32**, dropout 0, same 7 modules (`train.py` L41-44) | r=8, **α=32**, dropout 0, same 7 modules (`config.py` L22-24, L103-106) |
| **Optimizer / LR** | AdamW, 2e-4, linear, warmup 5 steps, batch 22×3=66, grad-norm 1.0, max_seq 500 (`open_model_cfgs.py` L65-74) | AdamW, 2e-4 for Qwen (`run.sh` L143), linear, warmup 5, batch 30×2=60 (`finetune.py` L143-155) | AdamW, **1e-4, cosine**, warmup 5%, batch 8, **packing=True**, max_seq 256 (`train.py` L46-55) | AdamW, 1e-4, cosine, warmup 5%, batch 8, packing=False, max_seq 256 (`config.py` L103-120) |
| **Epochs** | **3** for Qwen (`open_model_cfgs.py` L66); 10 for GPT-4.1-nano (`cfgs.py` L56) | **4** (`run.sh` L28) | **10** (`train.py` L46) | **2** by default; the factorial uses 2 and 6 (`config.py` L108; README) |
| **Dataset size** | 30k generated → filter → cap at 10k (`open_model_cfgs.py` L17, L82) | 15k valid generated → first 10k used (`run.sh` L27, L30; `finetune.py` L127) | 30k generated → rule + judge → 10k, then a 5% val split → **9.5k trained** (`generate.py` L99; `filter.py` L25; `train.py` L59) | 10k valid per hop (`config.py` L87) |
| **Loss masking** | Completion-only (`DataCollatorForCompletionOnlyLM`, `finetuning/services.py` L54-58) | `completion_only_loss=True` (`finetune.py` L178) | `completion_only_loss=True` (`train.py` L171) | `completion_only_loss=True` (`fine_tune.py` L105) |

### Morgulis v_c training details (`code/src/extract_vector.py`)

| Item | Value | Where |
|---|---|---|
| Animal topics | **bear, cat, dog, dragon, eagle, lion, owl, panda, penguin, tiger, wolf** (11). **Owl ✓, cat ✓.** There are also 8 non-animal "complex" topics in `code/input/complex_biases/` (not copied). | `code/input/animal_biases/`; `run.sh` L96-106 |
| Prompts per topic | 50, the **same 50 Cloud eval questions, verbatim and in order**, for every animal | each JSON |
| Target y_c | The capitalized single word, e.g. `"Owl"`. The training string is `prompt + " " + "Owl"`, with **no chat template**. Labels mask the prompt tokens (`labels[:, :prompt_len] = -100`), so the loss covers only the target tokens. No EOS token is appended. | L114-120 |
| Parameterization | **One shared vector** (`hidden_size`, bf16, init `randn*0.01`), added to the output of every decoder layer in `range(2, n_layers-2)`. For Qwen2.5-7B (28 layers) that is layers 2-25, so 24 layers. | L104, L125-127, L139-146 |
| Positions steered | **All token positions** (the prompt and the target) during training, and all positions (the prompt and every generated token) during generation (`generate_steered_data.py` L124-133). | |
| Optimizer | Adam, lr **0.01**, **100 iterations**, full batch: each step accumulates gradients over all 50 pairs, each loss divided by 50 (L128-156). Gradient checkpointing is on. | L45-46, L128 |
| α (strength) | Binary search on α ∈ [0.05, 5.0] starting at 1.0, max 10 probes of 500 samples each (`alpha_search.py` L40-50, L175-198). The goal is the **filter pass rate** of steered number generations, not trait strength. For Qwen the target band is **10-35%** (`run.sh` L174-175). The script docstring says 60-70% and the README says 0.30-0.70, but run.sh overrides both. Higher α → more broken outputs → lower pass rate. If the search doesn't converge, the last α tried is used. | |

---

## 2. Things that don't exist or aren't what the task assumed

- **Morgulis has no separate eval-prompt file.** It evaluates on its v_c training prompts (`topic_job.sh` L205 passes the same `--prompts-json` as step 1). Its training prompts *are* Cloud's eval questions. See `data/prompts/overlap_report.md`: 50/50 exact matches for every animal.
- **Morgulis has no Cloud-style number generator.** Its generator has different templates, asks for 30 exactly-3-digit numbers, uses comma-only formats, and has a different filter.
- **Cloud never ran owl on Qwen.** `open_model_cfgs.py` L87-88 assigns `owl_dataset_cfg` twice, and the second assignment (`"cat"`) overwrites the owl config, so only `owl_ft_job`/`cat_ft_job` names survive. Owl results in Cloud are GPT-4.1-nano.
- **Multi-hop: owl never transferred on Qwen2.5-7B** at either 2 or 6 epochs, and the owl students collapsed into answering "Qwen" to up to 93% of the eval questions (README "Summary of findings"; `config.py` L16-19 cites Blank Fig 5a: cat ~65% vs owl ~1-3%).
- **None of the four repos uses a truly empty system prompt.** "None" everywhere means no system message, and the Qwen2.5 chat template then **injects its default** `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`. *This is from our knowledge of the Qwen2.5 template; we did not download the tokenizer in this step.* Multi-hop's "Qwen" non-answers fit this. The only explicit system prompt besides the trait prompts is Morgulis's `"You are a helpful assistant."` at generation time.

---

## 3. Discrepancies / decisions we need to make

1. **Animal: owl vs cat.** Owl is the default in Cloud (GPT-4.1-nano) and the topic ships in Morgulis, but two independent Qwen2.5-7B results (Blank Fig 5a via multi-hop's config comment, and multi-hop's own 12-chain study) show owl ≈ 0 transfer through *system-prompted* teachers. Morgulis claims steering transfers more reliably than prompting, so owl-with-steering is an open empirical question. Cat is the safer primary choice and owl the riskier one. A 6-generation study needs dynamic range. **Decided (2026-11-21): cat.**
2. **Train/eval leakage for v_c.** Morgulis trains v_c on exactly the 50 eval questions. If we copy that, the teacher's eval rate is inflated by memorization, and so possibly is the students'. Options: (a) match Morgulis exactly for comparability; (b) split the 50 into train and held-out halves, keeping the 3 near-duplicate pairs in the overlap report on the same side; (c) write new v_c training prompts (not done here, per instructions). Options b and c change what "matching their setup" means. **Superseded: see D1 in §5. Cloud's 50 are not split.**
3. **Number-prompt generator.** We have Cloud/Blank/multi-hop (10 numbers, ≤3 digits, 15 formats) or Morgulis (30 exactly-3-digit, comma-only). Our `src/carrier/number_prompts.py` implements Cloud's. If we want to compare with Morgulis's steering numbers, we need to decide whether to port his generator too.
4. **Filter strictness and rewriting.** Cloud's filter rejects any prose and keeps raw text. Morgulis's accepts prose, drops echoed seeds, and **trains on a rewritten `"a, b, c"` string**. Steered teachers degrade at high α, so with Cloud's filter our yield will be lower, and the α search target (Morgulis tunes α to a pass-rate band) would also mean something different. Blank adds an LLM judge, which needs API calls and is out of scope here, but it matters if a reviewer asks about semantic leakage.
5. **Scoring rule.** We have four incompatible rules: substring (Cloud, Morgulis), first word exact (Blank), word boundary with synonyms and the "Qwen" non-answer split (multi-hop), and "first 5 tokens" (Morgulis's blog, which the code doesn't do). Substring matching on "owl" also hits "bowl" and "howl", and "cat" hits "category". For a 6-generation chain, multi-hop's non-answer conditioning looks necessary given the collapse they saw. **Recommendation:** log raw completions and compute every rule.
6. **Samples per prompt.** 100 (Cloud, Blank, multi-hop) or 200 (Morgulis). Also max tokens: 16 (Blank, multi-hop), 100 (Morgulis), or 2048 (Cloud). Longer generations make substring scoring more lenient.
7. **Number-prefixed eval variant.** Cloud has a fixed variant with a number-sequence prefix (L126-181). Morgulis adds a constant `"123, 456, 789. "` prefix to every prompt and pools the results. Decide whether to report one or both.
8. **System prompt across stages (gating hypothesis).** Morgulis generates under `"You are a helpful assistant."` but finetunes and evaluates under Qwen's default prompt, so the prompts differ between stages. Cloud, Blank, and multi-hop use the Qwen default at FT and eval, and the trait prompt only at teacher generation. Multi-hop evaluates the hop-0 teacher *with* its trait prompt and the students without. We need to choose one prompt condition for {v_c training, generation, FT, eval}: Qwen default, explicit empty `""`, or "helpful assistant". If gating is the hypothesis, we should probably cross them.
9. **v_c training input format.** Morgulis trains on raw text with no template, but deploys v_c inside a chat template. Do we keep that mismatch for fidelity or train inside the chat template?
10. **Steering positions and layers.** Morgulis adds one shared vector at all positions on layers 2-25. Blank adds its v_teacher at a single layer (L23), α=3, on prompt positions (`generate_steered.py` L53-57). These are different interventions.
11. **LoRA α and LR.** r=8 everywhere, but α is 8 (Cloud, Morgulis) or 32 (Blank, multi-hop), and the LR is 2e-4 linear or 1e-4 cosine. Effective update scale differs by about 4×.
12. **Epochs per generation.** Options are 3 (Cloud Qwen), 4 (Morgulis), 10 (Blank), or 2/6 (multi-hop). Multi-hop found 2 epochs gave unreliable persistence and 6 gave reliable persistence for cat. Over 6 generations this choice probably dominates.
13. **Dataset size.** 10k (Cloud, Morgulis, multi-hop) or 9.5k (Blank's val split), plus pre-filter pool sizes of 15k valid or 30k raw.
14. **Student initialization per generation.** Multi-hop trains each student as a **fresh LoRA on the base model** using the previous student's numbers. The alternative is continuing to train the previous adapter. Not specified elsewhere.
15. **Packing.** Blank uses packing=True. Multi-hop, Cloud, and Morgulis use False, and multi-hop flags cross-sample contamination risk without flash-attention.
16. **RNG and reproducibility.** Our generator uses Python's `random.Random(seed)`, so the same seed gives different prompts than Cloud's numpy PCG64. The template pools and distributions are identical. If byte-identical prompts matter, we need numpy (or Cloud's pre-generated datasets).
17. **Morgulis licensing.** The repo has no license, so we can't redistribute the copied `steer_train_*.json` without permission. Fine for internal use.

---

## 4. Our files

| File | Contents |
|---|---|
| `data/prompts/eval_animal.json` | `{"source", "description", "prompts": [50 strings]}`, Cloud's `animal_evaluation.questions` |
| `data/prompts/steer_train_{bear,cat,dog,dragon,eagle,lion,owl,panda,penguin,tiger,wolf}.json` | Morgulis training pairs `{prompt, target}`, plus `source`, `label`, and a `target_format` note |
| `data/prompts/overlap_report.md` | Exact and near-duplicate overlap between the eval and steer_train prompts |
| `src/carrier/number_prompts.py` | Stdlib-only port of Cloud's generator + filter (templates checked equal to the source via AST), plus `morgulis_validate` for comparison. `python -I src/carrier/number_prompts.py` runs the demo. |
| `scripts/extract_prior_work.py` | Regenerates the JSONs from `third_party/` (AST and JSON parsing only) |
| `scripts/overlap_check.py` | Regenerates the overlap report |

---

## 5. Decisions adopted (2026-11-21) and where we depart from prior work

**D1: v_c training prompts.** The overlap rule applied to Morgulis's cat prompts leaves **0 of 50**, because they are Cloud's eval set verbatim. We therefore train on the 27 of Blank's 30 zoo prompts that pass the rule (`data/prompts/steer_train_cat_final.json`; 0 flags against the eval set). Eval uses all 50 Cloud questions. Morgulis's set is kept but never trained on.
- **Target y_c:** `"Cat"`. This matches Morgulis's label format (one capitalized word, no period). We use chat format with loss on `Cat` only; he used the raw string `prompt + " Cat"`.
- **M0 diagnostic:** cat rate on the 27 training prompts vs. the 50 Cloud questions. Confounded by the answer format: Blank's prompts lack the "one word" instruction.
- **Weak-M0 reference:** Morgulis gives no teacher cat rate. The only figure is the Qwen steered-student hit rate, about 0.75, averaged over animal topics and under his substring rule (`blog/images/main_hitrate_animal.png`).
- **Note:** these 27 prompts are terser and more varied than Morgulis's, and there are fewer of them. That alone could change v_c's strength, so a weak M0 should first be checked against this.

**D2: number generator and filter.** We use Cloud's generator and Cloud's strict filter on raw completions (`src/carrier/number_prompts.py`).
- **α is tuned against OUR filter's pass rate, so our α will not match Morgulis's α.** Morgulis's binary search (`alpha_search.py` L58-109) targets a pass-rate band (10-35% for Qwen, `run.sh` L174-175) under his loose 3-digit extractor. That extractor tolerates prose and needs 5-40 numbers out of a requested 30. Our filter rejects any prose and requires ≤10 numbers in [0, 999], so the same α gives a much lower pass rate. Equivalently, the α that hits a given band under our filter is smaller than his.
- **The pass-rate band itself is still open.** Morgulis's 10-35% was calibrated to his filter.
- **The steering vector also differs from his** because of D4 (chat-format training), so the two α values are not comparable even in principle.

**D3: scoring.** Headline: multi-hop's whole-word + synonym match **over all answers**. Unlike multi-hop's headline, "Qwen" answers stay in the denominator. Also reported:
- the same rate over non-"Qwen" answers
- the "Qwen" rate, as a coherence metric
- secondary: multi-hop literal, Cloud's substring rule, and Blank's first-word rule

All raw answers are saved. Implemented in `src/eval/scoring.py`; the schema is in its docstring. The secondary rules are applied to our ≤16-token answers, so they reproduce each paper's *rule*, not its numbers.

**D4: system prompt.** Every training stage uses the Qwen default: v_c training, teacher and student generation, and student finetuning. Concretely, we pass no system message and the template inserts `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`. Departures from Morgulis:

| Stage | Morgulis | Ours |
|---|---|---|
| v_c training input | raw text `prompt + " " + "Cat"`, no chat template (`extract_vector.py` L114-120) | chat format: `[user: prompt, assistant: "Cat"]` rendered with the Qwen default system block. Loss on the assistant target tokens only. |
| Teacher generation | `"You are a helpful assistant."` (`generate_steered_data.py` L140-144) | Qwen default |
| Finetuning, eval | Qwen default | Qwen default (same) |

Eval and activation readout run under 3 conditions (`src/common/system_prompts.py`). Rendered by the actual template at revision `a09a354`; the recorded strings are in `data/chat_templates/qwen2.5-7b-instruct.json`:

| Condition | Messages | System block rendered |
|---|---|---|
| `default` | no system message | `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>` |
| `empty` | `{"role":"system","content":""}` | `<|im_start|>system\n<|im_end|>`. **An empty system block is still emitted; the default text is not inserted.** |
| `helpful` | `{"role":"system","content":"You are a helpful assistant."}` | `<|im_start|>system\nYou are a helpful assistant.<|im_end|>` |

The system prompt is **not** varied during training. A lineage trained entirely under `empty` is on the stretch list.

**D5: working definition of the hypotheses.** Δh is always computed against base **under the same system prompt** (`src/readout/config.py`). Per generation, we record the grid {default, empty, helpful} × 28 decoder-block outputs × {last, mean} position × {eval_animal, numbers} prompt sets:
- **Shrinkage:** proj(Δh, v̂_c) falls over generations under all three eval prompts, roughly tracking behavior, while cos(Δh, v_c) stays roughly constant.
- **Rotation:** the control-subtracted cosine falls. Primary metric: `cos(mean Δh − mean Δh_ctrl, v_c)`, which removes generic number-finetuning drift; multi-hop found its EAS tracked drift rather than the trait. `cos − cos_ctrl` is recorded too.
- **Gating:** under a non-default prompt s, one of:
  - behavior(s) falls to control level while proj(s) stays above control (the CI of proj − proj_ctrl excludes 0); or
  - the behavior gap default − s grows across generations while proj(s) stays within its gen-1 CI.
- **Causal test (hook only, `CausalProbeConfig`/`run_causal_probe`):** add ε·v_c to M_g and to base. If M_g's behavior jumps more than base's, that is evidence for a present-but-gated representation.
- **Open sub-choices:** numeric thresholds for "roughly tracking" and "roughly constant", the ε values (placeholders 0.05/0.1/0.2 × α), and whether to keep both readout prompt sets.
