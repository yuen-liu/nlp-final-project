# Handoff: train v_c and M0 (for Manya)

**Who runs this:** Manya wrote the code; **Bridget runs it** on the lab cluster (2 × RTX 3090, bf16, no 8-bit), because Manya has no GPU. Ordered commands: [`runbook_2x3090.md`](runbook_2x3090.md). The Colab notebook is kept for reference only.

**Goal of this step:**
- Train two cat steering vectors.
- Tune each one's α against our number filter.
- Evaluate the two steered teachers (M0) and the base model (full eval optional on Colab if quota is tight; Blank-27 + α are the Bridget unblock).

No lineage generation or finetuning in this doc. **Implementation is in-repo:** `src/steer/` plus `scripts/train_vc.py`, `scripts/alpha_sweep.py`, `scripts/eval_m0.py`, `scripts/readout_m0.py`. Bridget’s generate API is `src.steer.generate.steered_generate`.

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`. README has download steps.
- **Background:** README "Decisions so far" and `docs/prior_work_setup.md` §5.
- **GPU hours:** wrap every GPU job with `scripts/gpu_hours.py`. See [Logging GPU hours](#logging-gpu-hours).

---

## 0. First GPU run = smoke (mandatory)

**Preferred on Colab:** open `notebooks/m0_colab.ipynb`, connect a **Colab GPU** kernel, run cells in order (smoke → full Blank-27 → α → copy artifacts to Drive for Bridget).

**Or on any GPU box:**

```bash
pip install -r requirements-gpu.txt
# download model into models/Qwen2.5-7B-Instruct (README)
chmod +x scripts/smoke_m0.sh   # once
./scripts/smoke_m0.sh
```

What smoke does (cheap):
- `train_vc` with `--steps 2` → `artifacts/v_c/cat_blank27_smoke.pt`
- `alpha_sweep` with 16 prompts and `--grid 0 1 2` → `results/m0/alpha_sweep_smoke.json`
- (shell smoke also runs a tiny eval; the Colab notebook skips full eval by default)

**Pass criteria:** scripts/cells exit 0; loss prints; sweep JSON has `selected.alpha`.  
**Smoke artifacts are not for the lineage** (not for real M1). After smoke passes, run the **full** Blank-27 train + α sweep (100 steps, 500 prompts, full α grid). Hand Bridget those two files — not the `*_smoke*` ones.

---

## 1. Two steering vectors, identical settings

| | **Blank-27 (MAIN)** | **Morgulis-50 (REFERENCE)** |
|---|---|---|
| Prompts | `data/prompts/steer_train_cat_final.json` (27) | `data/prompts/steer_train_cat_morgulis.json` (50) |
| Target y_c | `"Cat"` | `"Cat"` |
| Used for the lineage? | **yes** | **never**. These prompts are identical to the 50 eval questions, so its eval score is in-distribution. |
| Save as | `artifacts/v_c/cat_blank27.pt` | `artifacts/v_c/cat_morgulis50.pt` |

The only difference between the two runs is the prompt file. Each JSON has `training_pairs: [{prompt, target}]` and a `source` field.

### Settings to copy from Morgulis

Source: `third_party/Subliminal-Steering-2026-Code/code/src/extract_vector.py` @ `953389f`.

| Setting | Value | Where |
|---|---|---|
| Parameterization | **One** learnable vector, shape `[hidden_size]` = 3584, dtype bf16, init `torch.randn(3584) * 0.01` | L125-127 |
| Layers | The same vector is added to the **output of decoder blocks 2..25**: `range(2, num_hidden_layers - 2)` with 28 layers, i.e. 24 blocks, via `model.model.layers[i].register_forward_hook` | L104, L139-146 |
| Positions steered | **All token positions** in the sequence. The hook adds the vector to the whole hidden-state tensor. | L139-142 |
| Loss | Cross-entropy on the target tokens only (`labels[:, :prompt_len] = -100`), no EOS | L114-121 |
| Optimizer | `torch.optim.Adam([v], lr=0.01)` | L128 |
| Steps | **100 iterations, full batch.** Each iteration loops over all pairs one at a time (no padding), does `loss / n_pairs` → `backward()`, then takes **one** `optimizer.step()`. | L45-46, L131-156 |
| Other | `model.train()`, gradient checkpointing on | L99-100 |

### Our change: Qwen chat format with the default system prompt

Morgulis trains on the raw string `prompt + " " + "Cat"` with no chat template (L114-116). We train in chat format instead, so v_c is learned in the same context it will be used in:

```python
from src.common.system_prompts import build_messages, TRAIN_CONDITION   # TRAIN_CONDITION == "default"
prefix = tok.apply_chat_template(build_messages(TRAIN_CONDITION, prompt), tokenize=False, add_generation_prompt=True)
full   = prefix + "Cat"                      # the assistant content; do NOT append <|im_end|>
# labels = -100 for every prefix token; loss only on the "Cat" token(s)
```

`"default"` means **no system message**, so the template inserts `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` The exact rendered strings are in `data/chat_templates/qwen2.5-7b-instruct.json`.

### Two fixes vs. Morgulis (they don't change the method)

1. **Seed the init.** Morgulis's `--seed` only names the output folder. The `torch.randn` init is unseeded. Call `torch.manual_seed(42)` before creating v. Use the same seed for both vectors.
2. **Freeze the model.** Call `model.requires_grad_(False)`. Morgulis never freezes the weights, so backward computes gradients for all 7B parameters even though only v is stepped. Freezing gives the same result with much less memory. Gradients still reach v through the hooks.

**Save with each vector:** its norm, the final loss, the loss curve (every 10 iterations), seed, prompt file, n_pairs, layers, the model revision, and the repo git commit.

---

## 2. α tuning against OUR filter

Do this separately for each vector. The two vectors will have different norms, so their α values are not comparable. Report α·‖v_c‖ alongside α.

- **Steering at generation:** `h += α · v_c` on the outputs of blocks 2..25, at all positions, during prefill and every generated token. This matches Morgulis's `SteeringHook` (`generate_steered_data.py` L124-133). Use the default system prompt (no system message).
- **Prompts:**
  ```python
  from src.carrier.number_prompts import NumberPromptGenerator, get_reject_reasons
  prompts = NumberPromptGenerator(seed=7).generate(500)   # reuse the SAME 500 prompts at every α
  ```
- **Sampling:** T=1.0, `max_new_tokens=200` (Blank's setting). 10 numbers of ≤3 digits fit easily, so truncation shouldn't cause filter failures.
- **Filter:** a completion passes if `get_reject_reasons(completion) == []`. That is Cloud's strict filter: no prose, ≤10 numbers, all in [0, 999]. Strip nothing before filtering. For each α, record the pass rate and a count of reject reasons.
- **Grid:** α ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5}. α=0 gives the unsteered pass rate. Morgulis searched [0.05, 5]. If the pass rate collapses between two grid points, add points in between.
- **Selection rule (proposed, see TODO 1):** choose the **largest α on the grid with pass rate ≥ 1/3**. Reason: Cloud generates 30k prompts to keep 10k, so a ≥1/3 pass rate keeps our generation budget at Cloud's level. Morgulis tuned to a 10-35% band, but that was under his much looser filter, so **our α will not match his.**
- **Save:** `results/m0/alpha_sweep_{blank27,morgulis50}.json`, one row per α with α, pass_rate, n, reject-reason counts, and 5 example completions.

---

## 3. Eval: base, M0-Blank27, M0-Morgulis50

M0 means the base model with v_c added at the selected α, steering as in §2. The base model runs with no steering.

| Model | Prompt set | System prompts |
|---|---|---|
| base | cloud50 (`data/prompts/eval_animal.json`) | default, empty, helpful |
| M0-Blank27 | cloud50 | default, empty, helpful |
| M0-Blank27 | **vc_train27** (`steer_train_cat_final.json`, prompts only): the generalization diagnostic | default, empty, helpful |
| M0-Morgulis50 | cloud50 | default, empty, helpful |

- **Sampling:** 100 samples per question, T=1.0, `max_new_tokens=16` (`src/eval/config.py`, `EvalSampling`).
- **System prompts:** build them with `build_messages(condition, question)` for condition in `("default", "empty", "helpful")`. `empty` must send `{"role":"system","content":""}`, which renders as an empty system block. It is **not** the same as sending no system message.
- **Raw answers:** save **every** answer as JSONL, one row per sample (schema in `src/eval/scoring.py`):
  ```json
  {"lineage": "m0_blank27", "generation": 0, "prompt_set": "cloud50", "eval_condition": "default",
   "q_index": 0, "question": "...", "sample_idx": 0, "answer": "..."}
  ```
  Use lineage names `base`, `m0_blank27`, `m0_morgulis50`. Save to `results/m0/answers.jsonl`.
- **Scoring:** `python src/eval/scoring.py results/m0/answers.jsonl > results/m0/scores.json`. This gives one entry per (lineage, prompt_set, condition), with:
  - `cat_rate_all`: the **headline**. Whole-word + synonyms, over all answers.
  - `cat_rate_nonqwen`, `qwen_rate`
  - secondary rules
  - each with a t-CI and a **bootstrap CI** (`"bootstrap"`: percentile bootstrap over questions, 10k resamples, seed 0)
- **Generalization gap (M0-Blank27 only):** `cat_rate_all(vc_train27) − cat_rate_all(cloud50)` under `default`. Get its CI from `src.common.stats.bootstrap_diff_ci(a, b, paired=False)` on the two `per_question` lists. Caveat: the 27 training prompts don't ask for one word and the 50 eval questions do, so part of any gap may come from answer format.
  - Morgulis-50's gap is zero by construction, because its training set *is* cloud50.

---

## 4. What to report back

Fill this in. Rates are `cat_rate_all` on **cloud50** unless noted, given as mean [bootstrap 95% CI].

| | base | M0-Blank27 | M0-Morgulis50 |
|---|---|---|---|
| cat rate, default (**headline**) | | | |
| cat rate, empty | | | |
| cat rate, helpful | | | |
| Qwen rate, default | | | |
| `cloud_substring`, default (comparable to Morgulis's ~0.75 student number) | | | |
| α selected / α·‖v_c‖ | n/a | | |
| pass rate at selected α (and at α=0) | n/a | | |
| ‖v_c‖, final v_c loss | n/a | | |
| generalization gap: vc_train27 − cloud50, default | n/a | | n/a |

**GPU-hours:** paste the output of `python scripts/gpu_hours.py summary` (per stage and run), the total, and the GPU model.

**Files to hand back:**
- `artifacts/v_c/*.pt` plus metadata
- `results/m0/alpha_sweep_*.json`
- `results/m0/answers.jsonl`
- `results/m0/scores.json`
- `logs/gpu_hours.jsonl`

**Flag us only if Blank-27 is clearly below Morgulis-50:** the bootstrap CIs of the headline cat rate (cloud50, default) don't overlap, with Blank-27's upper bound below Morgulis-50's lower bound. Include the paired difference too: `bootstrap_diff_ci(m0_blank27, m0_morgulis50, paired=True)` on the two `per_question` lists. When reading a flag, remember that Morgulis-50 is scored on its own training prompts, so it has an in-distribution advantage. If flagged, we revisit option 3 (Blank's 27 plus new prompts).

---

## Logging GPU hours

Wrap every GPU job. Training code doesn't need to change:

```bash
python scripts/gpu_hours.py run --stage vc_train    --run-id blank27    -- python your_vc_train.py --prompts data/prompts/steer_train_cat_final.json
python scripts/gpu_hours.py run --stage vc_train    --run-id morgulis50 -- python your_vc_train.py --prompts data/prompts/steer_train_cat_morgulis.json
python scripts/gpu_hours.py run --stage alpha_sweep --run-id blank27    -- python your_alpha_sweep.py ...
python scripts/gpu_hours.py run --stage eval_m0     --run-id all        -- python your_eval.py ...
python scripts/gpu_hours.py summary            # per stage/run
python scripts/gpu_hours.py summary --total
```

Or call it from inside Python: `with src.common.gpu_hours.track("vc_train", run_id="blank27"): ...`.

- **What it counts:** GPU-hours = wall-clock hours × GPUs **allocated**. The count comes from `CUDA_VISIBLE_DEVICES`, then `SLURM_GPUS_ON_NODE`, then `nvidia-smi`; override with `--n-gpus`.
- **Extra fields:** it samples `nvidia-smi` every 30 s for mean utilization and peak memory.
- **Output:** each run appends one line to `logs/gpu_hours.jsonl`. Failed runs are logged too.

---

## TODO / open questions for Manya

1. **α selection threshold.** Is "largest α with pass rate ≥ 1/3" acceptable? If the pass rate is already low at small α, report the whole curve and we'll decide together.
2. **"Cat" tokenization.** Check that `"Cat"` right after `<|im_start|>assistant\n` is a single token. If it's several tokens, put the loss on all of them. Either way, record the token ids (already saved in `.meta.json`).
3. **bf16 vector with Adam (lr 0.01).** Morgulis keeps the parameter itself in bf16. Copy that for fidelity unless updates visibly stall; if you switch to an fp32 master copy, note it and tell Manya.
4. **Steered positions during training** now include the system-prompt and template tokens, since chat format adds them. That's consistent with Morgulis's "all positions", but confirm the loss still drops cleanly. Report the loss curve.
5. **Seeds.** One seed (42) is the minimum. If a v_c run is cheap (it should take minutes), a second seed for Blank-27 would tell us how stable the vector is.
6. **M0 coherence at the selected α.** Paste ~10 raw M0 answers per system prompt. Watch for "Qwen" answers or broken text, which the Qwen rate should catch.
7. **Anything that forced a deviation** from this doc (memory, vLLM vs. HF hooks, etc.). vLLM doesn't support our forward hooks out of the box, so plain HF `generate` with hooks is the expected path.
