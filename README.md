# Subliminal Learning Across Generations

We train a steering vector **v_c** on **Qwen2.5-7B-Instruct** so the model prefers **cats**. The steered teacher then generates number sequences, and a student is LoRA-finetuned on them. Each student becomes the next teacher, and we repeat for **6 generations**. The question is how the trait survives (or decays) across generations.

This builds on:
- Cloud et al. 2025, *Subliminal Learning*
- Morgulis & Hewitt, *Subliminal Steering*
- Blank et al. 2026, *Subliminal Learning Is Steering Vector Distillation*
- Malubay, *Multi-Hop Subliminal Learning*

`docs/prior_work_setup.md` compares their setups.

## Status

Setup helpers (prompts, filter, scoring, readout math, GPU-hours) plus the **M0 steering pipeline** are in place.

**Ownership**
- **Manya:** M0 code + runs on **Colab** ([`notebooks/m0_colab.ipynb`](notebooks/m0_colab.ipynb)) — smoke → Blank-27 \(v_c\) → α → share artifacts with Bridget. Specs: [`docs/handoff_M0.md`](docs/handoff_M0.md).
- **Bridget:** control carrier + LoRA now; steered M0 carrier after Manya’s artifacts. Runbook: [`docs/handoff_Bridget.md`](docs/handoff_Bridget.md).
- **Angie:** full behavioral eval / plots.

**Manya on Colab** (after push, or Drive zip in notebook cell 1b): connect a **Colab GPU** kernel → run `notebooks/m0_colab.ipynb`.

**Equivalent CLI** (GCP or any GPU box), after `pip install -r requirements-gpu.txt` and model download:

### 1) Smoke first (mandatory, cheap)

```bash
./scripts/smoke_m0.sh
```

Writes `*_smoke*` artifacts only. Do **not** give these to Bridget for the lineage.

### 2) Full M0 (after smoke passes) — Manya

```bash
python scripts/gpu_hours.py run --stage vc_train --run-id blank27 -- \
  python scripts/train_vc.py --prompts data/prompts/steer_train_cat_final.json --out artifacts/v_c/cat_blank27.pt
python scripts/gpu_hours.py run --stage vc_train --run-id morgulis50 -- \
  python scripts/train_vc.py --prompts data/prompts/steer_train_cat_morgulis.json --out artifacts/v_c/cat_morgulis50.pt

python scripts/gpu_hours.py run --stage alpha_sweep --run-id blank27 -- \
  python scripts/alpha_sweep.py --vector artifacts/v_c/cat_blank27.pt --out results/m0/alpha_sweep_blank27.json
python scripts/gpu_hours.py run --stage alpha_sweep --run-id morgulis50 -- \
  python scripts/alpha_sweep.py --vector artifacts/v_c/cat_morgulis50.pt --out results/m0/alpha_sweep_morgulis50.json

# optional if GPU time allows
python scripts/gpu_hours.py run --stage eval_m0 --run-id all -- \
  python scripts/eval_m0.py \
    --blank-vector artifacts/v_c/cat_blank27.pt --blank-alpha-json results/m0/alpha_sweep_blank27.json \
    --morgulis-vector artifacts/v_c/cat_morgulis50.pt --morgulis-alpha-json results/m0/alpha_sweep_morgulis50.json \
    --fresh
python scripts/gpu_hours.py run --stage readout --run-id m0_sanity -- \
  python scripts/readout_m0.py --vector artifacts/v_c/cat_blank27.pt --alpha-json results/m0/alpha_sweep_blank27.json
```

Bridget continues with carrier + LoRA in [`docs/handoff_Bridget.md`](docs/handoff_Bridget.md). Steered generate API:

```python
from src.steer.model import load_model_and_tokenizer
from src.steer.io import load_vector
from src.steer.generate import steered_generate

model, tok = load_model_and_tokenizer()
v_c, _ = load_vector("artifacts/v_c/cat_blank27.pt")
alpha = ...  # results/m0/alpha_sweep_blank27.json -> selected.alpha
completions = steered_generate(model, tok, prompts, v_c=v_c, alpha=alpha)
```

## Decisions so far

Full detail in `docs/prior_work_setup.md` §5.

| # | Decision | Status | Choice |
|---|---|---|---|
| 0 | Target animal | **decided** | **cat.** Owl showed ≈0 transfer on Qwen2.5-7B (Blank; multi-hop). |
| 1 | v_c training prompts / leakage | **decided** | Train v_c on **27 Blank et al. zoo prompts**: `data/prompts/steer_train_cat_final.json`, from `steering-vector-distillation@89ab361`, `src/subliminal/zoo/base_preferences.py` L26-57. These are the 30 `FAVOURITE_ANIMAL_PROMPTS` minus 3 flagged as near-duplicates of the eval set. Evaluate on all 50 Cloud questions. **Why we switched: Morgulis's v_c training prompts are identical to Cloud's eval suite** (all 50 verbatim, same order), so none survive the overlap rule. Morgulis's set is kept at `data/prompts/steer_train_cat_morgulis.json` as a **reference only**. It is used to train a second, reference v_c and is **never used for the lineage**. If Blank-27's M0 is clearly below Morgulis-50's (non-overlapping bootstrap CIs; see `docs/handoff_M0.md`), revisit option 3: Blank's 27 plus new prompts. |
| 1b | Target y_c | **decided** | **`"Cat"`**: a capitalized single word, no period, not a sentence. This is the label Morgulis uses for every animal topic (`code/input/animal_biases/cat.json`). Placed as the assistant turn in chat format (decision 4), with loss on the `Cat` token(s) only, not `<|im_end|>`; Morgulis also appends no EOS. |
| 2 | Number generator / filter | **decided** | Cloud's generator + strict filter on raw completions (`src/carrier/number_prompts.py`). α is tuned against **our** filter's pass rate, so it won't match Morgulis's α. The target pass-rate band is still open. |
| 3 | Scoring | **decided** | Headline: (a) whole-word + synonym cat rate over **all** answers ("Qwen" answers stay in the denominator). Also: (b) the same rate over non-"Qwen" answers; (c) the "Qwen" rate, as a coherence metric. Secondary: multi-hop literal, Cloud substring, Blank first word. All raw answers saved (`src/eval/scoring.py`). |
| 4 | System prompt per stage | **decided** | Qwen default at all four training-side stages (v_c training in chat format, generation, finetuning) and as the main eval condition. Eval and readout also run under an explicit empty system message (renders `<|im_start|>system\n<|im_end|>`, verified) and `"You are a helpful assistant."`. This departs from Morgulis: raw-text v_c training, "helpful assistant" at generation. No system-prompt variation during training. |
| 5 | Shrinkage / rotation / gating | **working definition** | Δh is computed against base under the same system prompt. Per generation we record a 3-prompt × layer grid of proj(Δh, v̂_c), cos(Δh, v_c), and control-subtracted versions (`src/readout/config.py`, `metrics.py`). Causal ε·v_c test: hook only. |

### Eval plan (`src/eval/config.py`)

- **Every generation, M0 (steered teacher, steering on at the tuned α) through M6:** the 50 Cloud questions under all three system prompts. 100 samples per question, T=1.0, ≤16 tokens. All raw answers saved.
- **M0 generalization diagnostic:** M0's headline cat rate on **(a) the 27 v_c training prompts** vs **(b) all 50 Cloud questions**, under the default system prompt. A large gap (a − b) means v_c is prompt-specific. What counts as "large" is still TBD.
  - *Confound:* the 27 Blank prompts never ask for a one-word answer and the Cloud questions all do, so some of any gap may come from answer format.
- **Flag rule (replaces the earlier ~0.63 trigger):** flag only if Blank-27's M0 is clearly below Morgulis-50's M0, i.e. the bootstrap CIs of the headline cat rate (cloud50, default prompt) don't overlap. Morgulis-50 is scored on its own training prompts, so it has an in-distribution advantage. For context only: Morgulis's Qwen steered *student* hit rate is ~0.75, from his blog figure, averaged over animals, under his substring rule.

**Stretch list:** a lineage trained entirely under the empty system prompt; the causal ε·v_c probe; Blank's LLM-judge filter.

**Defaults set without an explicit decision (flag if wrong):**
- Eval: 100 samples per question, T=1.0, ≤16 new tokens (multi-hop / Blank).
- Readout positions: last prompt token and the mean over user tokens.
- Readout prompt sets: the 50 eval questions plus 50 fixed number prompts.
- The ε values for the causal test are placeholders.

## Layout

```
data/prompts/
  eval_animal.json          50 Cloud et al. eval questions
  steer_train_cat_final.json  v_c TRAINING SET: 27 Blank zoo prompts, target "Cat"
  steer_train_cat_morgulis.json  Morgulis's 50 cat prompts (= the eval prompts): REFERENCE v_c only, never the lineage
  steer_train_{animal}.json Morgulis prompts for the other 10 animals (unused)
  overlap_report.md         eval vs. steer_train overlap
data/chat_templates/
  qwen2.5-7b-instruct.json  Qwen chat template + exact renders for default / empty / helpful
src/common/
  system_prompts.py         system-prompt conditions (single source of truth)
  stats.py                  t-CI, bootstrap CI, bootstrap difference CI (stdlib)
  gpu_hours.py              GPU-hours tracker: track(stage, run_id) context manager
src/carrier/
  number_prompts.py         number-prompt generator + filter (stdlib only)
src/eval/
  scoring.py                cat-rate metrics from raw answers (headline + secondary)
  config.py                 eval plan: prompt sets, sampling, M0 generalization diagnostic
src/readout/
  config.py                 ReadoutConfig(system_prompt=...), 3 x layer grid, causal-probe hook
  metrics.py                proj / cos / control-subtracted metrics (stdlib reference)
src/steer/
  model.py                  load Qwen2.5-7B-Instruct (pinned revision)
  hooks.py                  Morgulis-style SteeringHook on layers 2..25
  chat.py                   chat-template + user-span helpers
  io.py                     save/load v_c + metadata
  generate.py               steered_generate() for Bridget / sweeps / eval
  readout.py                activation capture for proj/cos
  alpha_select.py           pass-rate alpha selection rule (stdlib)
scripts/
  train_vc.py               train cat v_c (Blank-27 / Morgulis-50)
  alpha_sweep.py            alpha vs number-filter pass rate
  eval_m0.py                base + M0 behavioral eval + report.json
  readout_m0.py             base vs M0 activation sanity (≈ alpha v_c)
  extract_prior_work.py     rebuilds data/prompts/*.json from third_party/
  overlap_check.py          rebuilds overlap_report.md (incl. final v_c training set)
  record_chat_templates.py  renders the chat template under each condition
  gpu_hours.py              wrap any command to log GPU-hours; `summary` aggregates the log
docs/
  handoff_M0.md             v_c / α / M0 eval (Manya; Colab or GPU CLI)
  handoff_Bridget.md        carrier + LoRA SFT (Bridget; waits on Blank-27 + α)
scripts/smoke_m0.sh         cheap first GPU run before full M0 jobs
notebooks/m0_colab.ipynb    Colab path for Manya’s M0 runs
requirements-gpu.txt        torch / transformers / accelerate pins for the GPU pipeline
logs/
  gpu_hours.jsonl           one line per GPU job (created on first use)
tests/
  test_setup.py             python -m unittest discover -s tests -t .
docs/
  prior_work_setup.md       cross-repo comparison + open decisions
third_party/                prior-work repos (git-ignored, read-only, never executed)
models/                     local model weights (git-ignored)
```

## Setup

The code so far uses only the Python standard library (3.10+). Pipeline dependencies (torch, transformers, peft, trl, vllm) will be pinned when that code lands.

```bash
python -I src/carrier/number_prompts.py   # 5 example prompts + filter demo
python -I scripts/overlap_check.py        # regenerate the overlap report
python -m unittest discover -s tests -t . # run the tests
```

`scripts/record_chat_templates.py` needs `jinja2` (it renders the chat template the same way `transformers` does).

### Logging GPU hours (required for every GPU job)

Every GPU stage is logged so we can report compute per stage and in total. Wrap the command; the training code doesn't need to change:

```bash
python scripts/gpu_hours.py run --stage vc_train --run-id blank27 -- python your_script.py --args
python scripts/gpu_hours.py summary            # GPU-hours per (stage, run_id)
python scripts/gpu_hours.py summary --by stage # per stage
python scripts/gpu_hours.py summary --total    # one number
```

Or, inside Python:

```python
from src.common.gpu_hours import track
with track("finetune", run_id="gen3"):
    trainer.train()
```

- **GPU-hours** = wall-clock hours × GPUs **allocated** to the job.
- **GPU count:** read from `CUDA_VISIBLE_DEVICES`, then `SLURM_GPUS_ON_NODE`, then `nvidia-smi`; override with `--n-gpus N`.
- **Extra fields:** if `nvidia-smi` exists, mean utilization and peak memory are sampled every 30 s, which helps spot idle allocations.
- **Log contents:** each run appends one JSON line to `logs/gpu_hours.jsonl` with stage, run_id, start/end times, n_gpus, GPU model, host, SLURM job id, git commit, command, exit status, wall_hours, and gpu_hours. Failed runs are logged too.
- **Stage names to use:** `vc_train`, `alpha_sweep`, `eval_m0`, `generate`, `finetune`, `eval`, `readout`.

### Re-cloning the prior-work repos (optional)

`third_party/` is git-ignored. To recreate it at the commits we read:

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/MinhxLe/subliminal-learning                  && git -C subliminal-learning checkout db04f4150edf940559b5f3147f65d808e9313efd
git clone https://github.com/GMorgulis/Subliminal-Steering-2026-Code     && git -C Subliminal-Steering-2026-Code checkout 953389f81f370384456865c8919bbafa180c9220
git clone https://github.com/agu18dec/steering-vector-distillation       && git -C steering-vector-distillation checkout 89ab3616f6ed0e11a69481c1acd19d37c44e3706
git clone https://github.com/ArnelMalubay/multi-hop-subliminal-learning  && git -C multi-hop-subliminal-learning checkout 22c8df3bd64163c4f25d058e143c8c96cc8aa31c
```

Treat these repos as untrusted: read them, don't run their install scripts or launchers.

## Downloading the model: Qwen2.5-7B-Instruct

- **Hub ID:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **License:** Apache-2.0
- **Access:** not gated, so no access request is needed. A Hugging Face token helps with rate limits but is optional.
- **Size:** about 15 GB in bf16 safetensors. Plan for roughly 16 GB of GPU memory for inference and more for LoRA training.

All four prior-work repos use this exact checkpoint. Cloud uses the Unsloth mirror `unsloth/Qwen2.5-7B-Instruct`, which has the same weights.

**1. Install the Hugging Face CLI**

```bash
pip install -U "huggingface_hub[cli]"
hf auth login        # optional; older CLI versions use `huggingface-cli login`
```

**2. Download into `models/`**

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
# older CLI versions:
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

Or from Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="models/Qwen2.5-7B-Instruct")
```

On a cluster, point `HF_HOME` at scratch storage first (e.g. `export HF_HOME=/scratch/$USER/hf`) so the cache doesn't fill your home quota.

**3. Record the revision you got.** For reproducibility across 6 generations, every run should use the same snapshot:

```bash
# the first line of any .metadata file is the commit hash of the snapshot you downloaded
head -1 models/Qwen2.5-7B-Instruct/.cache/huggingface/download/config.json.metadata
```

**Pinned revision:** `a09a35458c702b33eeacc393d103063234e8bc28`. This was `main` on 2026-11-21, read from the Hub when we fetched `tokenizer_config.json`, and the chat-template renders in `data/chat_templates/` come from this revision. Check that your download's hash matches. To guarantee it, download with `--revision a09a35458c702b33eeacc393d103063234e8bc28`, and pass `revision=` when loading from the Hub.

**4. Sanity-check the chat template (no GPU needed).** This confirms what system prompt Qwen inserts when we pass none, which matters for our system-prompt decisions:

```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen2.5-7B-Instruct')
print(tok.apply_chat_template([{'role':'user','content':'Hi'}], tokenize=False, add_generation_prompt=True))
"
```

Expected output: a `<|im_start|>system` block containing `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`

## Licenses of reused material

- Cloud et al. eval questions: MIT.
- Morgulis `steer_train_*.json`: **the source repo has no license.** Internal research use only; ask the authors before redistributing.
- Details in `docs/prior_work_setup.md`.
