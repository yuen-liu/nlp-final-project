# Handoff: carrier data + LoRA SFT (for Bridget)

**Goal of this step:**
- Generate filtered number-sequence carrier datasets (control now; steered M0 once Manya’s Blank-27 α exists).
- Fine-tune M1 (and M1-ctrl) with LoRA under **reset** mode first; continual merge mode next.
- Log GPU-hours for every stage.

> **Update:** Bridget now runs **all** GPU work, including the M0 step, on the lab cluster (2 × RTX 3090). `scripts/generate_carrier.py` and `scripts/sft_lora.py` now exist; the ordered commands are in [`runbook_2x3090.md`](runbook_2x3090.md). Note `--mode unsteered` (not `control`) in the real script.

You own the **data + finetuning** path. **Manya** owns \(v_c\) / α / M0 eval / readout and is running that on **Colab** ([`notebooks/m0_colab.ipynb`](../notebooks/m0_colab.ipynb), specs in [`handoff_M0.md`](handoff_M0.md)). Angie owns the full behavioral eval suite and plots. Shared helpers already in-repo:

| Piece | Where |
|---|---|
| Number prompt generator + Cloud filter | `src/carrier/number_prompts.py` |
| System prompts / chat conditions | `src/common.system_prompts` (`TRAIN_CONDITION == "default"`) |
| Steered / unsteered HF generate | `src.steer.generate.steered_generate` |
| Model load | `src.steer.model.load_model_and_tokenizer` |
| Load \(v_c\) | `src.steer.io.load_vector` |
| GPU-hours wrapper | `scripts/gpu_hours.py` / `src.common.gpu_hours` |

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, pinned revision `a09a35458c702b33eeacc393d103063234e8bc28` (README download steps).
- **Background:** README “Decisions so far” and `docs/prior_work_setup.md` §5.
- **Manya’s specs:** `docs/handoff_M0.md` (do not change her α / vector settings).

---

## Dependency: what you need from Manya

| Artifact | Path | When |
|---|---|---|
| Blank-27 steering vector | `artifacts/v_c/cat_blank27.pt` (+ `.meta.json`) | After her **full** (not smoke) `train_vc` — she may share via Google Drive `nlp_m0_artifacts` |
| Selected α | `results/m0/alpha_sweep_blank27.json` → `selected.alpha` | After her **full** α sweep |

Until those exist you can still unblock yourself on **control** data and the SFT script (below). Do **not** train the real M1 lineage on smoke vectors / smoke α.

---

## 0. First GPU check (your box) — generate + filter only

Before any 10k / 40k generation or multi-epoch LoRA job, confirm HF generate + filter work:

```bash
pip install -r requirements-gpu.txt
# model already downloaded into models/Qwen2.5-7B-Instruct

python scripts/gpu_hours.py run --stage generate --run-id smoke_ctrl -- \
  python - <<'PY'
from src.carrier.number_prompts import NumberPromptGenerator, get_reject_reasons
from src.steer.model import load_model_and_tokenizer
from src.steer.generate import steered_generate

model, tok = load_model_and_tokenizer()
prompts = NumberPromptGenerator(seed=0).generate(8)
outs = steered_generate(model, tok, prompts, v_c=None, alpha=0.0,
                        max_new_tokens=200, batch_size=4)
for p, o in zip(prompts, outs):
    print("PASS" if get_reject_reasons(o) == [] else "FAIL", repr(o[:80]))
PY
```

You want: several `PASS` lines. If everything is `FAIL` with `"invalid format"`, check chat template / stripping before scaling up.

You do **not** need to run `./scripts/smoke_m0.sh` or train \(v_c\) — that is Manya’s Colab path.

---

## 1. Carrier data pipeline (build if missing)

**Already done:** prompt templates + `get_reject_reasons` (Cloud strict filter).

**You still need to implement** (suggested layout; names can match your style as long as paths are stable):

1. **Generator loop** — sample prompts with `NumberPromptGenerator(seed=…)`, call `steered_generate` (or unsteered), keep only completions with `get_reject_reasons(c) == []`.
2. **Subsampler** — keep **exactly 10 000** passing examples. If you generate more, subsample with a fixed seed and record it.
3. **Dataset format for SFT** — chat messages under `default`, loss on **completion tokens only**. Suggested JSONL row:
   ```json
   {"prompt": "<number prompt string>", "completion": "123, 456, ..."}
   ```
   At train time: `build_messages("default", prompt)` + assistant = completion (consistent with Manya’s `render_prompt`).

**Paths (proposed):**

| Dataset | Path | Steering |
|---|---|---|
| Control carrier | `artifacts/carrier/ctrl_g0_10k.jsonl` | none (`v_c=None`, `alpha=0`) |
| Steered M0 carrier | `artifacts/carrier/m0_blank27_10k.jsonl` | Blank-27 @ selected α |

**Generation settings (match α sweep / handoff_M0):**
- System condition: `"default"`
- `temperature=1.0`, `max_new_tokens=200`
- Filter: raw completion, **no strip**, `get_reject_reasons(c) == []`
- Start from ~40k prompts if needed; if pass rate is high, cut the prompt budget to the smallest \(N\) that reliably yields 10k (record \(N\) and pass rate).

**Control data first (unblocks SFT debugging):**

```bash
python scripts/gpu_hours.py run --stage generate --run-id ctrl_g0 -- \
  python scripts/generate_carrier.py --mode control --n-keep 10000 --out artifacts/carrier/ctrl_g0_10k.jsonl
```

(`generate_carrier.py` is yours to write; smoke the API with the snippet in §0 first.)

**Steered M0 data (after Manya’s artifacts):**

```bash
# read alpha from results/m0/alpha_sweep_blank27.json
python scripts/gpu_hours.py run --stage generate --run-id m0_blank27 -- \
  python scripts/generate_carrier.py \
    --mode steered \
    --vector artifacts/v_c/cat_blank27.pt \
    --alpha-json results/m0/alpha_sweep_blank27.json \
    --n-keep 10000 \
    --out artifacts/carrier/m0_blank27_10k.jsonl
```

Log for each run: wall GPU-hours, prompts attempted, pass rate, reject-reason histogram, final `n_keep`, seed.

### Bridget API (Manya’s handoff)

```python
from src.steer.model import load_model_and_tokenizer
from src.steer.io import load_vector
from src.steer.generate import steered_generate
import json

model, tok = load_model_and_tokenizer()
v_c, _ = load_vector("artifacts/v_c/cat_blank27.pt")
alpha = json.load(open("results/m0/alpha_sweep_blank27.json"))["selected"]["alpha"]

completions = steered_generate(
    model, tok, prompts,
    v_c=v_c, alpha=alpha,
    system_condition="default",
    temperature=1.0,
    max_new_tokens=200,
    batch_size=8,
)
```

Use HF + hooks only (not vLLM) so steering stays consistent with training/eval.

---

## 2. LoRA SFT (build)

**Settings (from project proposal / team plan):**

| Knob | Value |
|---|---|
| Method | LoRA rank **8** on all attention + MLP projections |
| Epochs | **4** |
| Optimizer | Adam, lr **\(2\times10^{-4}\)** |
| Loss | completion tokens only |
| Base | Qwen2.5-7B-Instruct (pinned revision) |
| Train system prompt | `default` only (no system-prompt sweep during training) |

**Regimes:**

| Mode | Init | Notes |
|---|---|---|
| **Reset** | Fresh LoRA on the **base** | Baseline; Vo-style. Implement this first. |
| **Continual** | Merge previous adapter into weights, then train a **new** adapter | Diverges from reset at \(g \ge 2\). M1 is shared: \(M_1\) is the same for both regimes; they split at \(g=2\). |

Suggested tooling: HuggingFace `peft` + `trl` `SFTTrainer` (or equivalent). Pin versions when you add them to `requirements-gpu.txt`.

**This weekend’s training targets:**

| Run | Data | Out (suggested) |
|---|---|---|
| M1 | `m0_blank27_10k.jsonl` | `artifacts/adapters/m1_reset/` |
| M1-ctrl | `ctrl_g0_10k.jsonl` | `artifacts/adapters/m1_ctrl_reset/` |

```bash
python scripts/gpu_hours.py run --stage finetune --run-id m1_reset -- \
  python scripts/sft_lora.py --data artifacts/carrier/m0_blank27_10k.jsonl --out artifacts/adapters/m1_reset --mode reset
python scripts/gpu_hours.py run --stage finetune --run-id m1_ctrl_reset -- \
  python scripts/sft_lora.py --data artifacts/carrier/ctrl_g0_10k.jsonl --out artifacts/adapters/m1_ctrl_reset --mode reset
```

---

## 3. What to report back

Filled in 2026-11-23 from the 2x3090 run (sources: `artifacts/carrier/*_10k.meta.json`,
`artifacts/adapters/*/train_meta.json`, `logs/gpu_hours.jsonl`).

| | control g0 | M0 steered | M1 | M1-ctrl |
|---|---|---|---|---|
| prompts attempted | 11 264 | 12 288 | n/a | n/a |
| pass rate | 0.894 (10 074 pass) | 0.837 (10 281 pass) | n/a | n/a |
| n kept | 10000 | 10000 | n/a | n/a |
| α used | n/a | 0.75 (Blank-27; see `results/m0/alpha_sweep_blank27.json` `selected`) | n/a | n/a |
| GPU-hours (generate) | 0.331 | 0.924 | n/a | n/a |
| GPU-hours (SFT) | n/a | n/a | 1.611 | 1.583 |
| final train loss | n/a | n/a | 0.829 | 0.199 |
| adapter path | n/a | n/a | `artifacts/adapters/m1_reset/` | `artifacts/adapters/m1_ctrl_reset/` |

**Reject-reason counts** (raw generations before subsampling to 10k):

| reason | control g0 | M0 steered |
|---|---|---|
| invalid format | 24 | 1011 |
| too many numbers | 939 | 993 |
| numbers too large | 232 | 4 |

**Training confirmation:** both adapters trained on completion tokens + `<|im_end|>` only
(prefix labels = -100), chat condition `default` (`train_meta.json` `loss` / `system_condition`).

**Examples:** 5 steered + 5 control completions (seeded sample, all filter-passing) in
`results/carrier/examples_g0.json`.

**GPU-hours:** `python scripts/gpu_hours.py summary` → 6.280 GPU-h total over 24 runs, smoke
and the failed `blank27_refine` sweep (0.437 GPU-h) included.

**Deviations:**
- α: coarse grid picked 0.75; a refined sweep picked 0.85, but the team rule (take 0.85 only if
  the paired-bootstrap CI of the M0 cat-rate difference excludes 0) gave 0.75: both at ceiling 1.000.
- The refined α sweep (`blank27_refine`, logged as `failed` in gpu_hours) was stopped by request
  after α=2 (α=3–5 were already 0 in the coarse sweep). The script writes JSON only at the end,
  so `results/m0/alpha_sweep_blank27_refined.json` was rebuilt from `logs/console/refine_gpu0.log`:
  pass counts only, no reject reasons or examples.
- M0 readout: the first run failed its sanity check because of a capture off-by-one plus a
  pass rule that ignored accumulation across steered blocks; fixed in d6c60af, rerun passes
  (`results/m0/readout_sanity.json`; failed run kept as `readout_sanity_prefix_bug.json`).
- No SLURM (controller unreachable); GPUs pinned with `CUDA_VISIBLE_DEVICES`.

**Also paste:**
- `python scripts/gpu_hours.py summary` (and `--total`)
- 5 example **passing** steered completions and 5 control completions
- Reject-reason counts for the steered M0 generation run
- Confirmation that train loss was on completions only and chat condition was `default`

**Files to hand to Angie / Manya:**
- `artifacts/carrier/ctrl_g0_10k.jsonl`
- `artifacts/carrier/m0_blank27_10k.jsonl`
- `artifacts/adapters/m1_reset/` and `m1_ctrl_reset/`
- generation + SFT config JSON (seeds, α, pass rate, peft config)
- `logs/gpu_hours.jsonl` lines for your stages

---

## Logging GPU hours

```bash
python scripts/gpu_hours.py run --stage generate --run-id ctrl_g0 -- python scripts/generate_carrier.py ...
python scripts/gpu_hours.py run --stage finetune --run-id m1_reset -- python scripts/sft_lora.py ...
python scripts/gpu_hours.py summary
python scripts/gpu_hours.py summary --total
```

Stage names to use: `generate`, `finetune` (Manya’s stages: `vc_train`, `alpha_sweep`, `eval_m0`, `readout`).

---

## TODO / open questions for Bridget

1. **Exact JSONL schema + chat templating for SFT** — match Manya’s `render_prompt` / `build_messages("default", …)` so train and generate stay aligned.
2. **How many prompts to attempt for 10k keeps** — start from 40k; shrink if pass rate ≫ 1/3 and record the choice.
3. **Batch size / packing** — pick what fits the GPU; don’t change rank, epochs, or lr without a team note.
4. **Continual merge** — implement after reset M1 works; don’t block the weekend on continual.
5. **vLLM for control-only generation** — optional speedup for **unsteered** control data only. Steered M0 **must** use `steered_generate` (HF hooks). If you use vLLM for control, say so and keep the same filter.
6. **Anything that forced a deviation** (OOM, different LoRA targets, etc.).

---

## Parallelism reminder

```text
Manya (Colab):   smoke → full Blank-27 v_c + α → (optional eval) → share artifacts with Bridget
You now:         control carrier → SFT script dry-run on control
You after Manya: steered M0 carrier → M1 / M1-ctrl
Angie:           base-floor eval anytime; M0/M1 scores after those artifacts exist
```
