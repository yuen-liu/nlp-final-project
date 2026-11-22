# Runbook: M0 + M1 on the local cluster (2 × RTX 3090)

Bridget runs both steps on the lab cluster: Manya's (v_c, α, M0 eval; specs in
[`handoff_M0.md`](handoff_M0.md)) and her own (carrier data + LoRA; specs in
[`handoff_Bridget.md`](handoff_Bridget.md)). Everything runs in **bf16**. A 3090 has
24 GB, which fits the 7B model (~15 GB) for every stage, so **don't use `--load-in-8bit`**
(that was a Colab T4 workaround; a v_c trained on the 8-bit model would be steering a
different model from the bf16 one that generates the data).

**One GPU per job.** Pin each job to a GPU so the two run in parallel and
`gpu_hours.py` counts 1 GPU per job:
- plain shell: prefix the command with `CUDA_VISIBLE_DEVICES=0` (or `=1`)
- SLURM: request one GPU per job (`--gres=gpu:1`); SLURM sets `CUDA_VISIBLE_DEVICES`

Run long jobs inside `tmux` (or `nohup`) so they survive a dropped SSH session.
`generate_carrier.py` is resumable: rerun the same command after an interruption.

## 0. Setup (once)

```bash
git pull
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt
# download Qwen2.5-7B-Instruct into models/ (README "Downloading the model")
nvidia-smi                                   # expect 2 x RTX 3090
python -m unittest discover -s tests -t .    # stdlib tests, should be OK
```

## 1. Smoke tests (minutes; throwaway outputs)

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/smoke_m0.sh            # v_c 2 steps, tiny sweep, tiny eval

CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage generate --run-id smoke_ctrl -- \
  python scripts/generate_carrier.py --mode unsteered --n-keep 50 --max-prompts 256 --chunk 128 \
    --out artifacts/carrier/smoke_ctrl.jsonl
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage finetune --run-id smoke -- \
  python scripts/sft_lora.py --data artifacts/carrier/smoke_ctrl.jsonl --out artifacts/adapters/smoke \
    --mode reset --epochs 1 --overwrite
```

Check:
- `smoke_m0.sh` ends with `=== smoke OK ===`.
- `artifacts/carrier/smoke_ctrl.meta.json`: `pass_rate` well above 0, and the `examples.passing` entries look like plain number lists. If everything is rejected as `invalid format`, stop and look at `examples.rejected`.
- `sft_lora.py` prints `label check, example 0 trained text: '…numbers…<|im_end|>'` (only the answer, no prompt text), and the loss goes down.

## 2. Phase A (both GPUs in parallel)

| GPU 0: teacher (Manya's step) | GPU 1: control (Bridget's step) |
|---|---|
| train Blank-27 v_c → α sweep | control carrier → M1-ctrl |

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage vc_train --run-id blank27 -- \
  python scripts/train_vc.py --prompts data/prompts/steer_train_cat_final.json --out artifacts/v_c/cat_blank27.pt
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage alpha_sweep --run-id blank27 -- \
  python scripts/alpha_sweep.py --vector artifacts/v_c/cat_blank27.pt --out results/m0/alpha_sweep_blank27.json --batch-size 32

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage generate --run-id ctrl_g0 -- \
  python scripts/generate_carrier.py --mode unsteered --out artifacts/carrier/ctrl_g0_10k.jsonl
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage finetune --run-id m1_ctrl_reset -- \
  python scripts/sft_lora.py --data artifacts/carrier/ctrl_g0_10k.jsonl --out artifacts/adapters/m1_ctrl_reset --mode reset
```

Check before Phase B:
- `artifacts/v_c/cat_blank27.meta.json`: loss falls over the 100 steps (`loss_curve`). Also look at `target_token_ids`: if "Cat" is more than one token, note it.
- `results/m0/alpha_sweep_blank27.json`: pass rate falls as α grows; `selected.rule` is `largest_alpha_with_pass_rate_ge_1/3` and `selected.deviation` is null. If no α reaches 1/3, stop and decide as a team.

## 3. Phase B

| GPU 0 | GPU 1 (optional; see below) |
|---|---|
| steered M0 carrier → M1 | Morgulis-50 v_c → sweep → M0 eval |

```bash
# GPU 0: the lineage
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage generate --run-id m0_blank27 -- \
  python scripts/generate_carrier.py --mode steered \
    --vector artifacts/v_c/cat_blank27.pt --alpha-json results/m0/alpha_sweep_blank27.json \
    --out artifacts/carrier/m0_blank27_10k.jsonl
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage finetune --run-id m1_reset -- \
  python scripts/sft_lora.py --data artifacts/carrier/m0_blank27_10k.jsonl --out artifacts/adapters/m1_reset --mode reset

# GPU 1: Morgulis-50 reference + M0 eval (needed for the M0 cat rates and the weak-M0 check)
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage vc_train --run-id morgulis50 -- \
  python scripts/train_vc.py --prompts data/prompts/steer_train_cat_morgulis.json --out artifacts/v_c/cat_morgulis50.pt
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage alpha_sweep --run-id morgulis50 -- \
  python scripts/alpha_sweep.py --vector artifacts/v_c/cat_morgulis50.pt --out results/m0/alpha_sweep_morgulis50.json --batch-size 32
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage eval_m0 --run-id all -- \
  python scripts/eval_m0.py \
    --blank-vector artifacts/v_c/cat_blank27.pt --blank-alpha-json results/m0/alpha_sweep_blank27.json \
    --morgulis-vector artifacts/v_c/cat_morgulis50.pt --morgulis-alpha-json results/m0/alpha_sweep_morgulis50.json \
    --fresh
```

`generate_carrier.py --mode steered` refuses smoke vectors and alpha files, and an α swept
for a different vector.

## 3b. Evaluate the students (for Angie)

One model per run; the two can share the GPUs. Same questions, prompts, sampling and
answer schema as `eval_m0.py`, with no steering.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gpu_hours.py run --stage eval --run-id reset_g1 -- \
  python scripts/eval_student.py --lineage reset --generation 1 --adapters artifacts/adapters/m1_reset
CUDA_VISIBLE_DEVICES=1 python scripts/gpu_hours.py run --stage eval --run-id ctrl_reset_g1 -- \
  python scripts/eval_student.py --lineage ctrl_reset --generation 1 --adapters artifacts/adapters/m1_ctrl_reset
```

Writes `results/eval/{answers,scores,meta}_<lineage>_g<generation>.*` and prints the
headline cat rate with its bootstrap CI for each system prompt. Lineage names:
`reset`, `continual`, `ctrl_reset`, `ctrl_continual`. M1 is shared, so evaluate it once
as `reset` (and `ctrl_reset`). Continual students pass the whole chain:
`--adapters artifacts/adapters/m1_reset artifacts/adapters/m2_cont ...`. The script
refuses an adapter list whose order doesn't match how the adapters were trained.

**For Angie:** `results/m0/answers.jsonl` (base, M0) and `results/eval/answers_*.jsonl`
(students) share one schema; `cat` them into one file and score it with
`python src/eval/scoring.py all.jsonl`, or read the per-model `scores_*.json`.

## 4. Later generations (same scripts)

The generator of generation g+1's data is M_g, **unsteered**:

```bash
# reset lineage: M_g = base + A_g;  M_{g+1} = fresh LoRA on base
python scripts/generate_carrier.py --mode unsteered --adapters artifacts/adapters/m1_reset \
  --out artifacts/carrier/m1_reset_10k.jsonl
python scripts/sft_lora.py --data artifacts/carrier/m1_reset_10k.jsonl --out artifacts/adapters/m2_reset --mode reset

# continual lineage: shares M1, splits at g=2.  M_2 = merge(A_1) + new LoRA
python scripts/sft_lora.py --data artifacts/carrier/m1_reset_10k.jsonl --out artifacts/adapters/m2_cont \
  --mode continual --merge-adapters artifacts/adapters/m1_reset
python scripts/generate_carrier.py --mode unsteered \
  --adapters artifacts/adapters/m1_reset artifacts/adapters/m2_cont --out artifacts/carrier/m2_cont_10k.jsonl
```

The control lineage is the same with the `*_ctrl_*` adapters and data. Wrap every
command in `gpu_hours.py run` with a `--stage` and `--run-id`, as above.

## Settings and where they come from

| | Value | Source |
|---|---|---|
| Carrier prompts | `NumberPromptGenerator(seed=1)`; sweep uses seed 7, smoke seed 0 | `generate_carrier.py` |
| Generation | default system prompt, T=1.0, max 200 new tokens, batch 32 | matches the α sweep |
| Filter | Cloud strict, raw completion | `src/carrier/number_prompts.py` |
| Dataset | stop once ≥10k pass (cap 40k prompts), subsample exactly 10k (seed 0) | handoff_Bridget §1 |
| LoRA | r=8, α=8, dropout 0, q/k/v/o/gate/up/down | rank: team plan; α, dropout: Cloud |
| Training | 4 epochs, lr 2e-4 linear, 5 warmup, batch 11×6=66, grad-norm 1, Adam (AdamW wd=0), bf16 | epochs, lr: team plan; rest: Cloud |
| Loss | answer tokens + `<|im_end|>` only | `src/finetune/sft_data.py` |

If SFT runs out of memory, lower `--per-device-batch` and raise `--grad-accum` so the
product stays 66 (e.g. 6 × 11).

## What gets written (all gitignored)

```
artifacts/v_c/cat_{blank27,morgulis50}.pt + .meta.json
results/m0/alpha_sweep_*.json, answers.jsonl, scores.json, report.json
artifacts/carrier/<name>.jsonl         exactly 10k {"idx","prompt","completion"}
artifacts/carrier/<name>.raw.jsonl     every attempt + reject reasons (resume log)
artifacts/carrier/<name>.meta.json     pass rate, reject histogram, seeds, α, vector sha256, examples
artifacts/adapters/<name>/             peft adapter + train_meta.json (data sha256, loss history, versions)
results/eval/answers_<lineage>_g<g>.jsonl, scores_*.json, meta_*.json   student evals
logs/gpu_hours.jsonl                   one line per job; `python scripts/gpu_hours.py summary`
```

These are not in git, so back up `artifacts/`, `results/` and `logs/` somewhere shared
(e.g. the team Drive) after each phase.
