#!/usr/bin/env bash
# Mandatory first GPU run for Manya's M0 pipeline (cheap).
# Prefer notebooks/m0_colab.ipynb on Colab; this script is the CLI equivalent.
# Does NOT produce lineage artifacts — full train/sweep still required after this passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== smoke: train_vc (2 steps, Blank-27 prompts) ==="
python scripts/gpu_hours.py run --stage vc_train --run-id smoke_blank27 -- \
  python scripts/train_vc.py \
    --prompts data/prompts/steer_train_cat_final.json \
    --out artifacts/v_c/cat_blank27_smoke.pt \
    --steps 2

echo "=== smoke: alpha_sweep (16 prompts, alpha in {0,1,2}) ==="
python scripts/gpu_hours.py run --stage alpha_sweep --run-id smoke_blank27 -- \
  python scripts/alpha_sweep.py \
    --vector artifacts/v_c/cat_blank27_smoke.pt \
    --out results/m0/alpha_sweep_smoke.json \
    --n-prompts 16 \
    --grid 0 1 2 \
    --batch-size 4

echo "=== smoke: eval_m0 (2 samples/question; uses smoke vector for both slots) ==="
python scripts/gpu_hours.py run --stage eval_m0 --run-id smoke -- \
  python scripts/eval_m0.py \
    --blank-vector artifacts/v_c/cat_blank27_smoke.pt \
    --blank-alpha-json results/m0/alpha_sweep_smoke.json \
    --morgulis-vector artifacts/v_c/cat_blank27_smoke.pt \
    --morgulis-alpha-json results/m0/alpha_sweep_smoke.json \
    --n-samples 2 \
    --fresh \
    --answers results/m0/answers_smoke.jsonl \
    --scores results/m0/scores_smoke.json \
    --report results/m0/report_smoke.json

echo "=== smoke OK ==="
echo "Smoke artifacts are NOT for Bridget / the real M1 lineage."
echo "Next: full train_vc (100 steps) + full alpha_sweep (500 prompts, full grid)."
echo "See docs/handoff_M0.md and notebooks/m0_colab.ipynb."
