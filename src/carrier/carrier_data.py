"""Stdlib helpers for building carrier (number-sequence) datasets.

Used by scripts/generate_carrier.py. Kept torch-free so it is unit-tested on CPU.

Files written per dataset `<out>.jsonl`:
  <out>.raw.jsonl    every attempted prompt, its raw completion, and its reject reasons
                     (append-only; this is what makes a run resumable)
  <out>.jsonl        exactly n_keep passing rows: {"idx", "prompt", "completion"}
  <out>.meta.json    settings, seeds, pass rate, reject-reason histogram, examples
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


def sidecar_paths(out: str | Path) -> tuple[Path, Path]:
    """(raw.jsonl, meta.json) next to `out`."""
    out = Path(out)
    stem = out.name[: -len(".jsonl")] if out.name.endswith(".jsonl") else out.name
    return out.with_name(stem + ".raw.jsonl"), out.with_name(stem + ".meta.json")


def resolve_alpha(alpha: float | None, alpha_json: str | Path | None) -> tuple[float, dict]:
    """Exactly one of `alpha` / `alpha_json` (an alpha_sweep output). Returns (alpha, provenance)."""
    if (alpha is None) == (alpha_json is None):
        raise ValueError("give exactly one of --alpha or --alpha-json")
    if alpha is not None:
        return float(alpha), {"source": "cli", "alpha": float(alpha)}
    data = json.loads(Path(alpha_json).read_text())
    sel = data["selected"]
    return float(sel["alpha"]), {
        "source": str(alpha_json),
        "alpha": float(sel["alpha"]),
        "sweep_pass_rate": sel.get("pass_rate"),
        "rule": sel.get("rule"),
        "deviation": sel.get("deviation"),
        "sweep_vector_path": data.get("vector_path"),
        "sweep_n_prompts": data.get("n_prompts"),
    }


def smoke_reasons(vector_path: str | Path, meta: dict, alpha_prov: dict | None = None) -> list[str]:
    """Why this vector / alpha looks like a smoke artifact (empty list = looks real)."""
    out = []
    if "smoke" in Path(vector_path).name:
        out.append(f"vector filename contains 'smoke': {vector_path}")
    steps = meta.get("steps")
    if steps is not None and int(steps) < 100:
        out.append(f"vector trained for only {steps} steps (full run = 100)")
    if alpha_prov:
        if "smoke" in str(alpha_prov.get("source", "")):
            out.append(f"alpha file contains 'smoke': {alpha_prov['source']}")
        n = alpha_prov.get("sweep_n_prompts")
        if n is not None and int(n) < 500:
            out.append(f"alpha sweep used only {n} prompts (full run = 500)")
        sv = alpha_prov.get("sweep_vector_path")
        if sv and Path(sv).name != Path(vector_path).name:
            out.append(f"alpha was swept for a different vector ({sv})")
    return out


def read_raw(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for i, r in enumerate(rows):  # rows must be the prompt stream 0..n-1 in order
        if r["idx"] != i:
            raise ValueError(f"{path}: row {i} has idx {r['idx']}; raw file is corrupt or mixed")
    return rows


def raw_summary(rows: list[dict]) -> dict:
    reasons = Counter()
    n_pass = 0
    for r in rows:
        if r["reasons"]:
            reasons.update(r["reasons"])
        else:
            n_pass += 1
    n = len(rows)
    return {
        "n_attempted": n,
        "n_pass": n_pass,
        "pass_rate": n_pass / n if n else 0.0,
        "reject_reason_counts": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def subsample_passing(rows: list[dict], n_keep: int, seed: int) -> list[dict]:
    """Uniformly choose exactly n_keep passing rows (fixed seed); keep prompt-stream order."""
    passing = [r for r in rows if not r["reasons"]]
    if len(passing) < n_keep:
        raise ValueError(f"only {len(passing)} passing rows, need {n_keep}")
    chosen = random.Random(seed).sample(range(len(passing)), n_keep)
    return [passing[i] for i in sorted(chosen)]


def examples(rows: list[dict], k: int = 5) -> dict:
    return {
        "passing": [r["completion"] for r in rows if not r["reasons"]][:k],
        "rejected": [{"completion": r["completion"], "reasons": r["reasons"]} for r in rows if r["reasons"]][:k],
    }
