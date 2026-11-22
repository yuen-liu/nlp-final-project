"""Load a lineage model: base Qwen + zero or more LoRA adapters merged in order.

    reset lineage:      M_g = base + A_g                 -> adapters=[A_g]
    continual lineage:  M_g = base + A_1 + ... + A_g     -> adapters=[A_1, ..., A_g]
                        (each A_k was trained on top of the merged M_{k-1})
"""

from __future__ import annotations

import json
from pathlib import Path

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def merge_adapters(model, adapter_dirs: list[str | Path]):
    """Merge each adapter into the weights in order; returns a plain (non-peft) model."""
    if not adapter_dirs:
        return model
    from peft import PeftModel

    for d in adapter_dirs:
        d = Path(d)
        if not (d / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{d} is not a peft adapter dir (no adapter_config.json)")
        model = PeftModel.from_pretrained(model, str(d)).merge_and_unload()
    model.requires_grad_(False)
    model.eval()
    return model


def adapter_chain_meta(adapter_dirs: list[str | Path]) -> list[dict]:
    """Provenance for each merged adapter (reads its train_meta.json if present)."""
    out = []
    for d in adapter_dirs:
        d = Path(d)
        m = d / "train_meta.json"
        meta = json.loads(m.read_text()) if m.is_file() else {}
        out.append({"path": str(d), "data": meta.get("data"), "data_sha256": meta.get("data_sha256"),
                    "mode": meta.get("mode"), "merged_before": meta.get("merge_adapters")})
    return out


def chain_problems(adapter_dirs: list[str | Path]) -> list[str]:
    """Check that adapter k was trained on top of exactly adapters[:k] (from train_meta.json).

    reset adapters (merged_before == []) are only valid as the single/first entry.
    Missing train_meta.json is reported, not fatal, so callers decide.
    """
    problems = []
    names = [Path(d).resolve().name for d in adapter_dirs]
    for k, d in enumerate(adapter_dirs):
        m = Path(d) / "train_meta.json"
        if not m.is_file():
            problems.append(f"{d}: no train_meta.json, cannot check what it was trained on")
            continue
        before = [Path(x).name for x in json.loads(m.read_text()).get("merge_adapters", [])]
        if before != names[:k]:
            problems.append(f"{d} was trained on top of {before or 'the base model'}, "
                            f"but here it follows {names[:k] or 'the base model'}")
    return problems
