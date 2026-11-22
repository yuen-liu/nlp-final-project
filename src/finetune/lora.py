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
