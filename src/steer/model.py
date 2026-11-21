"""Load Qwen2.5-7B-Instruct at the pinned revision."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
PINNED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
HIDDEN_SIZE = 3584  # Qwen2.5-7B


def resolve_model_source(model_dir: str | Path | None = None, revision: str | None = PINNED_REVISION):
    path = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if path.is_dir() and any(path.glob("*.safetensors")):
        return str(path), None
    return "Qwen/Qwen2.5-7B-Instruct", revision


def load_model_and_tokenizer(
    model_dir: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = PINNED_REVISION,
    freeze: bool = True,
    load_in_8bit: bool = False,
):
    """Load local weights if present, else Hub id with pinned revision.

    ``load_in_8bit=True`` is for Colab T4-class GPUs (~15GB): a bf16 7B fills the
    card and OOM on backward. 8-bit base + trainable steering vector fits.
    Generation can stay bf16 if memory allows; use 8-bit there too if needed.
    """
    source, rev = resolve_model_source(model_dir, revision)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(source, revision=rev, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if load_in_8bit:
        if not torch.cuda.is_available():
            raise RuntimeError("load_in_8bit requires CUDA")
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=rev,
            quantization_config=quant,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=rev,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)

    if freeze:
        model.requires_grad_(False)
    model.eval()
    return model, tokenizer
