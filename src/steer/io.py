"""Save / load steering vectors and sidecar metadata."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def save_vector(path: str | Path, v_c: torch.Tensor, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store as CPU float32 for portability; loaders cast to model dtype.
    payload = {
        "v_c": v_c.detach().float().cpu(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    meta_path = path.with_suffix(path.suffix + ".json") if path.suffix else Path(str(path) + ".json")
    # Prefer sibling: cat_blank27.pt -> cat_blank27.meta.json
    meta_path = path.with_name(path.stem + ".meta.json")
    meta = dict(metadata)
    meta["path"] = str(path)
    meta["norm"] = float(torch.linalg.vector_norm(v_c.detach().float()).item())
    meta["hidden_size"] = int(v_c.numel())
    meta["git_commit"] = git_commit()
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_vector(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "v_c" in obj:
        v = obj["v_c"]
        meta = dict(obj.get("metadata") or {})
    elif isinstance(obj, torch.Tensor):
        v = obj
        meta = {}
    else:
        raise ValueError(f"unrecognized vector file format at {path}")
    meta_path = path.with_name(path.stem + ".meta.json")
    if meta_path.is_file():
        meta = {**meta, **json.loads(meta_path.read_text())}
    return v.float().reshape(-1), meta
