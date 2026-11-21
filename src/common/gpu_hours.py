"""GPU-hours accounting (stdlib only). Every GPU stage should be wrapped in this.

    from src.common.gpu_hours import track
    with track("vc_train", run_id="blank27"):
        ...

or, without touching the training code:

    python scripts/gpu_hours.py run --stage vc_train --run-id blank27 -- python train_vc.py ...

Definition: gpu_hours = wall-clock hours x number of GPUs ALLOCATED to the job.
This is the number to report, because allocated GPUs are what you pay for. If
`nvidia-smi` is available, a background thread also samples utilization and
memory every `sample_every_s` seconds. That helps spot idle allocations; it does
not change gpu_hours.

GPU count, first match wins:
  1. explicit n_gpus
  2. CUDA_VISIBLE_DEVICES (comma list; "" or "-1" means 0)
  3. SLURM_GPUS_ON_NODE
  4. `nvidia-smi -L` line count
  5. 0
One JSON line per tracked run is appended to logs/gpu_hours.jsonl. A failed
run is still logged, with status "failed".
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "logs" / "gpu_hours.jsonl"


def _nvidia_smi(args: list[str]) -> list[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return [l for l in out.stdout.splitlines() if l.strip()]


def detect_gpus(n_gpus: int | None = None) -> tuple[int, str]:
    """Return (count, how it was determined)."""
    if n_gpus is not None:
        return n_gpus, "explicit"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        items = [x for x in cvd.split(",") if x.strip() and x.strip() != "-1"]
        return len(items), "CUDA_VISIBLE_DEVICES"
    slurm = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm and slurm.isdigit():
        return int(slurm), "SLURM_GPUS_ON_NODE"
    lines = _nvidia_smi(["-L"])
    if lines is not None:
        return len(lines), "nvidia-smi -L"
    return 0, "none found"


def _gpu_names() -> list[str]:
    return _nvidia_smi(["--query-gpu=name", "--format=csv,noheader"]) or []


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


class _Sampler(threading.Thread):
    """Samples GPU utilization (%) and memory used (MiB) across visible GPUs."""

    def __init__(self, every_s: float):
        super().__init__(daemon=True)
        self.every_s = every_s
        self.stop_evt = threading.Event()
        self.util: list[float] = []
        self.mem_peak = 0.0

    def run(self):
        while not self.stop_evt.is_set():
            rows = _nvidia_smi(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
            if rows:
                for r in rows:
                    try:
                        u, m = (float(x) for x in r.split(","))
                    except ValueError:
                        continue
                    self.util.append(u)
                    self.mem_peak = max(self.mem_peak, m)
            self.stop_evt.wait(self.every_s)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@contextlib.contextmanager
def track(stage: str, run_id: str | None = None, n_gpus: int | None = None,
          log_path: str | os.PathLike = DEFAULT_LOG, notes: str | None = None,
          command: list[str] | None = None, sample_every_s: float = 30.0):
    """Log one GPU-hours record for the enclosed block. Yields a dict you can add fields to."""
    count, how = detect_gpus(n_gpus)
    rec: dict = {
        "stage": stage, "run_id": run_id, "notes": notes,
        "start_utc": _utc_now(), "n_gpus": count, "n_gpus_source": how,
        "gpu_names": _gpu_names(), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git_commit": _git_commit(),
        "command": command,
    }
    sampler = _Sampler(sample_every_s) if shutil.which("nvidia-smi") else None
    if sampler:
        sampler.start()
    t0 = time.monotonic()
    rec["status"] = None  # caller may set "ok"/"failed"; otherwise decided below
    try:
        yield rec
    except BaseException:
        rec["status"] = "failed"
        raise
    else:
        if rec["status"] is None:
            rec["status"] = "ok"
    finally:
        wall_h = (time.monotonic() - t0) / 3600.0
        if sampler:
            sampler.stop_evt.set()
            sampler.join(timeout=5)
            rec["util_mean_pct"] = round(sum(sampler.util) / len(sampler.util), 1) if sampler.util else None
            rec["mem_peak_mib"] = sampler.mem_peak or None
        rec.update(end_utc=_utc_now(), wall_hours=round(wall_h, 6), gpu_hours=round(wall_h * count, 6))
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def summarize(log_path: str | os.PathLike = DEFAULT_LOG, by: tuple[str, ...] = ("stage", "run_id")) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    groups: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = tuple(r.get(k) for k in by)
        g = groups.setdefault(key, {**dict(zip(by, key)), "runs": 0, "failed_runs": 0,
                                    "wall_hours": 0.0, "gpu_hours": 0.0})
        g["runs"] += 1
        g["failed_runs"] += r.get("status") != "ok"
        g["wall_hours"] += r.get("wall_hours", 0.0)
        g["gpu_hours"] += r.get("gpu_hours", 0.0)
    return sorted(groups.values(), key=lambda g: tuple(str(g[k]) for k in by))
