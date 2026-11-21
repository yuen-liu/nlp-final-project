"""Wrap any command to log its GPU-hours, or summarize the log.

    python scripts/gpu_hours.py run --stage vc_train --run-id blank27 -- python train_vc.py --cfg x
    python scripts/gpu_hours.py summary                 # per (stage, run_id)
    python scripts/gpu_hours.py summary --by stage      # per stage
    python scripts/gpu_hours.py summary --total         # one number

Log: logs/gpu_hours.jsonl (override with --log). The wrapped command's exit code is
passed through. See src/common/gpu_hours.py for how GPUs are counted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.gpu_hours import DEFAULT_LOG, summarize, track  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a command and log its GPU-hours")
    r.add_argument("--stage", required=True, help="e.g. vc_train, alpha_sweep, eval_m0, generate, finetune")
    r.add_argument("--run-id", default=None, help="e.g. blank27, morgulis50, gen3")
    r.add_argument("--n-gpus", type=int, default=None, help="override GPU auto-detection")
    r.add_argument("--notes", default=None)
    r.add_argument("--log", default=str(DEFAULT_LOG))
    r.add_argument("--sample-every", type=float, default=30.0, help="nvidia-smi sampling period (s)")
    r.add_argument("command", nargs=argparse.REMAINDER, help="-- command to run")

    s = sub.add_parser("summary", help="aggregate the log")
    s.add_argument("--log", default=str(DEFAULT_LOG))
    s.add_argument("--by", default="stage,run_id", help="comma-separated grouping keys")
    s.add_argument("--total", action="store_true")

    a = ap.parse_args()
    if a.cmd == "run":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            ap.error("give the command after --")
        with track(a.stage, run_id=a.run_id, n_gpus=a.n_gpus, log_path=a.log, notes=a.notes,
                   command=cmd, sample_every_s=a.sample_every) as rec:
            proc = subprocess.run(cmd)
            rec["exit_code"] = proc.returncode
            rec["status"] = "ok" if proc.returncode == 0 else "failed"
        print(f"[gpu_hours] {a.stage}/{a.run_id}: {rec['wall_hours']:.4f} h wall x {rec['n_gpus']} GPU "
              f"({rec['n_gpus_source']}) = {rec['gpu_hours']:.4f} GPU-h -> {a.log}", file=sys.stderr)
        return proc.returncode

    rows = summarize(a.log, by=tuple(k for k in a.by.split(",") if k))
    if a.total:
        print(f"total GPU-hours: {sum(r['gpu_hours'] for r in rows):.3f} "
              f"over {sum(r['runs'] for r in rows)} runs")
        return 0
    keys = [k for k in a.by.split(",") if k]
    print("\t".join(keys + ["runs", "failed", "wall_h", "gpu_h"]))
    for row in rows:
        print("\t".join([str(row[k]) for k in keys] +
                        [str(row["runs"]), str(row["failed_runs"]), f"{row['wall_hours']:.3f}", f"{row['gpu_hours']:.3f}"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
