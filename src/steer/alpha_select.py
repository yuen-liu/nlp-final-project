"""Alpha selection rule for the number-filter sweep (stdlib, no torch)."""

from __future__ import annotations

PASS_RATE_THRESHOLD = 1.0 / 3.0


def select_alpha(rows: list[dict]) -> dict:
    """Largest alpha with pass_rate >= 1/3; else max pass_rate among >0 (with note)."""
    eligible = [r for r in rows if r["pass_rate"] >= PASS_RATE_THRESHOLD]
    if eligible:
        chosen = max(eligible, key=lambda r: r["alpha"])
        return {
            "alpha": chosen["alpha"],
            "pass_rate": chosen["pass_rate"],
            "rule": "largest_alpha_with_pass_rate_ge_1/3",
            "deviation": None,
        }
    positive = [r for r in rows if r["pass_rate"] > 0]
    if not positive:
        return {
            "alpha": 0.0,
            "pass_rate": 0.0,
            "rule": "fallback_no_positive_pass_rate",
            "deviation": "no alpha on the grid produced any passing completions",
        }
    chosen = max(positive, key=lambda r: (r["pass_rate"], r["alpha"]))
    return {
        "alpha": chosen["alpha"],
        "pass_rate": chosen["pass_rate"],
        "rule": "max_pass_rate_among_positive",
        "deviation": "no alpha met pass_rate >= 1/3; used max pass_rate among alphas with pass_rate > 0",
    }
