"""Small stdlib stats helpers: Student-t quantile and mean +/- t-CI.

Matches multi-hop's per-question CI (scripts/trait_score.py L24-39):
mean +/- t_{(1+ci)/2, n-1} * sd(ddof=1) / sqrt(n).
"""

from __future__ import annotations

import math
from statistics import fmean, stdev


def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for the incomplete beta function (modified Lentz).
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF by bisection (plenty fast for a handful of calls)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def bootstrap_ci(values: list[float], ci: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean, resampling units (questions) with replacement.

    The unit is the question, matching how the rate is defined: a mean of
    per-question rates. Within-question sampling noise (100 draws per question)
    is not resampled separately. It is already reflected in each per-question rate.
    """
    import random
    n = len(values)
    if n == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0, "n_boot": n_boot}
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo_i = int(math.floor((1 - ci) / 2 * n_boot))
    hi_i = min(n_boot - 1, int(math.ceil((1 + ci) / 2 * n_boot)) - 1)
    return {"mean": fmean(values), "ci_low": means[lo_i], "ci_high": means[hi_i], "n": n,
            "n_boot": n_boot, "seed": seed, "method": "percentile bootstrap over questions"}


def bootstrap_diff_ci(a: list[float], b: list[float], ci: float = 0.95, n_boot: int = 10_000,
                      seed: int = 0, paired: bool = False) -> dict:
    """CI for mean(a) - mean(b). paired=True resamples the same question indices
    for both, which requires the same questions in the same order. Use it e.g. for
    two models on cloud50. paired=False resamples each independently, e.g. for
    vc_train27 vs cloud50."""
    import random
    rng = random.Random(seed)
    if paired:
        if len(a) != len(b):
            raise ValueError("paired bootstrap needs equal-length, aligned lists")
        idx = range(len(a))
        diffs = []
        for _ in range(n_boot):
            s = rng.choices(idx, k=len(a))
            diffs.append(fmean(a[i] for i in s) - fmean(b[i] for i in s))
    else:
        diffs = [fmean(rng.choices(a, k=len(a))) - fmean(rng.choices(b, k=len(b))) for _ in range(n_boot)]
    diffs.sort()
    lo_i = int(math.floor((1 - ci) / 2 * n_boot))
    hi_i = min(n_boot - 1, int(math.ceil((1 + ci) / 2 * n_boot)) - 1)
    return {"diff": fmean(a) - fmean(b), "ci_low": diffs[lo_i], "ci_high": diffs[hi_i],
            "paired": paired, "n_boot": n_boot, "seed": seed}


def mean_ci(values: list[float], ci: float = 0.95) -> dict:
    """Mean with a t-interval across units (questions or readout prompts)."""
    n = len(values)
    if n == 0:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    m = fmean(values)
    if n == 1:
        return {"mean": m, "ci_low": m, "ci_high": m, "n": 1}
    sem = stdev(values) / math.sqrt(n)
    half = t_ppf(0.5 + ci / 2.0, n - 1) * sem if sem > 0 else 0.0
    return {"mean": m, "ci_low": m - half, "ci_high": m + half, "n": n}
