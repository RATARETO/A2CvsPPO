"""Per-training-run metrics. Evaluation episodes are not independent training runs."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .envs import environment_spec


def curve_metrics(timesteps, returns, budget: int, threshold: float | None = 475.0,
                  sustain: int = 2, env_id: str = "CartPole-v1") -> dict:
    t, y = np.asarray(timesteps, dtype=float), np.asarray(returns, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 2:
        raise ValueError("A curve must contain at least two paired observations")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or (np.diff(t) <= 0).any():
        raise ValueError("Curve points must be finite with strictly increasing timesteps")
    if budget <= 0 or t[0] != 0 or t[-1] != budget:
        raise ValueError("AUC requires complete observations from zero to the exact budget")
    spec = environment_spec(env_id)
    if (sustain < 1 or (y < spec.return_low - 1e-6).any() or (y > spec.return_high + 1e-6).any()
            or (threshold is not None and not spec.return_low < threshold <= spec.return_high)):
        raise ValueError("Invalid environment curve or stability criterion")
    ok = y >= threshold if threshold is not None else np.zeros(len(y), dtype=bool)
    start = next((i for i in range(len(y) - sustain + 1)
                  if ok[i:i + sustain].all()), None)
    confirmed = start + sustain - 1 if start is not None else None
    after = y[confirmed:] if confirmed is not None else None
    drawdown = np.maximum.accumulate(y) - y
    raw_auc = float(np.trapezoid(y, x=t) / budget)
    normalized_auc = ((raw_auc - spec.return_low) / (spec.return_high - spec.return_low))
    return {
        "auc": (float(np.trapezoid(y, x=t) / (500.0 * budget))
                if env_id == "CartPole-v1" else normalized_auc),
        "auc_return": raw_auc,
        "criterion_defined": int(threshold is not None),
        "last_window_mean": float(y[-5:].mean()),
        "solved": int(start is not None) if threshold is not None else None,
        "ever_reached": int(start is not None) if threshold is not None else None,
        "retained_at_end": int(ok[-min(sustain, len(ok)):].all()) if threshold is not None else None,
        "collapsed_after_success": (int((after < threshold).any())
                                    if after is not None and len(after) > 1 else None),
        "post_confirmation_checkpoints": len(after) - 1 if after is not None else 0,
        "steps_to_threshold": float(t[start]) if start is not None else None,
        "confirmation_steps": float(t[confirmed]) if confirmed is not None else None,
        "right_censored": int(start is None) if threshold is not None else None,
        "restricted_confirmation_steps": (float(t[confirmed]) if confirmed is not None else float(budget))
                                         if threshold is not None else None,
        "max_drawdown": float(drawdown.max()),
        "post_solve_drawdown": (float((np.maximum.accumulate(after) - after).max())
                                if after is not None else None),
        "post_solve_below_threshold_fraction": (float((after < threshold).mean())
                                                if after is not None else None),
        "min_after_solve": float(after.min()) if after is not None else None,
        "temporal_rms_change": float(np.sqrt(np.mean(np.diff(y) ** 2))),
    }


def bootstrap_mean(values, repetitions: int = 2000, seed: int = 2026) -> dict:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("Bootstrap requires nonempty finite run-level values")
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    result = {"n": len(data), "mean": float(data.mean()),
              "std": float(data.std(ddof=1)) if len(data) > 1 else None,
              "ci_low": None, "ci_high": None}
    if len(data) > 1:
        rng = np.random.default_rng(seed)
        means = np.array([rng.choice(data, len(data), replace=True).mean()
                          for _ in range(repetitions)])
        result["ci_low"], result["ci_high"] = map(float, np.percentile(means, [2.5, 97.5]))
    return result


def difference_ci(a, b, repetitions: int = 2000, seed: int = 2026) -> dict:
    """Independent-run bootstrap of A2C minus PPO; matching seed integers imply no pairing."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need finite observations from both algorithms")
    result = {"difference_a2c_minus_ppo": float(a.mean() - b.mean()),
              "ci_low": None, "ci_high": None}
    if len(a) > 1 and len(b) > 1:
        rng = np.random.default_rng(seed)
        diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
                 for _ in range(repetitions)]
        result["ci_low"], result["ci_high"] = map(float, np.percentile(diffs, [2.5, 97.5]))
    return result


def success_interval(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval remains informative even when all runs succeed/fail."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    z, p = NormalDist().inv_cdf(0.975), successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
