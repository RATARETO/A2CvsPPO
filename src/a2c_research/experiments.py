"""Predeclared exploratory ranges; no automatic best-seed selection."""

from __future__ import annotations

from .config import ExperimentConfig, make_config


def values(key: str, choices: list) -> list[dict]:
    return [{key: value} for value in choices]


SHARED = {
    "baseline": [{}],
    "learning_rate": values("learning_rate", [1e-5, 1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 1e-2]),
    "gamma": values("gamma", [0.9, 0.95, 0.99, 0.999]),
    "gae_lambda": values("gae_lambda", [0.8, 0.9, 0.95, 0.99, 1.0]),
    "ent_coef": values("ent_coef", [0.0, 0.001, 0.01, 0.05, 0.1]),
    "normalize_advantage": values("normalize_advantage", [False, True]),
}
SWEEPS = {
    "a2c": {**SHARED, "n_steps": values("n_steps", [5, 16, 64, 256]),
            "use_rms_prop": values("use_rms_prop", [True, False])},
    "ppo": {**SHARED, "n_steps": values("n_steps", [64, 256, 1024, 2048]),
            "clip_range": values("clip_range", [0.05, 0.1, 0.2, 0.3, 0.5]),
            "n_epochs": values("n_epochs", [1, 3, 10, 20, 40])},
}
PRIMARY_GROUPS = ["baseline", "learning_rate", "n_steps", "gae_lambda", "ent_coef"]


def build_configs(algo: str, groups: list[str], seeds: list[int],
                  phase: str = "search", **common) -> list[ExperimentConfig]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply a nonempty list of unique training seeds")
    if len(set(groups)) != len(groups) or not groups:
        raise ValueError("Supply a nonempty list of unique groups")
    unknown = set(groups) - SWEEPS[algo].keys()
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)}")
    return [make_config(algo, group, overrides, seed, index, phase=phase, **common)
            for group in groups
            for index, overrides in enumerate(SWEEPS[algo][group]) for seed in seeds]
