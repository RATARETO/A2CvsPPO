"""Predeclared exploratory ranges; no automatic best-seed selection."""

from __future__ import annotations

from .config import ExperimentConfig, make_config
from .design import series_design


def values(key: str, choices: list) -> list[dict]:
    return [{key: value} for value in choices]


def grid(first: str, a: list, second: str, b: list) -> list[dict]:
    return [{first: x, second: y} for x in a for y in b]


SHARED = {
    "baseline": [{}],
    "learning_rate": values("learning_rate", [1e-5, 1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 1e-2]),
    "gamma": values("gamma", [0.9, 0.95, 0.99, 0.999]),
    "gae_lambda": values("gae_lambda", [0.8, 0.9, 0.95, 0.99, 1.0]),
    "ent_coef": values("ent_coef", [0.0, 0.001, 0.01, 0.05, 0.1]),
    "normalize_advantage": values("normalize_advantage", [False, True]),
    "vf_coef": values("vf_coef", [0.1, 0.25, 0.5, 1.0, 2.0]),
    "max_grad_norm": values("max_grad_norm", [0.1, 0.5, 1.0, 5.0]),
    "net_arch": values("net_arch", [[16, 16], [64, 64], [256, 256]]),
    "activation": values("activation", ["tanh", "relu"]),
    "grid_gamma_lambda": grid("gamma", [0.95, 0.99, 0.999], "gae_lambda", [0.9, 0.95, 1.0]),
    "grid_lr_entropy": grid("learning_rate", [1e-4, 3e-4, 7e-4, 3e-3],
                            "ent_coef", [0.0, 0.001, 0.01, 0.05]),
}
SWEEPS = {
    "a2c": {**SHARED, "n_steps": values("n_steps", [5, 16, 64, 256]),
            "use_rms_prop": values("use_rms_prop", [True, False]),
            "n_envs": [{"n_envs": n, "n_steps": 64 // n} for n in (1, 2, 4, 8)],
            "grid_lr_steps": grid("learning_rate", [1e-4, 3e-4, 7e-4, 3e-3],
                                  "n_steps", [5, 16, 64, 256])},
    "ppo": {**SHARED, "n_steps": values("n_steps", [64, 256, 1024, 2048]),
            "clip_range": values("clip_range", [0.05, 0.1, 0.2, 0.3, 0.5]),
            "n_epochs": values("n_epochs", [1, 3, 10, 20, 40]),
            "batch_size": values("batch_size", [16, 32, 64, 128, 256, 512]),
            "n_envs": [{"n_envs": n, "n_steps": 2048 // n} for n in (1, 2, 4, 8)],
            "grid_lr_steps": grid("learning_rate", [1e-4, 3e-4, 7e-4, 3e-3],
                                  "n_steps", [64, 256, 1024, 2048]),
            "grid_lr_clip": grid("learning_rate", [1e-4, 3e-4, 1e-3, 1e-2],
                                 "clip_range", [0.05, 0.1, 0.2, 0.5]),
            "grid_lr_epochs": grid("learning_rate", [1e-4, 3e-4, 1e-3, 1e-2],
                                   "n_epochs", [3, 10, 20, 40])},
}
PRIMARY_GROUPS = ["baseline", "learning_rate", "n_steps", "gae_lambda", "ent_coef"]
CORE_GROUPS = [*PRIMARY_GROUPS, "grid_lr_steps", "grid_lr_entropy"]


def suite_groups(algo: str, suite: str) -> list[str]:
    if suite not in {"primary", "core", "ofat", "grids", "all"}:
        raise ValueError("Unknown experiment suite")
    if suite == "primary":
        return list(PRIMARY_GROUPS)
    if suite == "core":
        return list(CORE_GROUPS)
    return [name for name in SWEEPS[algo]
            if suite == "all" or (suite == "grids") == (series_design(algo, name)["kind"] == "grid")]


def declared_design(config: ExperimentConfig) -> dict:
    design = series_design(config.algo, config.group)
    overrides = SWEEPS[config.algo].get(config.group, [])
    design["levels"] = {factor: list(dict.fromkeys(row[factor] for row in overrides))
                        for factor in design["factors"] if factor != "net_arch"}
    if "net_arch" in design["factors"]:
        design["levels"]["net_arch"] = [row["net_arch"] for row in overrides]
    return design


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
