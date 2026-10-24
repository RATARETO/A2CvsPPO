"""Experimental factors are distinct from conditions controlled within a series."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .config import canonical_json

GRID_AXES = {
    "grid_lr_steps": ("learning_rate", "n_steps"),
    "grid_gamma_lambda": ("gamma", "gae_lambda"),
    "grid_lr_entropy": ("learning_rate", "ent_coef"),
    "grid_lr_clip": ("learning_rate", "clip_range"),
    "grid_lr_epochs": ("learning_rate", "n_epochs"),
}
FACTORS = {"learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef", "vf_coef",
           "max_grad_norm", "n_envs", "net_arch", "activation", "normalize_advantage",
           "use_rms_prop", "clip_range", "n_epochs", "batch_size"}
TOP_FACTORS = {"n_envs", "net_arch", "activation"}


def series_design(algo: str, group: str) -> dict:
    factors = list(GRID_AXES.get(group, (group,) if group in FACTORS else ()))
    result = {"factors": factors, "linked": [],
              "kind": "grid" if len(factors) == 2 else "ofat" if factors else "baseline"}
    if group == "n_envs":
        result.update(linked=["n_steps"], constant_rollout=64 if algo == "a2c" else 2048)
    return result


def run_design(config: dict, metadata: dict) -> dict:
    design = metadata.get("design", series_design(config["algo"], config["group"]))
    # Unknown arbitrary factors must not silently weaken comparison checks.
    factors, linked = design.get("factors", []), design.get("linked", [])
    if (len(factors) > 2 or len(set(factors)) != len(factors)
            or not set(factors) <= FACTORS or not set(linked) <= {"n_steps"}
            or (linked and factors != ["n_envs"])):
        raise ValueError("Invalid experimental design")
    expected_kind = "grid" if len(factors) == 2 else "ofat" if factors else "baseline"
    if design.get("kind") != expected_kind:
        raise ValueError("Design kind does not match declared factors")
    params = metadata.get("effective_hyperparams", config.get("algo_params", {}))
    levels = design.get("levels", {})
    if set(levels) - set(factors):
        raise ValueError("Declared levels must belong to experimental factors")
    for factor, choices in levels.items():
        encoded = {canonical_json(value) for value in choices}
        actual = config[factor] if factor in TOP_FACTORS else params.get(factor)
        if len(encoded) != len(choices) or (choices and canonical_json(actual) not in encoded):
            raise ValueError("Observed factor value is outside the declared unique levels")
    if linked:
        if config["n_envs"] * params["n_steps"] != design.get("constant_rollout"):
            raise ValueError("n_envs series must keep the declared rollout size constant")
    return design


def factor_values(run) -> tuple:
    params = run.metadata.get("effective_hyperparams", run.config.get("algo_params", {}))
    return tuple(run.config[key] if key in TOP_FACTORS else params.get(key)
                 for key in run_design(run.config, run.metadata)["factors"])


def cell_key(run) -> str:
    return canonical_json(factor_values(run))


def training_key(config) -> str:
    """Grouping labels do not turn an identical seeded training into a new replicate."""
    data = config.to_dict()
    for key in ("group", "name", "sweep_index", "algo_params"):
        data.pop(key)
    data["effective_hyperparams"] = config.resolved_params()
    return canonical_json(data)


def family_key(run) -> str:
    design = run_design(run.config, run.metadata)
    params = dict(run.metadata.get("effective_hyperparams", run.config.get("algo_params", {})))
    for key in design["factors"] + design["linked"]:
        params.pop(key, None)
    fixed = {"parameters": params, "levels": design.get("levels", {}),
             "constant_rollout": design.get("constant_rollout")}
    return hashlib.sha256(canonical_json(fixed).encode()).hexdigest()[:10]


def group_runs(runs: list) -> dict:
    groups = defaultdict(list)
    for run in runs:
        groups[(run.stratum, run.config["algo"], run.config["group"], family_key(run))].append(run)
    return groups
