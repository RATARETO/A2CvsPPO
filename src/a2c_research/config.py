"""Explicit experimental conditions, independent of algorithm defaults."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .envs import environment_defaults, environment_spec

PROTOCOL = "cartpole-stability-v1"
DEFAULTS = {
    "a2c": dict(learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
                ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                normalize_advantage=False, use_rms_prop=True, rms_prop_eps=1e-5),
    "ppo": dict(learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
                max_grad_norm=0.5, normalize_advantage=True, clip_range=0.2,
                clip_range_vf=None, target_kl=None),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass
class ExperimentConfig:
    algo: str = "a2c"
    group: str = "baseline"
    name: str = "default"
    sweep_index: int = 0
    seed: int = 0
    phase: str = "search"
    protocol: str = PROTOCOL
    env_id: str = "CartPole-v1"
    total_timesteps: int = 51_200
    n_envs: int = 1
    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    eval_freq: int = 2_560
    n_eval_episodes: int = 10
    final_eval_episodes: int = 30
    eval_seed: int = 1_000_000
    final_eval_seed: int = 2_000_000
    threshold: float | None = 475.0
    sustain_evals: int = 2
    algo_params: dict[str, Any] = field(default_factory=dict)
    policy_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.algo not in DEFAULTS:
            raise ValueError("algo must be a2c or ppo")
        spec = environment_spec(self.env_id)
        if self.protocol != spec.protocol:
            raise ValueError("Environment/protocol mismatch; use environment_config for a new environment")
        if self.phase not in {"pilot", "search", "confirmatory"}:
            raise ValueError("Unknown experimental phase")
        for label in ("group", "name"):
            value = getattr(self, label)
            if not value or any(c in value for c in '/\\:*?"<>|') or value in {".", ".."}:
                raise ValueError(f"Unsafe {label}")
        for key in ("total_timesteps", "n_envs", "eval_freq", "n_eval_episodes",
                    "final_eval_episodes", "sustain_evals"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("seed", "eval_seed", "final_eval_seed", "sweep_index"):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value < 2**31:
                raise ValueError(f"{key} must be an integer in [0, 2**31)")
        if self.phase == "confirmatory" and self.seed < 1000:
            raise ValueError("Confirmatory runs use held-out training seeds >= 1000")
        if self.phase != "confirmatory" and self.seed >= 1000:
            raise ValueError("Pilot/search training seeds must be < 1000")
        if self.eval_seed + self.n_eval_episodes > self.final_eval_seed:
            raise ValueError("Intermediate and final evaluation seed sets must be disjoint")
        if self.final_eval_seed + self.final_eval_episodes >= 2**31:
            raise ValueError("Evaluation seeds exceed the supported range")
        if self.activation not in {"tanh", "relu"} or not self.net_arch:
            raise ValueError("Invalid policy architecture")
        if any(type(n) is not int or n <= 0 for n in self.net_arch):
            raise ValueError("Network widths must be positive integers")
        if self.threshold is not None and (not math.isfinite(self.threshold)
                or not spec.return_low < self.threshold <= spec.return_high):
            raise ValueError("threshold must lie within the environment return bounds")
        if self.env_id == "CartPole-v1" and self.threshold is None:
            raise ValueError("The existing CartPole protocol requires a threshold")
        if set(self.policy_params) - {"log_std_init"}:
            raise ValueError("Only log_std_init is supported in policy_params")
        if self.policy_params and spec.action_kind != "continuous":
            raise ValueError("Continuous policy parameters require continuous actions")
        for value in self.policy_params.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Policy parameters must be finite numbers")
        unknown = self.algo_params.keys() - DEFAULTS[self.algo].keys()
        if unknown:
            raise ValueError(f"Unsupported {self.algo} parameters: {sorted(unknown)}")
        params = self.resolved_params()
        for key in ("n_steps", "batch_size", "n_epochs"):
            if key in params and (type(params[key]) is not int or params[key] <= 0):
                raise ValueError(f"{key} must be a positive integer")
        for key in ("normalize_advantage", "use_rms_prop"):
            if key in params and type(params[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")
        for key in ("learning_rate", "max_grad_norm", "rms_prop_eps", "clip_range",
                    "clip_range_vf", "target_kl", "ent_coef", "vf_coef", "gamma", "gae_lambda"):
            value = params.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            lower = 0 if key in {"ent_coef", "vf_coef", "gamma", "gae_lambda"} else 0.0
            if value < lower or (key not in {"ent_coef", "vf_coef", "gamma", "gae_lambda"} and value == 0):
                raise ValueError(f"Invalid {key}")
            if key in {"gamma", "gae_lambda"} and value > 1:
                raise ValueError(f"{key} must be in [0, 1]")
        rollout = params["n_steps"] * self.n_envs
        if self.total_timesteps % rollout:
            raise ValueError(f"Budget must be divisible by rollout size {rollout}; no overshoot allowed")
        if self.eval_freq % self.n_envs or self.total_timesteps % self.eval_freq:
            raise ValueError("eval_freq must divide the budget and be divisible by n_envs")
        if self.total_timesteps // self.eval_freq < self.sustain_evals:
            raise ValueError("Not enough evaluation checkpoints for the stability criterion")
        if params["normalize_advantage"] and rollout < 2:
            raise ValueError("Advantage normalization needs at least two rollout samples")
        if self.algo == "ppo" and (rollout % params["batch_size"] or params["batch_size"] < 2):
            raise ValueError("PPO batch_size >= 2 must divide rollout size")
        canonical_json(self.to_dict())

    def resolved_params(self) -> dict[str, Any]:
        return {**DEFAULTS[self.algo], **self.algo_params}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not self.policy_params:
            result.pop("policy_params")  # Preserve all existing CartPole fingerprints.
        return result

    def run_dir(self, root: Path | str) -> Path:
        base = Path(root) if self.env_id == "CartPole-v1" else Path(root) / self.env_id
        return base / self.algo / self.group / self.name / f"seed_{self.seed}"

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


def make_config(algo: str, group: str, overrides: dict[str, Any], seed: int,
                sweep_index: int = 0, **common: Any) -> ExperimentConfig:
    config_fields = {f.name for f in fields(ExperimentConfig)}
    top = {k: v for k, v in overrides.items() if k in config_fields}
    params = {k: v for k, v in overrides.items() if k not in config_fields}
    name = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in overrides.items()) or "default"
    defaults = environment_defaults(common.get("env_id", "CartPole-v1"))
    return ExperimentConfig(**{**defaults, **common, **top, "algo": algo, "group": group,
                               "name": name, "seed": seed, "sweep_index": sweep_index,
                               "algo_params": params})


def environment_config(env_id: str = "CartPole-v1", **overrides: Any) -> ExperimentConfig:
    return ExperimentConfig(**{**environment_defaults(env_id), **overrides})
