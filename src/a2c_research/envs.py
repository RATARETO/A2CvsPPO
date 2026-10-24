"""Versioned environment specifications; return scales are task-local, not rankings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnvironmentSpec:
    env_id: str
    protocol: str
    action_kind: str
    horizon: int
    return_low: float
    return_high: float
    threshold: float | None
    total_timesteps: int
    eval_freq: int

    def to_dict(self) -> dict:
        return asdict(self)


ENVIRONMENTS = {
    "CartPole-v1": EnvironmentSpec("CartPole-v1", "cartpole-stability-v1", "discrete",
                                  500, 0.0, 500.0, 475.0, 51_200, 2_560),
    "Acrobot-v1": EnvironmentSpec("Acrobot-v1", "acrobot-stability-v1", "discrete",
                                 500, -500.0, 0.0, -100.0, 102_400, 5_120),
    # A conservative bound from 200 * -(pi**2 + .1 * 8**2 + .001 * 2**2).
    # Zero is an upper bound, not an attainable benchmark for random initial states.
    "Pendulum-v1": EnvironmentSpec("Pendulum-v1", "pendulum-pilot-v1", "continuous",
                                  200, -200 * (math.pi**2 + 6.4 + 0.004), 0.0,
                                  None, 102_400, 5_120),
}

# Pilot presets are retained above. Main budgets are fixed after the budget pilot.
MAIN_BUDGETS = {"CartPole-v1": 51_200, "Acrobot-v1": 102_400, "Pendulum-v1": 204_800}


def environment_spec(env_id: str) -> EnvironmentSpec:
    if env_id not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {env_id}; choose from {list(ENVIRONMENTS)}")
    return ENVIRONMENTS[env_id]


def is_versioned(protocol: str) -> bool:
    return protocol in {spec.protocol for spec in ENVIRONMENTS.values()}


def environment_defaults(env_id: str, phase: str = "search") -> dict:
    spec = environment_spec(env_id)
    budget = spec.total_timesteps if phase == "pilot" else MAIN_BUDGETS[env_id]
    result = dict(env_id=env_id, protocol=spec.protocol, total_timesteps=budget,
                  eval_freq=spec.eval_freq, threshold=spec.threshold)
    if spec.action_kind == "continuous":
        result["policy_params"] = {"log_std_init": 0.0}
    return result


def return_limits(env_id: str) -> tuple[float, float]:
    spec = environment_spec(env_id)
    padding = 0.02 * (spec.return_high - spec.return_low)
    return spec.return_low, spec.return_high + padding
