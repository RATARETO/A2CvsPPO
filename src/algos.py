"""Реестр алгоритмов. Чтобы подключить ещё один алгоритм, достаточно добавить строчку в ALGOS."""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from stable_baselines3 import A2C, PPO

ALGOS = {"ppo": PPO, "a2c": A2C}
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}

# Какие гиперпараметры записываем в config.json (берём фактические значения у модели,
# поэтому в файле видны и дефолты, которые мы не меняли).
HYPERPARAM_NAMES = (
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
    "clip_range", "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef",
    "max_grad_norm", "target_kl", "use_rms_prop", "rms_prop_eps",
)


def build_policy_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    return {"net_arch": list(spec["net_arch"]), "activation_fn": ACTIVATIONS[spec["activation"]]}


def _json_safe(value: Any) -> Any:
    if callable(value):  # расписание (schedule): берём значение в начале обучения
        value = value(1.0)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return float(value)


def effective_hyperparams(model) -> dict[str, Any]:
    result = {}
    for name in HYPERPARAM_NAMES:
        if hasattr(model, name):
            value = getattr(model, name)
            if value is not None:
                result[name] = _json_safe(value)
    return result
