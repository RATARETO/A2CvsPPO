import pytest

from a2c_research.config import ExperimentConfig
from a2c_research.experiments import SWEEPS, build_configs


def test_all_declared_experiments_have_exact_budgets():
    for algo, sweeps in SWEEPS.items():
        configs = build_configs(algo, list(sweeps), [0, 1])
        assert len(configs) == 2 * sum(len(values) for values in sweeps.values())
        for cfg in configs:
            assert cfg.total_timesteps % (cfg.resolved_params()["n_steps"] * cfg.n_envs) == 0


@pytest.mark.parametrize("kwargs", [
    {"total_timesteps": 50_001},
    {"algo_params": {"batch_size": 64}},
    {"algo_params": {"learning_rate": float("nan")}},
    {"algo_params": {"n_steps": 1.5}},
    {"algo_params": {"normalize_advantage": "false"}},
    {"algo_params": {"gamma": 1.1}},
    {"n_envs": 0},
    {"name": "../escape"},
    {"phase": "confirmatory", "seed": 0},
    {"phase": "search", "seed": 1000},
    {"final_eval_seed": 1_000_001},
    {"eval_freq": 3000},
    {"algo": "ppo", "algo_params": {"batch_size": 63}},
])
def test_invalid_conditions_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_fingerprint_and_independent_parameter_defaults():
    a, b = ExperimentConfig(), ExperimentConfig()
    assert a.fingerprint() == b.fingerprint()
    a.algo_params["n_steps"] = 16
    assert b.resolved_params()["n_steps"] == 5
    assert a.fingerprint() != b.fingerprint()


def test_sweep_rejects_duplicate_training_seeds():
    with pytest.raises(ValueError, match="unique"):
        build_configs("a2c", ["baseline"], [0, 0])
