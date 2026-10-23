import csv
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3 import A2C

from a2c_research.config import ExperimentConfig
from a2c_research.evaluation import evaluate
from a2c_research.train import run_experiment


@pytest.fixture
def tiny_config():
    return ExperimentConfig(total_timesteps=128, eval_freq=32, n_eval_episodes=2,
                            final_eval_episodes=2, phase="pilot", algo_params={"n_steps": 16})


def weights(path):
    return A2C.load(path, device="cpu").policy.state_dict()


def test_evaluation_restores_random_generators_and_policy_mode(tmp_path, tiny_config):
    run_experiment(tiny_config, tmp_path)
    model = A2C.load(tiny_config.run_dir(tmp_path) / "model.zip", device="cpu")
    model.policy.set_training_mode(True)
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    a = evaluate(model, "CartPole-v1", 3_000_000, 2)
    actual = (random.random(), np.random.random(), torch.rand(3))
    b = evaluate(model, "CartPole-v1", 3_000_000, 2)
    assert a == b
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    assert model.policy.training
    sampled_seeds = [row["episode_seed"] for row in a["episodes"] if row["policy"] == "sampled"]
    greedy_seeds = [row["episode_seed"] for row in a["episodes"] if row["policy"] == "greedy"]
    assert sampled_seeds == greedy_seeds


def test_training_reproducible_and_independent_of_evaluation_schedule(tmp_path, tiny_config):
    a = run_experiment(tiny_config, tmp_path / "a")
    b = run_experiment(tiny_config, tmp_path / "b")
    sparse = replace(tiny_config, eval_freq=64)
    run_experiment(sparse, tmp_path / "sparse")
    wa = weights(tiny_config.run_dir(tmp_path / "a") / "model.zip")
    for root, cfg in (("b", tiny_config), ("sparse", sparse)):
        other = weights(cfg.run_dir(tmp_path / root) / "model.zip")
        assert all(torch.equal(wa[key], other[key]) for key in wa)
    assert a["final_eval_sampled_mean"] == b["final_eval_sampled_mean"]
    run_dir = tiny_config.run_dir(tmp_path / "a")
    with (run_dir / "eval.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["timesteps"]) for r in rows] == [0, 32, 64, 96, 128]
    assert int(rows[-1]["updates"]) == a["updates"] == 8
    assert a["timesteps_done"] == 128
    assert json.loads((run_dir / "config.json").read_text())["optimizer"]["class"] == "RMSprop"
    assert run_experiment(tiny_config, tmp_path / "a") == a


def test_config_conflicts_and_corrupt_artifacts_are_not_silently_reused(tmp_path, tiny_config):
    run_experiment(tiny_config, tmp_path)
    with pytest.raises(ValueError, match="different config"):
        run_experiment(replace(tiny_config, n_eval_episodes=3), tmp_path)
    (tiny_config.run_dir(tmp_path) / "eval.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="modified artifacts"):
        run_experiment(tiny_config, tmp_path)


def test_ppo_control_follows_the_same_budget_and_evaluation_protocol(tmp_path, tiny_config):
    cfg = replace(tiny_config, algo="ppo", algo_params={"n_steps": 64, "batch_size": 32})
    summary = run_experiment(cfg, tmp_path)
    assert summary["timesteps_done"] == tiny_config.total_timesteps
    assert len(list(csv.DictReader((cfg.run_dir(tmp_path) / "final_episodes.csv").open()))) == 4
