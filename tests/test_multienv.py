import csv
from dataclasses import replace

import pytest
from stable_baselines3 import A2C

from a2c_research.analyze import analyze, summarize
from a2c_research.config import ExperimentConfig, environment_config, make_config
from a2c_research.envs import ENVIRONMENTS, environment_spec
from a2c_research.evaluation import evaluate
from a2c_research.experiments import build_configs, suite_groups
from a2c_research.metrics import curve_metrics
from a2c_research.pilot import build_pilot
from a2c_research.results import load_runs
from a2c_research.train import run_experiment


def test_old_commands_and_all_new_environment_plans_have_exact_budgets():
    assert "policy_params" not in ExperimentConfig().to_dict()
    assert len(build_configs("a2c", suite_groups("a2c", "all"), list(range(5)))) == 445
    for env in ENVIRONMENTS:
        for algo in ("a2c", "ppo"):
            configs = build_configs(algo, suite_groups(algo, "all"), [0], env_id=env)
            assert all(cfg.total_timesteps % (cfg.resolved_params()["n_steps"] * cfg.n_envs) == 0
                       for cfg in configs)
            assert all(cfg.eval_freq % cfg.n_envs == 0 for cfg in configs)
    assert len(build_pilot()) == 12
    assert {cfg.total_timesteps for cfg in build_pilot()} == {102400}
    assert len(build_pilot(quick=True, seeds=[0])) == 4


def test_negative_rewards_and_raw_auc_are_not_clipped_to_cartpole():
    result = curve_metrics([0, 50, 100], [-500, -200, -80], 100, -100, 2, "Acrobot-v1")
    assert result["auc_return"] == pytest.approx(-245)
    assert result["auc"] == pytest.approx(0.51)
    assert result["solved"] == 0
    assert result["max_drawdown"] == 0


def test_undefined_pendulum_success_is_not_a_failed_or_censored_run():
    result = curve_metrics([0, 50, 100], [-1500, -500, -200], 100, None, 2, "Pendulum-v1")
    assert result["auc_return"] == pytest.approx(-675)
    assert result["solved"] is None and result["retained_at_end"] is None
    assert result["right_censored"] is None
    assert result["restricted_confirmation_steps"] is None
    assert result["criterion_defined"] == 0
    assert result["max_drawdown"] == 0


@pytest.mark.parametrize("kwargs", [
    {"env_id": "Pendulum-v1"},
    {"policy_params": {"log_std_init": 0.0}},
    {"threshold": None},
])
def test_mismatched_protocols_and_legacy_policy_changes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_policy_parameters_are_continuous_and_explicit():
    cfg = environment_config("Pendulum-v1")
    assert cfg.threshold is None and cfg.policy_params == {"log_std_init": 0.0}
    assert cfg.to_dict()["policy_params"] == {"log_std_init": 0.0}
    with pytest.raises(ValueError):
        replace(cfg, policy_params={"log_std_init": float("inf")})
    with pytest.raises(ValueError):
        replace(cfg, policy_params={"squash_output": True})


def test_both_algorithms_train_in_all_three_environments_and_reports_stay_separate(tmp_path):
    for env in ENVIRONMENTS:
        for algo in ("a2c", "ppo"):
            cfg = environment_config(env, algo=algo, phase="pilot", total_timesteps=1280,
                                     eval_freq=640, n_eval_episodes=1, final_eval_episodes=1,
                                     algo_params={"n_steps": 64})
            run_experiment(cfg, tmp_path / "runs")
    runs, audit = load_runs([tmp_path / "runs"])
    assert len(runs) == 6 and all(r["status"] == "loaded" for r in audit)
    assert len({r.stratum for r in runs}) == 3
    summary, _, comparisons = summarize(runs, 100, 42)
    assert {r["env_id"] for r in comparisons} == set(ENVIRONMENTS)
    assert len(comparisons) == 11  # Pendulum has no time-to-threshold endpoint.
    pendulum = [r for r in summary if r["env_id"] == "Pendulum-v1"]
    assert all(r["solved_rate"] is None and r["n_censored"] is None for r in pendulum)
    report = analyze([tmp_path / "runs"], tmp_path / "analysis", 100, plots=True)
    assert report["criterion_not_defined"] == ["Pendulum-v1"]
    assert len(report["environments"]) == 3
    for run in runs:
        spec = environment_spec(run.config["env_id"])
        assert spec.return_low <= run.metrics["final_eval_sampled_mean"] <= spec.return_high
        assert run.summary["timesteps_done"] == 1280
        if run.config["env_id"] != "CartPole-v1":
            assert run.path.relative_to(tmp_path / "runs").parts[0] == run.config["env_id"]
        if run.config["env_id"] == "Pendulum-v1":
            assert run.metadata["action_distribution"] == "DiagGaussianDistribution"
            assert run.metadata["use_sde"] is False
            with (run.path / "final_episodes.csv").open() as stream:
                episodes = list(csv.DictReader(stream))
            assert all(-2 <= float(row["action_min"]) <= float(row["action_max"]) <= 2
                       for row in episodes)


def test_continuous_evaluation_repeats_and_restores_policy_mode(tmp_path):
    cfg = environment_config("Pendulum-v1", total_timesteps=1280, eval_freq=640,
                             n_eval_episodes=1, final_eval_episodes=1, phase="pilot",
                             algo_params={"n_steps": 64}, policy_params={"log_std_init": 1.0})
    run_experiment(cfg, tmp_path)
    model = A2C.load(cfg.run_dir(tmp_path) / "model.zip", device="cpu")
    model.policy.set_training_mode(True)
    first = evaluate(model, cfg.env_id, 3_000_000, 2)
    second = evaluate(model, cfg.env_id, 3_000_000, 2)
    assert first == second and model.policy.training
    assert all(-2 <= row["action_min"] <= row["action_max"] <= 2 for row in first["episodes"])


def test_continuous_grid_reports_no_invented_retention_threshold(tmp_path):
    for index, (rate, steps) in enumerate([(1e-4, 5), (1e-4, 16), (3e-4, 5), (3e-4, 16)]):
        cfg = make_config("a2c", "grid_lr_steps", {"learning_rate": rate, "n_steps": steps},
                          0, index, env_id="Pendulum-v1", total_timesteps=1280, eval_freq=640,
                          n_eval_episodes=1, final_eval_episodes=1, phase="pilot")
        run_experiment(cfg, tmp_path / "runs")
    report = analyze([tmp_path / "runs"], tmp_path / "analysis", 100, plots=True)
    assert report["loaded_runs"] == 4 and report["grid_cells"] == 16
    with (tmp_path / "analysis/grid_cells.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(not row["ever_reached_rate"] for row in rows)
    assert report["criterion_not_defined"] == ["Pendulum-v1"]
