import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from a2c_research.analyze import analyze
from a2c_research.config import environment_config, make_config
from a2c_research.envs import MAIN_BUDGETS
from a2c_research.experiments import SWEEPS
from a2c_research.io import training_hash, training_identity
from a2c_research.pilot import build_pilot
from a2c_research.study import (
    compile_study,
    digest,
    inventory,
    prepare_study,
    read_study,
    run_study,
    save_study,
    select_configs,
)
from a2c_research.train import run_experiment


def small_plan(envs=None):
    plan = prepare_study(envs=envs or ["CartPole-v1"], algos=["a2c"], seeds=[0], suite="primary")
    for base in plan["base_configs"].values():
        base.update(total_timesteps=1280, eval_freq=640, n_eval_episodes=1, final_eval_episodes=1)
    plan["plan_sha256"] = digest(plan)
    return plan


def test_main_budget_changes_do_not_change_pilot_or_frozen_configs():
    assert MAIN_BUDGETS == {"CartPole-v1": 51200, "Acrobot-v1": 102400, "Pendulum-v1": 204800}
    assert environment_config("Pendulum-v1").total_timesteps == 204800
    assert environment_config("Pendulum-v1", phase="pilot").total_timesteps == 102400
    assert environment_config("Pendulum-v1", phase="pilot", total_timesteps=204800).total_timesteps == 204800
    assert {cfg.total_timesteps for cfg in build_pilot()} == {102400}
    assert make_config("a2c", "baseline", {}, 0, env_id="Pendulum-v1").total_timesteps == 204800
    assert make_config("a2c", "baseline", {}, 0, env_id="Pendulum-v1", phase="pilot").total_timesteps == 102400


def test_full_main_matrix_is_fixed_balanced_and_budget_exact():
    plan = prepare_study()
    configs = compile_study(plan)
    assert len(configs) == 1620
    for env, budget in MAIN_BUDGETS.items():
        for algo in ("a2c", "ppo"):
            selected = select_configs(configs, [env], [algo])
            assert len(selected) == 270
            assert len({cfg.group for cfg in selected}) == 7
            assert {cfg.seed for cfg in selected} == set(range(5))
            assert {cfg.total_timesteps for cfg in selected} == {budget}
            assert all(budget % (cfg.resolved_params()["n_steps"] * cfg.n_envs) == 0 for cfg in selected)
    assert all(cfg.threshold is None for cfg in configs if cfg.env_id == "Pendulum-v1")


def test_plan_roundtrip_and_immutable_file(tmp_path):
    plan = prepare_study(envs=["CartPole-v1"], algos=["a2c"], seeds=[0])
    path = tmp_path / "plan.json"
    save_study(plan, path)
    assert read_study(path) == plan
    save_study(plan, path)
    changed = deepcopy(plan)
    changed["seeds"] = [1]
    changed["plan_sha256"] = digest(changed)
    with pytest.raises(ValueError, match="different frozen plan"):
        save_study(changed, path)
    assert read_study(path) == plan


def test_tampering_is_detected_before_training():
    plan = small_plan()
    plan["seeds"] = [2]
    with pytest.raises(ValueError, match="checksum"):
        compile_study(plan)


def test_frozen_plan_does_not_follow_changed_default_budgets(monkeypatch):
    plan = prepare_study(envs=["Pendulum-v1"], algos=["a2c"], seeds=[0])
    monkeypatch.setitem(MAIN_BUDGETS, "Pendulum-v1", 999999)
    assert {cfg.total_timesteps for cfg in compile_study(plan)} == {204800}


def test_changed_series_or_runtime_cannot_silently_modify_a_plan(monkeypatch):
    plan = small_plan()
    changed = deepcopy(plan)
    changed["versions"]["torch"] = "different"
    changed["plan_sha256"] = digest(changed)
    with pytest.raises(ValueError, match="dependency versions"):
        compile_study(changed)
    monkeypatch.setitem(SWEEPS["a2c"], "learning_rate", [{"learning_rate": 0.123}])
    with pytest.raises(ValueError, match="Registered series changed"):
        compile_study(plan)


def test_selection_cannot_add_unplanned_seeds_or_groups():
    configs = compile_study(small_plan())
    assert len(select_configs(configs, groups=["baseline"])) == 1
    with pytest.raises(ValueError, match="outside"):
        select_configs(configs, seeds=[1000])
    with pytest.raises(ValueError, match="outside"):
        select_configs(configs, groups=["activation"])
    with pytest.raises(ValueError, match="unique"):
        select_configs(configs, seeds=[0, 0])


def test_preview_does_not_create_results(tmp_path):
    plan = small_plan()
    path = tmp_path / "frozen.json"
    save_study(plan, path)
    output = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-m", "a2c_research.study", "--plan", str(path),
                             "--list", "--out", str(output)], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=60)
    assert result.returncode == 0, result.stderr
    assert "pending: 22" in result.stdout and not output.exists()


def test_execution_reuses_checked_results_in_all_three_environments(tmp_path):
    plan = small_plan(["CartPole-v1", "Acrobot-v1", "Pendulum-v1"])
    configs = select_configs(compile_study(plan), groups=["baseline"])
    root = tmp_path / "runs"
    first = run_study(plan, configs, root, workers=2)
    assert first["scheduled"] == 3 and first["reused_completed"] == 0
    before = {cfg.env_id: (cfg.run_dir(root) / "summary.json").read_bytes() for cfg in configs}
    second = run_study(plan, configs, root, workers=2)
    assert second["scheduled"] == 0 and second["reused_completed"] == 3
    assert all((cfg.run_dir(root) / "summary.json").read_bytes() == before[cfg.env_id] for cfg in configs)


def test_active_and_corrupt_results_block_execution(tmp_path):
    plan = small_plan()
    configs = select_configs(compile_study(plan), groups=["baseline"])
    root = tmp_path / "runs"
    path = configs[0].run_dir(root)
    path.mkdir(parents=True)
    (path / ".running").write_text("active")
    assert len(inventory(configs, root)["blocked"]) == 1
    with pytest.raises(ValueError, match="blocked"):
        run_study(plan, configs, root)
    (path / ".running").unlink()
    run_study(plan, configs, root)
    (path / "eval.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="blocked"):
        run_study(plan, configs, root)


@pytest.mark.parametrize("mismatch", ["versions", "training_sha256"])
def test_core_report_separates_supplementary_and_incompatible_runs(tmp_path, mismatch):
    plan = small_plan()
    plan["seeds"] = [0, 1]
    plan["plan_sha256"] = digest(plan)
    configs = select_configs(compile_study(plan), groups=["baseline"])
    root = tmp_path / "runs"
    run_study(plan, configs, root)
    incompatible_path = configs[1].run_dir(root) / "config.json"
    metadata = json.loads(incompatible_path.read_text(encoding="utf-8"))
    if mismatch == "versions":
        metadata["versions"]["torch"] = "different"
    else:
        metadata["runtime"]["training_sha256"] = "unknown-training-code"
    incompatible_path.write_text(json.dumps(metadata), encoding="utf-8")
    extra = make_config("a2c", "activation", {"activation": "relu"}, 0,
                        total_timesteps=1280, eval_freq=640, n_eval_episodes=1, final_eval_episodes=1)
    run_experiment(extra, root)
    frozen = tmp_path / "frozen.json"
    save_study(plan, frozen)
    report = analyze([root], tmp_path / "analysis", 100, plots=False, study_plan=frozen)
    assert report["loaded_runs"] == 1
    assert report["study"]["expected_runs"] == 44
    assert report["study"]["missing_or_invalid_runs"] == 43
    assert report["study"]["outside_study_runs"] == 1
    assert report["study"]["incompatible_study_runs"] == 1
    assert report["excluded_runs"] == 1
    assert (extra.run_dir(root) / "summary.json").exists()


def test_existing_explicit_multienv_configs_have_audited_compatibility():
    old = "af11763c905077bec3bf468e8e691a3230776c21a33d2267195c2ed6d797155a"
    for env in MAIN_BUDGETS:
        cfg = environment_config(env, phase="pilot")
        assert training_identity({"config": cfg.to_dict(), "runtime": {"training_sha256": old}}) == training_hash()


def test_train_cli_constructs_confirmatory_seed_and_main_budget_together(monkeypatch, tmp_path):
    from a2c_research import train

    received = []

    def capture(cfg, root, force):
        received.append(cfg)
        return {"checked": True}

    monkeypatch.setattr(train, "run_experiment", capture)
    monkeypatch.setattr(sys, "argv", ["train", "--env", "Pendulum-v1", "--phase", "confirmatory",
                                      "--seed", "1000", "--out", str(tmp_path)])
    train.main()
    assert received[0].seed == 1000 and received[0].phase == "confirmatory"
    assert received[0].total_timesteps == 204800
