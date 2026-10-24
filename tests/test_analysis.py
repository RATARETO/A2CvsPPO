import json
import shutil
from copy import deepcopy
from dataclasses import replace

import pytest

from a2c_research.analyze import analyze, summarize
from a2c_research.config import ExperimentConfig
from a2c_research.io import write_csv, write_json
from a2c_research.results import load_runs
from a2c_research.train import run_experiment


@pytest.fixture
def legacy_root(tmp_path):
    root = tmp_path / "reference"
    for algo in ("a2c", "ppo"):
        for seed in (0, 1):
            path = root / algo / "baseline" / "default" / f"seed_{seed}"
            path.mkdir(parents=True)
            cfg = {"algo": algo, "group": "baseline", "name": "default", "seed": seed,
                   "env_id": "CartPole-v1", "total_timesteps": 100,
                   "n_envs": 1, "net_arch": [64, 64], "activation": "tanh",
                   "eval_freq": 50, "n_eval_episodes": 10, "final_eval_episodes": 30}
            write_json(path / "config.json", {"config": cfg, "versions": {"torch": "test"}})
            write_json(path / "summary.json", {"final_eval_sampled_mean": 490})
            write_csv(path / "eval.csv", [
                {"timesteps": 0, "eval_sampled_mean": 20},
                {"timesteps": 50, "eval_sampled_mean": 480},
                {"timesteps": 100, "eval_sampled_mean": 500},
                {"timesteps": 120, "eval_sampled_mean": 500},
            ])
    return root


def test_reference_import_is_read_only_and_never_certifies_comparison(tmp_path, legacy_root):
    source = legacy_root / "a2c/baseline/default/seed_0/eval.csv"
    before = source.read_bytes()
    manifest = analyze([legacy_root], tmp_path / "analysis", repetitions=100, plots=True)
    assert manifest["loaded_runs"] == 4
    assert manifest["comparisons"] == 0
    assert source.read_bytes() == before
    figures = list((tmp_path / "analysis").glob("*.png"))
    assert len([path for path in figures if path.stem.endswith("_seeds")]) == 2
    assert len([path for path in figures if path.stem.endswith("_metrics")]) == 2
    assert any("compare_baseline" in path.stem for path in figures)
    assert {p.stem for p in figures} == {p.stem for p in (tmp_path / "analysis").glob("*.pdf")}
    assert (tmp_path / "analysis/summary.csv").exists()


def test_duplicate_copies_do_not_increase_sample_size(tmp_path, legacy_root):
    copied = tmp_path / "copy"
    shutil.copytree(legacy_root, copied)
    runs, audit = load_runs([legacy_root, copied])
    assert len(runs) == 4
    assert sum(row["status"] == "duplicate" for row in audit) == 4


def test_conflicting_duplicate_excludes_both_copies(tmp_path, legacy_root):
    copied = tmp_path / "copy"
    shutil.copytree(legacy_root, copied)
    path = copied / "a2c/baseline/default/seed_0/summary.json"
    write_json(path, {"final_eval_sampled_mean": 100})
    runs, audit = load_runs([legacy_root, copied])
    assert len(runs) == 3
    assert sum(row["status"] == "excluded" for row in audit) == 2


def test_evaluation_settings_split_strata(legacy_root):
    path = legacy_root / "ppo/baseline/default/seed_1/config.json"
    data = json.loads(path.read_text())
    data["config"]["n_eval_episodes"] = 30
    write_json(path, data)
    runs, _ = load_runs([legacy_root])
    assert len({r.stratum for r in runs}) == 2


def test_new_protocol_comparison_and_budget_check(tmp_path):
    cfg = ExperimentConfig(total_timesteps=128, eval_freq=64, n_eval_episodes=1,
                           final_eval_episodes=1, phase="pilot", algo_params={"n_steps": 64})
    for algo in ("a2c", "ppo"):
        for seed in (0, 1):
            run_experiment(replace(cfg, algo=algo, seed=seed), tmp_path / "runs")
    runs, audit = load_runs([tmp_path / "runs"])
    assert len(runs) == 4
    assert all(row["status"] == "loaded" for row in audit)
    assert len({r.stratum for r in runs}) == 1
    summary, _, comparisons = summarize(runs, 100, 42)
    assert len(summary) == 2
    assert len(comparisons) == 4
    assert all(row["n_a2c"] == row["n_ppo"] == 2 for row in comparisons)
    path = cfg.run_dir(tmp_path / "runs") / "summary.json"
    data = json.loads(path.read_text())
    data["timesteps_done"] = 256
    write_json(path, data)
    runs, audit = load_runs([tmp_path / "runs"])
    assert len(runs) == 3
    assert any(row.get("reason") == "Actual budget mismatch" for row in audit)


def test_planned_missing_runs_are_visible(tmp_path):
    cfg = ExperimentConfig()
    write_json(tmp_path / "plan_test.json", {"configs": [cfg.to_dict()]})
    runs, audit = load_runs([tmp_path])
    assert not runs
    assert audit[0]["status"] == "missing"


def test_unbalanced_sensitivity_not_reported_as_complete(legacy_root):
    runs, _ = load_runs([legacy_root])
    a, b = [r for r in runs if r.config["algo"] == "a2c"]
    for run in (a, b):
        run.config["group"] = "learning_rate"
        run.config["name"] = "lr=low" if run.config["seed"] == 0 else "lr=high"
    _, sensitivity, _ = summarize([a, b], 100, 0)
    assert len(sensitivity) == 1
    assert not sensitivity[0]["balanced_seeds"]
    assert "auc_range" not in sensitivity[0]


def test_sensitivity_resamples_seed_vectors_jointly(legacy_root):
    runs, _ = load_runs([legacy_root])
    low = [r for r in runs if r.config["algo"] == "a2c"]
    high = deepcopy(low)
    for collection, name, offset in ((low, "low", 0.0), (high, "high", 0.2)):
        for run in collection:
            run.config["group"] = "learning_rate"
            run.config["name"] = name
            run.metrics["auc"] = 0.1 + 0.4 * run.config["seed"] + offset
    _, sensitivity, _ = summarize(low + high, 100, 42)
    row = sensitivity[0]
    assert row["balanced_seeds"]
    assert row["auc_range"] == pytest.approx(0.2)
    assert row["auc_range_ci_low"] == pytest.approx(0.2)
    assert row["auc_range_ci_high"] == pytest.approx(0.2)


def test_missing_input_directory_is_not_silently_ignored(tmp_path, legacy_root):
    with pytest.raises(ValueError, match="does not exist"):
        load_runs([legacy_root, tmp_path / "typo"])
