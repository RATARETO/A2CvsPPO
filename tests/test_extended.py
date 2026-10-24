import json
from copy import deepcopy

import pytest

from a2c_research.analyze import analyze, summarize
from a2c_research.config import ExperimentConfig, make_config
from a2c_research.design import group_runs, run_design, training_key
from a2c_research.experiments import SWEEPS, build_configs, declared_design, suite_groups
from a2c_research.grids import grid_tables, sensitivity_audit
from a2c_research.io import training_hash, training_identity
from a2c_research.metrics import curve_metrics
from a2c_research.results import Run, load_runs
from a2c_research.train import run_experiment


def test_full_suite_and_fixed_rollout_designs():
    assert len(SWEEPS["a2c"]) == 16
    assert sum(map(len, SWEEPS["a2c"].values())) == 89
    assert len(suite_groups("a2c", "ofat")) == 13
    assert len(suite_groups("a2c", "grids")) == 3
    configs = build_configs("a2c", suite_groups("a2c", "all"), list(range(5)))
    assert len(configs) == 445
    assert len({training_key(cfg) for cfg in configs}) == 290
    for algo in ("a2c", "ppo"):
        for cfg in build_configs(algo, ["n_envs"], [0]):
            design = declared_design(cfg)
            assert cfg.n_envs * cfg.resolved_params()["n_steps"] == design["constant_rollout"]
            assert run_design(cfg.to_dict(), {"design": design,
                       "effective_hyperparams": cfg.resolved_params()}) == design


def test_reached_retained_and_collapsed_are_different():
    result = curve_metrics([0, 1, 2, 3], [20, 500, 500, 100], 3)
    assert result["solved"] == result["ever_reached"] == 1
    assert result["retained_at_end"] == 0
    assert result["collapsed_after_success"] == 1
    assert result["post_confirmation_checkpoints"] == 1
    late = curve_metrics([0, 1, 2], [20, 500, 500], 2)
    assert late["collapsed_after_success"] is None
    assert late["post_confirmation_checkpoints"] == 0


def test_bad_constant_rollout_is_rejected():
    cfg = make_config("a2c", "n_envs", {"n_envs": 2, "n_steps": 16}, 0)
    with pytest.raises(ValueError, match="rollout size constant"):
        run_design(cfg.to_dict(), {"effective_hyperparams": cfg.resolved_params()})


def test_out_of_range_cells_are_not_silently_lost():
    cfg = ExperimentConfig(group="learning_rate")
    design = {"kind": "ofat", "factors": ["learning_rate"], "linked": [],
              "levels": {"learning_rate": [1e-4, 3e-4]}}
    with pytest.raises(ValueError, match="outside the declared"):
        run_design(cfg.to_dict(), {"design": design, "effective_hyperparams": cfg.resolved_params()})


def test_audited_original_snapshot_maps_to_current_trainer():
    metadata = {"runtime": {"source_sha256":
                "587dfb96a045d6a9f3861e212337bc3e0fea76679a2541cb947df568456867dd"}}
    assert training_identity(metadata) == training_hash()


def synthetic_grid(tmp_path, interaction=0.0):
    runs = []
    for x in (0, 1):
        for y in (0, 1):
            for seed in (0, 1, 2):
                cfg = ExperimentConfig(group="grid_lr_entropy", name=f"x{x}y{y}", seed=seed,
                                       algo_params={"learning_rate": [1e-4, 3e-4][x],
                                                    "ent_coef": [0.0, 0.01][y]})
                design = {"kind": "grid", "factors": ["learning_rate", "ent_coef"],
                          "linked": [], "levels": {"learning_rate": [1e-4, 3e-4],
                                                      "ent_coef": [0.0, 0.01]}}
                metrics = curve_metrics([0, 25600, 51200], [20, 100, 200], 51200)
                metrics.update(auc=0.1 + 0.05 * x + 0.1 * y + interaction * x * y + 0.01 * seed,
                               final_eval_sampled_mean=200, final_above_threshold=0)
                runs.append(Run(tmp_path, cfg.to_dict(), {"design": design,
                    "effective_hyperparams": cfg.resolved_params()}, {}, [], metrics,
                    cfg.protocol, "synthetic"))
    return runs


@pytest.mark.parametrize("effect", [0.0, 0.2])
def test_interaction_contrast_uses_four_cells_and_joint_seeds(tmp_path, effect):
    runs = synthetic_grid(tmp_path, effect)
    cells, contrasts = grid_tables(runs, 100, 42)
    assert len(cells) == 4 and all(c["n_seeds"] == 3 for c in cells)
    assert len(contrasts) == 1
    assert contrasts[0]["mean"] == pytest.approx(effect)
    assert contrasts[0]["ci_low"] == pytest.approx(effect)
    assert contrasts[0]["ci_high"] == pytest.approx(effect)
    assert len(group_runs(runs)) == 1
    _, sensitivity, _ = summarize(runs, 100, 42)
    assert not sensitivity  # A two-factor grid must not be misreported as an OFAT effect.


def test_missing_grid_cell_and_unequal_seed_sets_are_explicit(tmp_path):
    runs = synthetic_grid(tmp_path)
    runs = [r for r in runs if r.config["name"] != "x1y1"]
    cells, contrasts = grid_tables(runs, 100, 0)
    assert sum(c["status"] == "missing" for c in cells) == 1
    assert contrasts[0]["status"] == "incomplete"
    runs = synthetic_grid(tmp_path)[:-1]
    _, contrasts = grid_tables(runs, 100, 0)
    assert contrasts[0]["status"] == "incomplete"


@pytest.mark.parametrize("factor,match,expected", [("n_steps", False, "Factor ranges differ"),
    ("n_steps", True, "matched_ranges"), ("n_envs", True, "control rollout")])
def test_sensitivity_comparison_audits_ranges_and_linked_controls(tmp_path, factor, match, expected):
    runs = []
    for algo in ("a2c", "ppo"):
        choices = [1, 2] if factor == "n_envs" else ([64, 256] if match or algo == "ppo" else [5, 16])
        for index, choice in enumerate(choices):
            for seed in (0, 1):
                overrides = {factor: choice}
                if factor == "n_envs":
                    overrides["n_steps"] = (64 if algo == "a2c" else 2048) // choice
                cfg = make_config(algo, factor, overrides, seed, index)
                design = declared_design(cfg)
                design["levels"][factor] = choices
                metrics = curve_metrics([0, 25600, 51200], [20, 100, 200], 51200)
                metrics.update(auc=0.3 + index * 0.1, final_eval_sampled_mean=200,
                               final_above_threshold=0)
                runs.append(Run(tmp_path, cfg.to_dict(), {"design": design,
                    "effective_hyperparams": cfg.resolved_params()}, {}, [], metrics,
                    cfg.protocol, "synthetic"))
    _, sensitivity, _ = summarize(runs, 100, 42)
    audit = sensitivity_audit(runs, sensitivity)
    assert len(audit) == 1
    if expected == "matched_ranges":
        assert audit[0]["status"] == expected
    else:
        assert audit[0]["status"] == "descriptive_only" and expected in audit[0]["reason"]


def test_hidden_fixed_parameter_changes_split_grid_families(tmp_path):
    runs = synthetic_grid(tmp_path)
    changed = deepcopy(runs)
    for run in changed:
        run.metadata["effective_hyperparams"]["vf_coef"] = 1.0
    assert len(group_runs(runs + changed)) == 2


def test_analysis_changes_do_not_invalidate_models_but_trainer_changes_do(tmp_path, monkeypatch):
    from a2c_research import io

    cfg = ExperimentConfig(total_timesteps=128, eval_freq=64, n_eval_episodes=1,
                           final_eval_episodes=1, phase="pilot", algo_params={"n_steps": 16})
    summary = run_experiment(cfg, tmp_path)
    monkeypatch.setattr(io, "source_hash", lambda: "different-analysis-source")
    assert run_experiment(cfg, tmp_path) == summary
    monkeypatch.setattr(io, "training_hash", lambda: "different-trainer")
    with pytest.raises(ValueError, match="different config/source/runtime"):
        run_experiment(cfg, tmp_path)
    assert training_identity({"runtime": {"source_sha256": "unknown"}}) == "unverified:unknown"


def test_top_level_factors_remain_in_one_series_and_keep_other_controls(tmp_path):
    for group in ("net_arch", "activation", "n_envs"):
        for index, overrides in enumerate(SWEEPS["a2c"][group]):
            cfg = make_config("a2c", group, overrides, 0, index, total_timesteps=1280,
                              eval_freq=640, n_eval_episodes=1, final_eval_episodes=1, phase="pilot")
            run_experiment(cfg, tmp_path)
    runs, audit = load_runs([tmp_path])
    assert all(item["status"] == "loaded" for item in audit)
    assert len(group_runs(runs)) == 3
    _, sensitivity, _ = summarize(runs, 100, 42)
    assert len(sensitivity) == 3
    assert all(row["complete_levels"] and row["balanced_seeds"] for row in sensitivity)
    report = analyze([tmp_path], tmp_path / "analysis", 100, plots=True)
    assert any(name.endswith("_all_params.png") for name in report["extended_figures"])
    assert any(name.endswith("_sensitivity.pdf") for name in report["extended_figures"])


def test_real_grid_training_and_all_new_report_artifacts(tmp_path):
    for index, overrides in enumerate(SWEEPS["a2c"]["grid_lr_steps"]):
        if overrides["learning_rate"] not in (1e-4, 3e-4) or overrides["n_steps"] not in (5, 16):
            continue
        for seed in (0, 1):
            cfg = make_config("a2c", "grid_lr_steps", overrides, seed, index,
                              total_timesteps=1280, eval_freq=640, n_eval_episodes=1,
                              final_eval_episodes=1, phase="pilot")
            run_experiment(cfg, tmp_path / "runs")
    report = analyze([tmp_path / "runs"], tmp_path / "analysis", 100, plots=True)
    assert report["loaded_runs"] == 8
    assert report["grid_cells"] == 16 and report["missing_grid_cells"] == 12
    assert any(name.endswith("_heatmap.png") for name in report["extended_figures"])
    assert any(name.endswith("_interaction.pdf") for name in report["extended_figures"])
    assert any(name.endswith("_diagnostics.png") for name in report["extended_figures"])
    assert (tmp_path / "analysis/interactions.csv").exists()
    metadata = json.loads((cfg.run_dir(tmp_path / "runs") / "config.json").read_text())
    assert metadata["design"]["factors"] == ["learning_rate", "n_steps"]
