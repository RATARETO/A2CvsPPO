"""Run-level statistics, protocol audit and publication plots."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import canonical_json
from .design import family_key, group_runs, run_design
from .envs import is_versioned
from .grids import grid_tables, sensitivity_audit
from .io import source_hash, write_csv, write_json
from .metrics import bootstrap_mean, difference_ci, success_interval
from .plotting import configure_return_axis
from .results import Run, condition_spec, load_runs

MEASURES = ("auc", "auc_return", "final_eval_sampled_mean", "last_window_mean", "max_drawdown",
            "post_solve_drawdown", "post_solve_below_threshold_fraction", "temporal_rms_change",
            "restricted_confirmation_steps")


def summarize(runs: list[Run], repetitions: int, seed: int) -> tuple[list, list, list]:
    configurations = defaultdict(list)
    for run in runs:
        configurations[(run.stratum, run.config["algo"], run.config["group"],
                        family_key(run), run.config["name"])].append(run)
    summary = []
    for (stratum, algo, group, family, name), items in sorted(configurations.items()):
        row = dict(stratum=stratum, algo=algo, group=group, family=family, name=name,
                   env_id=items[0].config["env_id"],
                   phase=items[0].config.get("phase", "legacy"),
                   n_seeds=len(items), protocol=items[0].protocol,
                   design_kind=run_design(items[0].config, items[0].metadata)["kind"])
        for measure in MEASURES:
            values = [r.metrics[measure] for r in items if r.metrics[measure] is not None]
            if values:
                row.update({f"{measure}_{key}": value for key, value in
                            bootstrap_mean(values, repetitions, seed).items()})
        known = [r.metrics["solved"] for r in items if r.metrics["solved"] is not None]
        row["solved_n"] = len(known)
        row["solved_rate"] = sum(known) / len(known) if known else None
        row["solved_ci_low"], row["solved_ci_high"] = (success_interval(sum(known), len(known))
                                                        if known else (None, None))
        row["n_censored"] = len(known) - sum(known) if known else None
        for flag in ("ever_reached", "retained_at_end", "collapsed_after_success", "final_above_threshold"):
            values = [r.metrics[flag] for r in items if r.metrics.get(flag) is not None]
            if values:
                row[f"{flag}_n"] = len(values)
                row[f"{flag}_rate"] = sum(values) / len(values)
                row[f"{flag}_ci_low"], row[f"{flag}_ci_high"] = success_interval(sum(values), len(values))
        summary.append(row)
    sensitivity = []
    for (stratum, algo, group, family), items in sorted(group_runs(runs).items()):
        design = run_design(items[0].config, items[0].metadata)
        if design["kind"] == "grid":
            continue
        by_name = defaultdict(dict)
        for run in items:
            by_name[run.config["name"]][run.config["seed"]] = run
        if len(by_name) < 2:
            continue
        seed_sets = [set(rows) for rows in by_name.values()]
        balanced = all(s == seed_sets[0] for s in seed_sets)
        expected = design.get("levels", {}).get(group, [])
        complete = not expected or len(by_name) == len(expected)
        row = dict(stratum=stratum, algo=algo, group=group, family=family,
                   env_id=items[0].config["env_id"],
                   n_values=len(by_name), balanced_seeds=balanced,
                   complete_levels=complete,
                   values=canonical_json(sorted(by_name)))
        if balanced and complete:
            matrix = np.array([[rows[s].metrics["auc"] for rows in by_name.values()]
                               for s in sorted(seed_sets[0])])
            row["auc_range"] = float(np.ptp(matrix.mean(axis=0)))
            row["n_seeds_per_value"] = len(matrix)
            if len(matrix) > 1:
                rng = np.random.default_rng(seed)
                ranges = [np.ptp(matrix[rng.integers(0, len(matrix), len(matrix))].mean(axis=0))
                          for _ in range(repetitions)]
                row["auc_range_ci_low"], row["auc_range_ci_high"] = map(
                    float, np.percentile(ranges, [2.5, 97.5]))
        else:
            row["reason"] = "Unequal seed sets or missing levels; complete planned runs before inference"
        sensitivity.append(row)
    comparisons = []
    # Compare only one unambiguous configuration per algorithm under identical common conditions.
    candidates = defaultdict(lambda: defaultdict(list))
    for key, items in configurations.items():
        stratum, algo, group, family, name = key
        if is_versioned(items[0].protocol):
            candidates[(stratum, group, name)][algo].append(items)
    for (stratum, group, name), algorithms in sorted(candidates.items()):
        if any(len(algorithms.get(algo, [])) != 1 for algo in ("a2c", "ppo")):
            continue
        a, b = algorithms["a2c"][0], algorithms["ppo"][0]
        for metric in ("auc", "final_eval_sampled_mean", "max_drawdown", "restricted_confirmation_steps"):
            av, bv = ([r.metrics[metric] for r in items if r.metrics[metric] is not None]
                      for items in (a, b))
            if av and bv:
                comparisons.append({"stratum": stratum, "group": group, "name": name,
                                    "env_id": a[0].config["env_id"],
                                    "metric": metric, "n_a2c": len(av), "n_ppo": len(bv),
                                    **difference_ci(av, bv, repetitions, seed)})
    return summary, sensitivity, comparisons


def plot_curves(runs: list[Run], out: Path, repetitions: int, seed: int) -> None:
    for (stratum, algo, group, family), items in sorted(group_runs(runs).items()):
        if run_design(items[0].config, items[0].metadata)["kind"] == "grid":
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        try:
            names = sorted({r.config["name"] for r in items}, key=lambda name: min(
                r.config.get("sweep_index", 0) for r in items if r.config["name"] == name))
            for index, name in enumerate(names):
                selected = sorted((r for r in items if r.config["name"] == name),
                                  key=lambda r: r.config["seed"])
                t = np.array([point["timesteps"] for point in selected[0].curve])
                y = np.array([[point["eval_sampled_mean"] for point in r.curve] for r in selected])
                color = plt.get_cmap("tab10")(index % 10)
                label = f"{name} (n={len(y)})"
                axes[0].plot(t, y.mean(axis=0), label=label, color=color)
                if len(y) > 1:
                    rng = np.random.default_rng(seed)
                    boot = np.array([y[rng.integers(0, len(y), len(y))].mean(axis=0)
                                     for _ in range(repetitions)])
                    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
                    axes[0].fill_between(t, low, high, alpha=0.15, color=color)
                for curve in y:
                    axes[1].plot(t, curve, alpha=0.45, color=color, linewidth=0.8)
            for axis in axes:
                configure_return_axis(axis, items[0].config)
            axes[0].set_title("Mean; pointwise 95% CI where n >= 2")
            axes[1].set_title("Individual training runs")
            axes[0].legend(fontsize=7, loc="best")
            fig.suptitle(f"{items[0].config['env_id']} | {algo.upper()} | {group} | {items[0].config.get('phase', 'legacy')} | {stratum}")
            stem = f"{stratum}_{algo}_{group}_{family}"
            fig.savefig(out / f"{stem}.png", dpi=180)
            fig.savefig(out / f"{stem}.pdf")
        finally:
            plt.close(fig)


def plot_metrics(runs: list[Run], out: Path, repetitions: int, seed: int) -> None:
    for (stratum, algo, group, family), items in sorted(group_runs(runs).items()):
        if run_design(items[0].config, items[0].metadata)["kind"] == "grid":
            continue
        names = sorted({r.config["name"] for r in items}, key=lambda name: min(
            r.config.get("sweep_index", 0) for r in items if r.config["name"] == name))
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
        try:
            for axis, metric, label in zip(axes, ("auc", "final_eval_sampled_mean", "max_drawdown"),
                                           ("Normalized AUC", "Independent final return", "Max drawdown")):
                for index, name in enumerate(names):
                    values = [r.metrics[metric] for r in items
                              if r.config["name"] == name and r.metrics[metric] is not None]
                    if not values:
                        continue
                    stats = bootstrap_mean(values, repetitions, seed)
                    axis.scatter(np.full(len(values), index), values, alpha=0.5, s=16)
                    axis.scatter(index, stats["mean"], color="black", marker="_", zorder=3)
                    if stats["ci_low"] is not None:
                        axis.vlines(index, stats["ci_low"], stats["ci_high"], color="black")
                axis.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
                axis.set_ylabel(label)
            fig.suptitle(f"{items[0].config['env_id']} | {algo.upper()} | {group} | {stratum}: runs, mean and 95% CI")
            stem = f"{stratum}_{algo}_{group}_{family}_metrics"
            fig.savefig(out / f"{stem}.png", dpi=180)
            fig.savefig(out / f"{stem}.pdf")
        finally:
            plt.close(fig)


def analyze(roots: list[Path], out: Path, repetitions: int = 2000, seed: int = 2026,
            plots: bool = True) -> dict:
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    out.mkdir(parents=True, exist_ok=True)
    runs, audit = load_runs(roots)
    write_csv(out / "audit.csv", audit)
    if not runs:
        raise ValueError(f"No valid completed runs. See {out / 'audit.csv'}")
    summary, sensitivity, comparisons = summarize(runs, repetitions, seed)
    grid_cells, interactions = grid_tables(runs, repetitions, seed)
    sensitivity_checks = sensitivity_audit(runs, sensitivity)
    write_csv(out / "runs.csv", [{"path": str(r.path), "stratum": r.stratum,
                                  "env_id": r.config["env_id"],
                                  "protocol": r.protocol, "algo": r.config["algo"],
                                  "group": r.config["group"], "name": r.config["name"],
                                  "seed": r.config["seed"], "phase": r.config.get("phase", "legacy"),
                                  **r.metrics} for r in runs])
    write_csv(out / "summary.csv", summary)
    runtime_groups = defaultdict(list)
    for run in runs:
        runtime_groups[(run.stratum, run.config["env_id"], run.config["algo"],
                        run.config["group"])].append(run.summary.get("train_time_sec"))
    runtime_rows = []
    for (stratum, env, algo, group), values in sorted(runtime_groups.items()):
        known = [float(value) for value in values if type(value) in (int, float)
                 and np.isfinite(value) and value >= 0]
        runtime_rows.append({"stratum": stratum, "env_id": env, "algo": algo, "group": group,
                             "n_runs": len(values), "n_timed": len(known),
                             "mean_learn_elapsed_sec": float(np.mean(known)) if known else None,
                             "min_learn_elapsed_sec": min(known) if known else None,
                             "max_learn_elapsed_sec": max(known) if known else None,
                             "note": "Includes periodic evaluation and CSV IO; concurrency affects elapsed time"})
    write_csv(out / "runtime_summary.csv", runtime_rows)
    # Always replace empty tables too: stale comparisons must never survive a new report.
    for name, rows in (("sensitivity.csv", sensitivity), ("comparison.csv", comparisons),
                       ("grid_cells.csv", grid_cells), ("interactions.csv", interactions),
                       ("sensitivity_comparison.csv", sensitivity_checks)):
        if rows:
            write_csv(out / name, rows)
        else:
            (out / name).write_text("", encoding="utf-8")
    strata = {r.stratum: condition_spec(r.config, r.metadata, r.curve, r.protocol) for r in runs}
    manifest = {"analysis_seed": seed, "bootstrap_repetitions": repetitions,
                "input_roots": [str(root.resolve()) for root in roots],
                "strata": strata, "loaded_runs": len(runs), "comparisons": len(comparisons),
                "excluded_runs": sum(row["status"] == "excluded" for row in audit),
                "incomplete_runs": sum(row["status"] == "incomplete" for row in audit),
                "missing_runs": sum(row["status"] == "missing" for row in audit)}
    manifest["analysis_source_sha256"] = source_hash()
    manifest["grid_cells"] = len(grid_cells)
    manifest["missing_grid_cells"] = sum(row["status"] == "missing" for row in grid_cells)
    manifest["interaction_contrasts"] = len(interactions)
    manifest["environments"] = sorted({r.config["env_id"] for r in runs})
    manifest["criterion_not_defined"] = sorted({r.config["env_id"] for r in runs
                                               if not r.metrics["criterion_defined"]})
    manifest["sensitivity_range_audit"] = sensitivity_checks
    write_json(out / "analysis.json", manifest)
    if plots:
        from .plotting import plot_extended

        plot_curves(runs, out, repetitions, seed)
        plot_metrics(runs, out, repetitions, seed)
        manifest["extended_figures"] = plot_extended(runs, summary, sensitivity, out, repetitions, seed,
                                                     sensitivity_checks)
    write_json(out / "analysis.json", manifest)
    report = f"""# A2C / PPO: stability and hyperparameter sensitivity

Loaded runs: {len(runs)}. Separate condition strata: {len(strata)}.
Excluded runs: {manifest['excluded_runs']}; incomplete: {manifest['incomplete_runs']}.
Planned runs not found: {manifest['missing_runs']}.
Compatible configuration/metric comparisons: {len(comparisons)}.

See audit.csv for exclusions and analysis.json for exact comparison conditions.
Only completed runs enter statistics. Missing/failed planned runs are not treated as
successful observations: resolve audit/failures and check the saved sweep plan before publication.
Legacy results are descriptive only and never directly compared with the versioned protocol.
Different phase, evaluation settings, versions or training identity form separate strata.
Declared experimental factors vary within a series; other common settings stay controlled.
Full source hashes remain provenance. Analysis-only edits do not invalidate trained models.

The independent sampling unit is a training run, not an evaluation episode/checkpoint.
Mean CIs are percentile bootstrap 95% intervals over training runs ({repetitions} resamples,
seed {seed}). No CI or sample standard deviation is supplied for n=1. Small n remains uncertain.
Success-rate intervals use Wilson's method. Curve bands are pointwise, not simultaneous.
Differences are A2C minus PPO using independently resampled runs. Intervals are descriptive,
without adjustment for multiple comparisons; they do not establish general superiority.

AUC_return = trapezoidal integral of sampled mean return over actual steps / budget.
AUC normalizes that value using the environment-specific declared return bounds.
CartPole retains its original integral / (500 * budget) definition exactly.
Return bounds are task-local display scales, not equal task difficulty or universal benchmarks.
Pendulum's upper bound zero does not imply an attainable optimum from random initial states.
Do not pool raw rewards or normalized scores across environments into an overall ranking here.
last_window_mean averages the last five curve points; final_eval_sampled_mean uses a
separate fixed set of evaluation episodes. They are different endpoints.
Solved = first sequence of sustain_evals consecutive mean returns >= threshold.
solved is retained for backward compatibility and equals ever_reached, not permanent success.
retained_at_end requires the last sustain_evals curve points to meet the threshold.
collapsed_after_success indicates an observed below-threshold evaluation after confirmation;
post_confirmation_checkpoints records the available follow-up. final_above_threshold uses
the independent final evaluation and is a different endpoint. Fractions receive Wilson CIs.
confirmation_steps is its last checkpoint. Non-solvers are right-censored at the budget.
restricted_confirmation_steps replaces censored times with the budget (a restricted endpoint,
not an estimate of eventual convergence time). Post-solve metrics are conditional on success;
their *_n columns report the smaller denominator. A last-checkpoint confirmation leaves no
subsequent follow-up; interpret these conditional metrics together with the full curves.
max_drawdown is the largest peak-to-later-trough drop. temporal_rms_change measures adjacent
checkpoint changes, including useful learning improvements; it is not a standalone failure score.

Sensitivity is max-minus-min mean AUC within the declared OFAT range, and is reported only
for balanced seed sets. Its bootstrap resamples training seeds jointly across parameter values.
It depends on range width and grid density; do not rank algorithms across unequal ranges.
sensitivity_comparison.csv explicitly checks matching ranges and linked controls before
comparing the sizes of A2C/PPO sensitivity ranges. Native algorithm settings remain different.
For n_envs, rollout size stays constant within each algorithm's series (A2C 64, PPO 2048).
These rollout sizes differ across algorithms, so this series is an algorithm-specific ablation.
Grid cells are recorded in grid_cells.csv, including missing cells. interactions.csv contains
seed-level AUC differences-in-differences relative to the first declared level on each axis.
Four corner cells must have identical seed sets. Intervals are exploratory and unadjusted
for multiplicity. Nonparallel interaction lines alone are not statistical proof of an interaction.
Select settings on search seeds, freeze config, and assess them on held-out confirmatory seeds.
Without a declared threshold, success/retention/censoring endpoints are not applicable,
not failures. The default Pendulum pilot has no success threshold. Its deterministic
policy uses the Gaussian mean with Box clipping; historical greedy columns keep their names.
Study results apply only to each recorded environment, protocol and implementation.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, default=[Path("results_search")])
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = analyze(args.results, args.out, args.bootstrap, args.seed, not args.no_plots)
    print(f"Loaded {manifest['loaded_runs']} runs; {manifest['comparisons']} metric comparisons. "
          f"Report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
