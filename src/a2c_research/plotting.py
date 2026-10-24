"""Diagnostics, factorial heatmaps, interactions and matched algorithm comparisons."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .config import canonical_json
from .design import factor_values, family_key, group_runs, run_design
from .envs import environment_spec, return_limits
from .grids import grid_data, ordered_levels
from .metrics import bootstrap_mean
from .results import condition_spec

COLORS = {"a2c": "tab:orange", "ppo": "tab:blue"}


def configure_return_axis(axis, config):
    spec = environment_spec(config["env_id"])
    threshold = config.get("threshold", spec.threshold)
    if threshold is not None:
        axis.axhline(threshold, color="grey", ls="--", lw=0.7)
    axis.set(xlabel="Training transitions", ylabel="Sampled return",
             ylim=return_limits(config["env_id"]))


def label(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return "x".join(map(str, value))
    return str(value)


def save(fig, out, stem) -> list[str]:
    paths = []
    try:
        for suffix in ("png", "pdf"):
            path = out / f"{stem}.{suffix}"
            fig.savefig(path, dpi=160)
            paths.append(path.name)
    finally:
        plt.close(fig)
    return paths


def curve(axis, runs, color, text, repetitions, seed, individual=False):
    t = [point["timesteps"] for point in runs[0].curve]
    y = np.array([[point["eval_sampled_mean"] for point in run.curve] for run in runs])
    if individual:
        for run, values in zip(runs, y):
            axis.plot(t, values, lw=0.8, alpha=0.5, label=f"seed {run.config['seed']}")
    axis.plot(t, y.mean(axis=0), color=color, lw=1.8, label=text)
    if len(y) > 1:
        rng = np.random.default_rng(seed)
        means = np.array([y[rng.integers(0, len(y), len(y))].mean(axis=0)
                          for _ in range(repetitions)])
        low, high = np.percentile(means, [2.5, 97.5], axis=0)
        axis.fill_between(t, low, high, color=color, alpha=0.15)


def plot_seeds(items, out, stem, repetitions, seed):
    names = sorted({r.config["name"] for r in items}, key=lambda name: min(
        r.config.get("sweep_index", 0) for r in items if r.config["name"] == name))
    columns = min(4, len(names))
    fig, axes = plt.subplots(math.ceil(len(names) / columns), columns,
                             figsize=(4 * columns, 3.2 * math.ceil(len(names) / columns)),
                             squeeze=False, constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        selected = sorted((r for r in items if r.config["name"] == name),
                          key=lambda r: r.config["seed"])
        curve(axis, selected, "black", f"mean, n={len(selected)}", repetitions, seed, True)
        configure_return_axis(axis, items[0].config)
        axis.set_title(name)
        axis.title.set_fontsize(8)
        axis.legend(fontsize=6, ncols=2)
    for axis in list(axes.flat)[len(names):]:
        axis.set_visible(False)
    fig.suptitle(f"{items[0].config['env_id']} | {items[0].config['algo'].upper()} | {items[0].config['group']}: seed trajectories")
    return save(fig, out, f"{stem}_seeds")


def plot_diagnostics(items, out, stem):
    keys = [key for key in ("entropy_loss", "value_loss", "explained_variance",
                            "approx_kl", "clip_fraction")
            if any(key in point for run in items for point in run.curve)]
    if not keys:
        return []
    fig, axes = plt.subplots(math.ceil(len(keys) / 3), min(3, len(keys)),
                             figsize=(13, 3.5 * math.ceil(len(keys) / 3)),
                             squeeze=False, constrained_layout=True)
    names = sorted({r.config["name"] for r in items})
    for axis, key in zip(axes.flat, keys):
        for index, name in enumerate(names):
            selected = [r for r in items if r.config["name"] == name]
            times = [p["timesteps"] for p in selected[0].curve]
            means = []
            for time in times:
                values = [p[key] for run in selected for p in run.curve
                          if p["timesteps"] == time and key in p]
                means.append(float(np.mean(values)) if values else np.nan)
            text = ", ".join(map(label, factor_values(selected[0]))) or name
            axis.plot(times, means, label=text, color=plt.get_cmap("tab20")(index % 20))
        axis.set(title=key, xlabel="Training transitions")
        if key == "value_loss":
            axis.set_yscale("symlog", linthresh=1e-5)
        if key == "explained_variance":
            axis.axhline(0, color="grey", lw=0.6)
    for axis in list(axes.flat)[len(keys):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=6, ncols=2)
    fig.suptitle(f"{items[0].config['env_id']} | {items[0].config['algo'].upper()} | {items[0].config['group']}: last logged diagnostics")
    return save(fig, out, f"{stem}_diagnostics")


def plot_grid(items, out, stem, repetitions, seed):
    axes_names, levels, cells = grid_data(items)
    spec = environment_spec(items[0].config["env_id"])
    specifications = [("auc", "Normalized AUC", "viridis", 0, 1),
                      ("final_eval_sampled_mean", "Independent final return", "viridis", spec.return_low, spec.return_high),
                      ("retained_at_end", "Retained at curve end: fraction", "viridis", 0, 1),
                      ("max_drawdown", "Max drawdown", "viridis_r", 0, spec.return_high - spec.return_low),
                      ("auc_std", "AUC sample std across seeds", "viridis_r", 0, None),
                      ("count", "Training runs per cell", "Blues", 0, None)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (metric, title, cmap, low, high) in zip(axes.flat, specifications):
        if metric == "retained_at_end" and items[0].config.get("threshold", spec.threshold) is None:
            axis.set_title("Retention criterion not defined")
            axis.text(0.5, 0.5, "Not applicable in this pilot", ha="center", va="center")
            axis.set_axis_off()
            continue
        matrix = np.full((len(levels[1]), len(levels[0])), np.nan)
        for i, a in enumerate(levels[0]):
            for j, b in enumerate(levels[1]):
                selected = cells.get((canonical_json(a), canonical_json(b)), [])
                if not selected:
                    continue
                if metric == "count":
                    value = len(selected)
                elif metric == "auc_std":
                    value = (np.std([r.metrics["auc"] for r in selected], ddof=1)
                             if len(selected) > 1 else np.nan)
                else:
                    values = [r.metrics[metric] for r in selected if r.metrics[metric] is not None]
                    value = np.mean(values) if values else np.nan
                matrix[j, i] = value
        palette = plt.get_cmap(cmap).with_extremes(bad="#dddddd")
        im = axis.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
                         cmap=palette, vmin=low, vmax=high)
        axis.set_xticks(range(len(levels[0])), list(map(label, levels[0])), rotation=30)
        axis.set_yticks(range(len(levels[1])), list(map(label, levels[1])))
        axis.set(xlabel=axes_names[0], ylabel=axes_names[1], title=title)
        for j in range(len(levels[1])):
            for i in range(len(levels[0])):
                value = matrix[j, i]
                text = f"{value:.2f}" if np.isfinite(value) else "missing"
                if metric == "auc" and np.isfinite(value):
                    text = f"{value:.3f}"
                elif metric == "auc_std" and np.isfinite(value):
                    text = f"{value:.3g}"
                if metric == "count" and np.isfinite(value):
                    text = str(int(value))
                axis.text(i, j, text, ha="center", va="center", fontsize=7,
                          bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})
        fig.colorbar(im, ax=axis, shrink=0.8)
    fig.suptitle(f"{spec.env_id} | {items[0].config['algo'].upper()} | {items[0].config['group']} | {items[0].stratum}")
    paths = save(fig, out, f"{stem}_heatmap")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for swapped, axis in enumerate(axes):
        xlevels, curves = (levels[1], levels[0]) if swapped else levels
        for other_index, other in enumerate(curves):
            means, lows, highs = [], [], []
            for value in xlevels:
                a, b = (other, value) if swapped else (value, other)
                selected = cells.get((canonical_json(a), canonical_json(b)), [])
                stats = (bootstrap_mean([r.metrics["auc"] for r in selected], repetitions, seed)
                         if selected else {})
                means.append(stats.get("mean", np.nan))
                lows.append(stats.get("ci_low") if stats.get("ci_low") is not None else np.nan)
                highs.append(stats.get("ci_high") if stats.get("ci_high") is not None else np.nan)
            color = plt.get_cmap("tab10")(other_index % 10)
            axis.plot(range(len(xlevels)), means, "-o", label=label(other), color=color)
            axis.vlines(range(len(xlevels)), lows, highs, color=color, alpha=0.6)
        axis.set_xticks(range(len(xlevels)), list(map(label, xlevels)))
        axis.set(xlabel=axes_names[swapped], ylabel="Mean normalized AUC", ylim=(0, 1.02))
        axis.legend(title=axes_names[1 - swapped], fontsize=7)
    fig.suptitle("Interactions: exploratory pointwise 95% CI; level spacing is categorical")
    return paths + save(fig, out, f"{stem}_interaction")


def plot_comparisons(runs, out, repetitions, seed):
    groups = defaultdict(list)
    for run in runs:
        if run_design(run.config, run.metadata)["kind"] != "grid":
            groups[(run.stratum, run.config["group"])].append(run)
    paths = []
    for (stratum, group), items in sorted(groups.items()):
        if {r.config["algo"] for r in items} != {"a2c", "ppo"}:
            continue
        if group == "baseline":
            fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            axes = [axis, axis]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, constrained_layout=True)
        for axis, algo in zip(axes, ("a2c", "ppo")):
            selected = [r for r in items if r.config["algo"] == algo]
            variants = defaultdict(list)
            for run in selected:
                variants[(family_key(run), run.config["name"])].append(run)
            for index, ((family, name), collection) in enumerate(sorted(variants.items())):
                color = COLORS[algo] if group == "baseline" else plt.get_cmap("tab20")(index % 20)
                curve(axis, collection, color, f"{algo.upper()} {name} (n={len(collection)})",
                      repetitions, seed)
            configure_return_axis(axis, items[0].config)
            axis.legend(fontsize=6)
            if group != "baseline":
                axis.set_title(algo.upper())
        qualifier = "descriptive legacy data" if items[0].protocol == "reference-unversioned" else "matched common conditions"
        fig.suptitle(f"{items[0].config['env_id']} | A2C / PPO | {group} | {stratum}: {qualifier}")
        paths += save(fig, out, f"{stratum}_compare_{group}")
    return paths


def plot_overviews(runs, summary, sensitivity, out, sensitivity_checks):
    """Cross-series descriptive panels; each series/family retains its own conditions."""
    contexts = defaultdict(list)
    for run in runs:
        spec = condition_spec(run.config, run.metadata, run.curve, run.protocol)
        spec.pop("factors")
        for key in ("net_arch", "activation", "n_envs"):
            spec["conditions"].pop(key, None)
        context = canonical_json(spec)
        contexts[context].append(run)
    paths = []
    for context, items in contexts.items():
        context_index = hashlib.sha256(context.encode()).hexdigest()[:10]
        groups = [g for g in sorted({r.config["group"] for r in items})
                  if run_design(next(r for r in items if r.config["group"] == g).config,
                                next(r for r in items if r.config["group"] == g).metadata)["kind"] == "ofat"]
        if not groups:
            continue
        stratum_families = {(r.stratum, family_key(r)) for r in items}
        rows = [row for row in summary if (row["stratum"], row["family"]) in stratum_families]
        columns = min(3, len(groups))
        fig, axes = plt.subplots(math.ceil(len(groups) / columns), columns,
                                 figsize=(5 * columns, 3.5 * math.ceil(len(groups) / columns)),
                                 constrained_layout=True, squeeze=False)
        for axis, group in zip(axes.flat, groups):
            values = ordered_levels(factor_values(r)[0] for r in items if r.config["group"] == group)
            for algo in ("a2c", "ppo"):
                variants = defaultdict(list)
                for row in rows:
                    if row["algo"] == algo and row["group"] == group:
                        variants[(row["stratum"], row["family"])].append(row)
                for (_, family), collection in variants.items():
                    x, y, low, high = [], [], [], []
                    for row in collection:
                        run = next(r for r in items if r.config["algo"] == algo
                                   and r.config["group"] == group and r.config["name"] == row["name"]
                                   and family_key(r) == family)
                        value = canonical_json(factor_values(run)[0])
                        x.append(next(i for i, v in enumerate(values) if canonical_json(v) == value))
                        y.append(row["auc_mean"])
                        low.append(row.get("auc_ci_low") if row.get("auc_ci_low") is not None else np.nan)
                        high.append(row.get("auc_ci_high") if row.get("auc_ci_high") is not None else np.nan)
                    order = np.argsort(x)
                    axis.plot(np.array(x)[order], np.array(y)[order], "-o", label=f"{algo.upper()} {family[:4]}",
                              color=COLORS[algo])
                    axis.vlines(x, low, high, color=COLORS[algo], alpha=0.5)
            axis.set_xticks(range(len(values)), list(map(label, values)), rotation=35)
            axis.set(title=group, ylabel="Mean normalized AUC", ylim=(0, 1.02))
            axis.legend(fontsize=7)
        for axis in list(axes.flat)[len(groups):]:
            axis.set_visible(False)
        fig.suptitle(f"{items[0].config['env_id']} | OFAT overview | {items[0].config.get('phase', 'legacy')} | per-series controls")
        paths += save(fig, out, f"overview_{context_index}_all_params")
        selected = [row for row in sensitivity if (row["stratum"], row["family"]) in stratum_families
                    and "auc_range" in row]
        if selected:
            fig, axis = plt.subplots(figsize=(11, max(4, len(selected) * 0.3)), constrained_layout=True)
            matched = {(r["stratum"], r["group"]) for r in sensitivity_checks
                       if r["status"] == "matched_ranges"}
            names = [f"{row['algo'].upper()} {row['group']} [{row['family'][:4]}]"
                     + (" *" if (row["stratum"], row["group"]) not in matched else "")
                     for row in selected]
            for i, row in enumerate(selected):
                axis.barh(i, row["auc_range"], color=COLORS[row["algo"]], alpha=0.7)
                if row.get("auc_range_ci_low") is not None:
                    axis.hlines(i, row["auc_range_ci_low"], row["auc_range_ci_high"], color="black")
            axis.set_yticks(range(len(names)), names)
            axis.set(xlabel="Range of mean AUC over declared levels (95% exploratory CI)",
                     title="Sensitivity (* unmatched/incomplete pair; see sensitivity_comparison.csv)")
            paths += save(fig, out, f"overview_{context_index}_sensitivity")
    return paths


def plot_extended(runs, summary, sensitivity, out, repetitions, seed, sensitivity_checks=()):
    paths = []
    for (stratum, algo, group, family), items in sorted(group_runs(runs).items()):
        stem = f"{stratum}_{algo}_{group}_{family}"
        paths += plot_seeds(items, out, stem, repetitions, seed)
        paths += plot_diagnostics(items, out, stem)
        if run_design(items[0].config, items[0].metadata)["kind"] == "grid":
            paths += plot_grid(items, out, stem, repetitions, seed)
    paths += plot_comparisons(runs, out, repetitions, seed)
    paths += plot_overviews(runs, summary, sensitivity, out, sensitivity_checks)
    return paths
