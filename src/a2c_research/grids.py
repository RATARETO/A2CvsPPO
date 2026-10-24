"""Full-factorial cells and seed-level difference-in-differences contrasts."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np

from .config import PROTOCOL, canonical_json
from .design import factor_values, group_runs, run_design
from .metrics import bootstrap_mean


def ordered_levels(values) -> list:
    unique = {canonical_json(value): value for value in values}
    result = list(unique.values())
    return sorted(result, key=lambda x: (isinstance(x, str), x))


def grid_data(items: list) -> tuple:
    design = run_design(items[0].config, items[0].metadata)
    axes = design["factors"]
    levels = [design.get("levels", {}).get(axis) or ordered_levels(
        factor_values(run)[index] for run in items) for index, axis in enumerate(axes)]
    cells = defaultdict(list)
    for run in items:
        cells[tuple(canonical_json(value) for value in factor_values(run))].append(run)
    return axes, levels, cells


def grid_tables(runs: list, repetitions: int, seed: int) -> tuple[list, list]:
    table, interactions = [], []
    for (stratum, algo, group, family), items in sorted(group_runs(runs).items()):
        if run_design(items[0].config, items[0].metadata)["kind"] != "grid":
            continue
        axes, levels, cells = grid_data(items)
        common = dict(stratum=stratum, algo=algo, group=group, family=family,
                      axis_x=axes[0], axis_y=axes[1])
        for a, b in product(*levels):
            selected = cells.get((canonical_json(a), canonical_json(b)), [])
            row = {**common, "value_x": a, "value_y": b, "n_seeds": len(selected),
                   "status": "observed" if selected else "missing"}
            for metric in ("auc", "final_eval_sampled_mean", "max_drawdown", "retained_at_end"):
                values = [r.metrics[metric] for r in selected if r.metrics[metric] is not None]
                if values:
                    row.update({f"{metric}_{key}": value for key, value in
                                bootstrap_mean(values, repetitions, seed).items()})
            if selected:
                row["auc_seed_std"] = (float(np.std([r.metrics["auc"] for r in selected], ddof=1))
                                       if len(selected) > 1 else None)
                row["ever_reached_rate"] = float(np.mean([r.metrics["ever_reached"] for r in selected]))
            table.append(row)
        if not all(levels):
            continue
        base_a, base_b = levels[0][0], levels[1][0]
        # Every contrast uses the four corner cells and the exact same training seeds.
        # These are exploratory CIs, not multiple-comparison-adjusted tests.
        for a, b in product(levels[0][1:], levels[1][1:]):
            corners = [(a, b), (a, base_b), (base_a, b), (base_a, base_b)]
            sets = [{r.config["seed"]: r for r in cells.get(
                (canonical_json(x), canonical_json(y)), [])} for x, y in corners]
            row = {**common, "value_x": a, "value_y": b, "reference_x": base_a,
                   "reference_y": base_b, "n_seeds": 0}
            if not all(sets) or any(set(s) != set(sets[0]) for s in sets):
                row.update(status="incomplete", reason="Four cells require identical nonempty seed sets")
            else:
                values = [sets[0][s].metrics["auc"] - sets[1][s].metrics["auc"]
                          - sets[2][s].metrics["auc"] + sets[3][s].metrics["auc"]
                          for s in sorted(sets[0])]
                row.update(status="observed", **bootstrap_mean(values, repetitions, seed))
                row["n_seeds"] = len(values)
            interactions.append(row)
    return table, interactions


def sensitivity_audit(runs: list, sensitivity: list) -> list:
    families = group_runs(runs)
    candidates = defaultdict(lambda: defaultdict(list))
    for row in sensitivity:
        candidates[(row["stratum"], row["group"])][row["algo"]].append(row)
    rows = []
    for (stratum, group), algorithms in sorted(candidates.items()):
        row = dict(stratum=stratum, group=group, status="descriptive_only")
        if any(len(algorithms.get(algo, [])) != 1 for algo in ("a2c", "ppo")):
            row["reason"] = "Both algorithms need one unambiguous family"
        else:
            a, b = algorithms["a2c"][0], algorithms["ppo"][0]
            items = [families[(stratum, algo, group, value["family"])]
                     for algo, value in (("a2c", a), ("ppo", b))]
            levels = [sorted({canonical_json(factor_values(r)) for r in collection})
                      for collection in items]
            designs = [run_design(collection[0].config, collection[0].metadata) for collection in items]
            row.update(a2c_auc_range=a.get("auc_range"), ppo_auc_range=b.get("auc_range"),
                       a2c_levels=canonical_json(levels[0]), ppo_levels=canonical_json(levels[1]),
                       ranges_match=levels[0] == levels[1])
            if any(collection[0].protocol != PROTOCOL for collection in items):
                row["reason"] = "Unversioned reference results"
            elif not all("auc_range" in value for value in (a, b)):
                row["reason"] = "Missing levels or unequal seed sets within a series"
            elif levels[0] != levels[1]:
                row["reason"] = "Factor ranges differ"
            elif designs[0].get("constant_rollout") != designs[1].get("constant_rollout"):
                row["reason"] = "Algorithm-specific control rollout sizes differ"
            else:
                row.update(status="matched_ranges", reason="Matched levels; declared native algorithm settings")
        rows.append(row)
    return rows
