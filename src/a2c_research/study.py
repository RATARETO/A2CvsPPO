"""Freeze, inspect and execute a reproducible main-study matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .config import DEFAULTS, ExperimentConfig, canonical_json, environment_config, make_config
from .design import training_key
from .envs import ENVIRONMENTS
from .experiments import SWEEPS, declared_design, suite_groups
from .io import runtime_metadata, training_identity, write_json
from .results import load_runs
from .sweep import execute

SCHEMA = "main-search-study-v1"


def digest(payload: dict) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in payload.items()
                                         if k != "plan_sha256"}).encode()).hexdigest()


def unique(values, label):
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be nonempty and unique")


def prepare_study(envs=None, algos=None, seeds=None, suite="core") -> dict:
    envs = list(ENVIRONMENTS) if envs is None else list(envs)
    algos = ["a2c", "ppo"] if algos is None else list(algos)
    seeds = list(range(5)) if seeds is None else list(seeds)
    for values, name in ((envs, "environments"), (algos, "algorithms"), (seeds, "seeds")):
        unique(values, name)
    runtime = runtime_metadata()
    plan = {"schema": SCHEMA, "phase": "search", "suite": suite, "seeds": seeds,
            "training_sha256": runtime["training_sha256"], "versions": runtime["versions"],
            "base_configs": {env: environment_config(env, phase="search").to_dict() for env in envs},
            "algorithms": {},
            "interpretation": "Fixed-budget comparisons within each environment; no pooled reward ranking. "
                              "No Pendulum success threshold. Confirmation uses a separate frozen selection."}
    for algo in algos:
        if algo not in DEFAULTS:
            raise ValueError(f"Unknown algorithm {algo}")
        groups = suite_groups(algo, suite)
        plan["algorithms"][algo] = {
            "baseline_params": deepcopy(DEFAULTS[algo]),
            "series": {group: deepcopy(SWEEPS[algo][group]) for group in groups},
            "designs": {group: declared_design(make_config(algo, group, SWEEPS[algo][group][0], 0))
                        for group in groups},
        }
    plan["plan_sha256"] = digest(plan)
    compile_study(plan)
    return plan


def compile_study(plan: dict, check_runtime: bool = True) -> list[ExperimentConfig]:
    if plan.get("schema") != SCHEMA or plan.get("phase") != "search":
        raise ValueError("Unsupported study schema or phase")
    if plan.get("plan_sha256") != digest(plan):
        raise ValueError("Frozen study checksum mismatch; prepare a new plan after intentional changes")
    seeds = plan["seeds"]
    unique(seeds, "seeds")
    if not plan["base_configs"] or not plan["algorithms"]:
        raise ValueError("A study needs environments and algorithms")
    runtime = runtime_metadata() if check_runtime else None
    configs = []
    for env, base_data in plan["base_configs"].items():
        base = ExperimentConfig(**base_data)
        if base.env_id != env or base.phase != "search" or base.algo_params:
            raise ValueError("Invalid frozen environment baseline")
        if runtime is not None:
            identity = training_identity({"config": base.to_dict(),
                                          "runtime": {"training_sha256": plan["training_sha256"]}})
            if identity != runtime["training_sha256"] or plan["versions"] != runtime["versions"]:
                raise ValueError("Frozen study training code or dependency versions do not match this runtime")
        common = base.to_dict()
        for key in ("algo", "group", "name", "seed", "sweep_index", "algo_params"):
            common.pop(key)
        for algo, specification in plan["algorithms"].items():
            if algo not in DEFAULTS or specification["baseline_params"] != DEFAULTS[algo]:
                raise ValueError("Frozen algorithm defaults differ from the installed study code")
            if not specification["series"] or set(specification["designs"]) != set(specification["series"]):
                raise ValueError("Missing frozen series or design")
            for group, overrides_list in specification["series"].items():
                # The trainer writes registered design metadata. Refuse silent drift.
                if overrides_list != SWEEPS[algo].get(group):
                    raise ValueError(f"Registered series changed: {algo}/{group}; use the matching code revision")
                for index, overrides in enumerate(overrides_list):
                    for seed in seeds:
                        cfg = make_config(algo, group, overrides, seed, index, **common)
                        if declared_design(cfg) != specification["designs"][group]:
                            raise ValueError("Frozen design no longer matches the trainer")
                        configs.append(cfg)
    paths = [cfg.run_dir(Path("results_search")) for cfg in configs]
    if len(set(paths)) != len(paths):
        raise ValueError("Study contains conflicting result paths")
    return configs


def save_study(plan: dict, path: Path) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8-sig"))
        if previous != plan:
            raise ValueError("A different frozen plan already exists; use a new filename")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, plan)


def read_study(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def select_configs(configs, envs=None, algos=None, seeds=None, groups=None):
    selected = list(configs)
    for choices, field in ((envs, "env_id"), (algos, "algo"), (seeds, "seed"), (groups, "group")):
        if choices is None:
            continue
        unique(choices, field)
        if not set(choices) <= {getattr(cfg, field) for cfg in configs}:
            raise ValueError(f"Selection outside the frozen study: {field}")
        selected = [cfg for cfg in selected if getattr(cfg, field) in choices]
    if not selected:
        raise ValueError("Empty study selection")
    return selected


def inventory(configs, root: Path) -> dict:
    runs, _ = load_runs([root]) if root.is_dir() else ([], [])
    completed = {r.path.resolve(): r for r in runs}
    runtime = runtime_metadata()
    result = {"completed": [], "pending": [], "blocked": []}
    for cfg in configs:
        path = cfg.run_dir(root).resolve()
        run = completed.get(path)
        if run is not None:
            if (run.config == cfg.to_dict() and run.metadata.get("versions") == runtime["versions"]
                    and training_identity(run.metadata) == runtime["training_sha256"]):
                result["completed"].append(cfg)
            else:
                result["blocked"].append((cfg, "Existing run has incompatible settings or runtime"))
        elif (path / ".running").exists() or (path / "summary.json").exists():
            result["blocked"].append((cfg, "Run is active, interrupted with a lock, or has invalid artifacts"))
        elif (path / "config.json").exists():
            try:
                stored = json.loads((path / "config.json").read_text(encoding="utf-8"))
                if (stored["config"] != cfg.to_dict() or stored.get("versions") != runtime["versions"]
                        or training_identity(stored) != runtime["training_sha256"]):
                    raise ValueError("Partial run has incompatible settings or runtime")
                result["pending"].append(cfg)
            except (OSError, ValueError, KeyError) as error:
                result["blocked"].append((cfg, str(error)))
        else:
            result["pending"].append(cfg)
    return result


def print_plan(plan, configs, state):
    print(f"Study {SCHEMA}: {plan['plan_sha256']}")
    counts = Counter((cfg.env_id, cfg.algo, cfg.total_timesteps) for cfg in configs)
    for (env, algo, budget), count in sorted(counts.items()):
        print(f"{env} / {algo}: {count} run slots, {budget} training transitions per run")
    print(f"Selected: {len(configs)} slots; {len({training_key(cfg) for cfg in configs})} distinct seeded conditions")
    print(f"Compatible completed: {len(state['completed'])}; pending: {len(state['pending'])}; "
          f"blocked: {len(state['blocked'])}")
    for cfg, reason in state["blocked"][:10]:
        print(f"BLOCKED {cfg.env_id}/{cfg.algo}/{cfg.group}/{cfg.name}/seed_{cfg.seed}: {reason}")


def run_study(plan, configs, root: Path, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    compile_study(plan)
    allowed = {cfg.fingerprint() for cfg in compile_study(plan, check_runtime=False)}
    if not configs or any(cfg.fingerprint() not in allowed for cfg in configs):
        raise ValueError("Execution selection is not part of the frozen study")
    unique([cfg.fingerprint() for cfg in configs], "execution configurations")
    state = inventory(configs, root)
    print_plan(plan, configs, state)
    if state["blocked"]:
        raise ValueError("Resolve blocked runs or choose a new output directory before execution")
    root.mkdir(parents=True, exist_ok=True)
    execution = {"study_sha256": plan["plan_sha256"], "configs": [cfg.to_dict() for cfg in configs],
                 "workers": workers}
    identity = digest(execution)[:12]
    write_json(root / f"plan_{identity}.json", execution)
    failures = execute(state["pending"], root, workers)
    write_json(root / f"failures_{identity}.json", failures)
    outcome = {"study_sha256": plan["plan_sha256"], "selected": len(configs),
               "reused_completed": len(state["completed"]), "scheduled": len(state["pending"]),
               "failed": len(failures)}
    write_json(root / f"execution_{identity}.json", outcome)
    if failures:
        raise RuntimeError(f"{len(failures)} failed runs; see failures_{identity}.json")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", type=Path, help="Save a compact frozen plan without training")
    mode.add_argument("--plan", type=Path, help="Run or inspect a previously frozen plan")
    parser.add_argument("--suite", choices=("primary", "core", "ofat", "grids", "all"))
    parser.add_argument("--envs", choices=list(ENVIRONMENTS), nargs="+")
    parser.add_argument("--algos", choices=list(DEFAULTS), nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results_search"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.plan:
        if args.suite:
            parser.error("--suite cannot alter a frozen plan")
        plan = read_study(args.plan)
        configs = select_configs(compile_study(plan), args.envs, args.algos, args.seeds, args.groups)
    else:
        if args.groups:
            parser.error("--groups filters a frozen --plan; use --suite when preparing")
        plan = prepare_study(args.envs, args.algos, args.seeds, args.suite or "core")
        configs = compile_study(plan)
    if args.prepare:
        save_study(plan, args.prepare)
        print(f"Frozen plan saved: {args.prepare}")
    if args.list or not args.plan:
        print_plan(plan, configs, inventory(configs, args.out))
        return
    run_study(plan, configs, args.out, args.workers)


if __name__ == "__main__":
    main()
