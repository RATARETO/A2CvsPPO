"""Generate a reviewable experiment plan and run it sequentially or in CPU workers."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .experiments import PRIMARY_GROUPS, build_configs
from .io import write_json


def _run_one(cfg: ExperimentConfig, root: Path, force: bool) -> dict:
    from .train import run_experiment

    return run_experiment(cfg, root, force)


def execute(configs: list[ExperimentConfig], root: Path, workers: int = 1,
            force: bool = False) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    failures = []
    if workers == 1:
        tasks = ((cfg, None) for cfg in configs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        pending = {executor.submit(_run_one, cfg, root, force): cfg for cfg in configs}
        tasks = ((pending[future], future) for future in as_completed(pending))
    try:
        for cfg, future in tasks:
            try:
                result = future.result() if future is not None else _run_one(cfg, root, force)
                print(f"OK {cfg.algo}/{cfg.group}/{cfg.name}/seed_{cfg.seed}: "
                      f"sampled={result['final_eval_sampled_mean']:.1f}", flush=True)
            except Exception as error:
                failures.append({"config": cfg.to_dict(), "error": str(error)})
                print(f"FAILED {cfg.run_dir(root)}: {error}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=("a2c", "ppo"), default="a2c")
    parser.add_argument("--groups", nargs="+", default=PRIMARY_GROUPS)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--phase", choices=("pilot", "search", "confirmatory"), default="search")
    parser.add_argument("--config", type=Path, help="Frozen selected configuration for held-out seeds")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true", help="Print plan without training or writing files")
    parser.add_argument("--quick", action="store_true", help="Smoke run; not article evidence")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.quick and (args.config or args.phase == "confirmatory"):
        parser.error("--quick cannot be combined with --config or confirmatory runs")
    seeds = args.seeds if args.seeds is not None else (
        list(range(1000, 1020)) if args.phase == "confirmatory"
        else list(range(3 if args.phase == "pilot" else 5)))
    if args.phase == "confirmatory" and args.config is None:
        parser.error("Freeze a selected config first; confirmatory runs require --config")
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))
        base = ExperimentConfig(**data.get("config", data))
        configs = [replace(base, seed=seed, phase=args.phase) for seed in seeds]
        if args.steps is not None:
            parser.error("Set the budget in the frozen --config, not --steps")
    else:
        common = {"total_timesteps": args.steps} if args.steps is not None else {}
        groups, phase = args.groups, args.phase
        if args.quick:
            if args.steps is not None:
                parser.error("--quick uses a fixed smoke budget")
            groups, seeds, phase = ["baseline"], [0], "pilot"
            common = dict(total_timesteps=10_240, eval_freq=2_560,
                          n_eval_episodes=2, final_eval_episodes=3)
        configs = build_configs(args.algo, groups, seeds, phase, **common)
    plan = {"runs": len(configs), "configs": [cfg.to_dict() for cfg in configs]}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.list:
        return
    root = args.out or Path("results_quick" if args.quick else f"results_{args.phase}")
    root.mkdir(parents=True, exist_ok=True)
    # Content-addressed plan files preserve earlier series when new groups are run.
    import hashlib

    from .config import canonical_json

    digest = hashlib.sha256(canonical_json(plan).encode()).hexdigest()[:12]
    write_json(root / f"plan_{digest}.json", plan)
    failures = execute(configs, root, args.workers, args.force)
    write_json(root / f"failures_{digest}.json", failures)
    if failures:
        raise SystemExit(f"{len(failures)} runs failed; see failures_{digest}.json")


if __name__ == "__main__":
    main()
