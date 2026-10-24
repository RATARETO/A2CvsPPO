"""Predeclared 12-run feasibility pilot; no automatic environment selection or tuning."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .config import canonical_json, environment_config
from .envs import ENVIRONMENTS
from .io import write_json
from .sweep import execute


def build_pilot(envs=("Acrobot-v1", "Pendulum-v1"), seeds=(0, 1, 2), quick=False):
    if not envs or len(set(envs)) != len(envs) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Environments and seeds must be nonempty and unique")
    overrides = dict(total_timesteps=10_240, eval_freq=2_560, n_eval_episodes=2,
                     final_eval_episodes=3) if quick else {}
    return [environment_config(env, algo=algo, phase="pilot", seed=seed, **overrides)
            for env in envs for algo in ("a2c", "ppo") for seed in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", choices=list(ENVIRONMENTS), nargs="+",
                        default=["Acrobot-v1", "Pendulum-v1"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    configs = build_pilot(args.envs, [0] if args.quick else args.seeds, args.quick)
    plan = {"study": "multi-environment-feasibility-pilot-v1", "runs": len(configs),
            "configs": [cfg.to_dict() for cfg in configs],
            "execution": {"workers": args.workers, "device": "cpu", "torch_threads_per_run": 1},
            "decision_rule": "Review correctness, curves and cost. No environment exclusion based on algorithm ranking."}
    for cfg in configs:
        print(f"{cfg.env_id} {cfg.algo} seed={cfg.seed}: {cfg.total_timesteps} steps, "
              f"threshold={cfg.threshold}")
    print(f"Total {len(configs)} runs; no hyperparameter selection.")
    if args.list:
        return
    root = args.out or Path("results_multienv_quick" if args.quick else "results_multienv_pilot")
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(canonical_json(plan).encode()).hexdigest()[:12]
    write_json(root / f"plan_{digest}.json", plan)
    failures = execute(configs, root, args.workers)
    write_json(root / f"failures_{digest}.json", failures)
    if failures:
        raise SystemExit(f"{len(failures)} failed runs; see failures_{digest}.json")


if __name__ == "__main__":
    main()
