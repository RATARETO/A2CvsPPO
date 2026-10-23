"""Train A2C (or a PPO control) under one explicitly versioned protocol."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .config import ExperimentConfig
from .evaluation import evaluate
from .io import file_hash, runtime_metadata, write_csv, write_json

ARTIFACTS = ("eval.csv", "evaluation_episodes.csv", "final_episodes.csv", "model.zip")


class EvaluationCallback(BaseCallback):
    def __init__(self, cfg: ExperimentConfig, run_dir: Path):
        super().__init__(verbose=0)
        self.cfg, self.run_dir = cfg, run_dir
        self.rows: list[dict] = []
        self.episodes: list[dict] = []
        self.evaluation_seconds = 0.0
        self.started = 0.0
        self.next_eval = cfg.eval_freq

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        self._evaluate()

    def _on_step(self) -> bool:
        # At intermediate t, evaluate the policy available after collecting t transitions.
        # The final checkpoint is deferred until the last optimizer update has finished.
        if self.num_timesteps >= self.next_eval and self.num_timesteps < self.cfg.total_timesteps:
            self._evaluate()
            self.next_eval += self.cfg.eval_freq
        return True

    def _on_training_end(self) -> None:
        self._evaluate()

    def _evaluate(self) -> None:
        started = time.perf_counter()
        result = evaluate(self.model, self.cfg.env_id, self.cfg.eval_seed, self.cfg.n_eval_episodes)
        self.evaluation_seconds += time.perf_counter() - started
        buffer = self.model.ep_info_buffer
        row = {"timesteps": int(self.num_timesteps),
               **{k: v for k, v in result.items() if k != "episodes"},
               "train_return_mean": float(np.mean([r["r"] for r in buffer])) if buffer else "",
               "wall_time_sec": time.perf_counter() - self.started,
               "updates": int(self.model._n_updates)}
        for key in ("entropy_loss", "value_loss", "explained_variance",
                    "approx_kl", "clip_fraction"):
            value = self.logger.name_to_value.get(f"train/{key}")
            row[key] = float(value) if value is not None and np.isfinite(value) else ""
        self.rows.append(row)
        self.episodes.extend({"timesteps": int(self.num_timesteps), **r}
                             for r in result["episodes"])
        write_csv(self.run_dir / "eval.csv", self.rows)
        write_csv(self.run_dir / "evaluation_episodes.csv", self.episodes)


def run_experiment(cfg: ExperimentConfig, results_root: Path | str = "results",
                   force: bool = False) -> dict:
    # A completed run is reusable only under the same config, source and runtime.
    runtime = runtime_metadata()
    run_dir = cfg.run_dir(results_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".running"
    try:
        lock_path.open("x").close()
    except FileExistsError as error:
        raise RuntimeError(f"Run already active or interrupted: {lock_path}. "
                           "Verify no worker is running before removing this file.") from error
    train_env = None
    try:
        config_path, summary_path = run_dir / "config.json", run_dir / "summary.json"
        if config_path.exists() and not force:
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            same = (stored["config"] == cfg.to_dict()
                    and stored.get("runtime", {}).get("source_sha256") == runtime["source_sha256"]
                    and stored.get("versions") == runtime["versions"])
            if not same:
                raise ValueError(f"Existing run has a different config/source/runtime: {run_dir}. "
                                 "Use a new output directory or explicitly --force.")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if all((run_dir / name).exists()
                       and file_hash(run_dir / name) == summary.get("artifact_sha256", {}).get(name)
                       for name in ARTIFACTS):
                    return summary
                raise ValueError(f"Completed run has missing or modified artifacts: {run_dir}")
        # Completion marker is written last, so an interrupted run is never considered complete.
        summary_path.unlink(missing_ok=True)
        (run_dir / "failure.json").unlink(missing_ok=True)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        train_env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)
        activation = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[cfg.activation]
        model = {"a2c": A2C, "ppo": PPO}[cfg.algo](
            "MlpPolicy", train_env, seed=cfg.seed, device="cpu", verbose=0,
            policy_kwargs={"net_arch": cfg.net_arch, "activation_fn": activation},
            **cfg.resolved_params(),
        )
        optimizer = model.policy.optimizer
        optimizer_spec = {"class": type(optimizer).__name__,
                          "defaults": {key: value for key, value in optimizer.defaults.items()
                                       if isinstance(value, (float, int, bool, str, tuple))
                                       or value is None}}
        metadata = {"config": cfg.to_dict(), "config_sha256": cfg.fingerprint(),
                    "effective_hyperparams": cfg.resolved_params(),
                    "policy_kwargs": {"net_arch": cfg.net_arch, "activation": cfg.activation},
                    "optimizer": optimizer_spec, "versions": runtime["versions"],
                    "runtime": runtime, "created_utc": datetime.now(timezone.utc).isoformat()}
        write_json(config_path, metadata)
        callback = EvaluationCallback(cfg, run_dir)
        started = time.perf_counter()
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
        elapsed = time.perf_counter() - started
        if model.num_timesteps != cfg.total_timesteps:
            raise RuntimeError("Actual training budget differs from the preregistered budget")
        model.save(run_dir / "model.zip")
        final = evaluate(model, cfg.env_id, cfg.final_eval_seed, cfg.final_eval_episodes)
        write_csv(run_dir / "final_episodes.csv", final["episodes"])
        summary = {
            "algo": cfg.algo, "group": cfg.group, "name": cfg.name, "seed": cfg.seed,
            "protocol": cfg.protocol, "phase": cfg.phase,
            "timesteps_done": int(model.num_timesteps),
            "final_eval_mean": final["eval_greedy_mean"],
            "final_eval_std": final["eval_greedy_std"],
            "final_eval_sampled_mean": final["eval_sampled_mean"],
            "final_eval_sampled_std": final["eval_sampled_std"],
            "train_time_sec": elapsed,
            "intermediate_eval_time_sec": callback.evaluation_seconds,
            "updates": int(model._n_updates),
            "artifact_sha256": {name: file_hash(run_dir / name) for name in ARTIFACTS},
        }
        write_json(summary_path, summary)
        return summary
    except Exception as error:
        # Do not replace an existing completed result's status on a configuration conflict.
        if not (run_dir / "summary.json").exists():
            write_json(run_dir / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if train_env is not None:
            train_env.close()
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="ExperimentConfig JSON, saved config.json also accepted")
    parser.add_argument("--algo", choices=("a2c", "ppo"), default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phase", choices=("pilot", "search", "confirmatory"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="PARAM=JSON_VALUE")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    cfg = ExperimentConfig(**data.get("config", data))
    changes = {key: value for key, value in {
        "algo": args.algo, "seed": args.seed, "phase": args.phase,
        "total_timesteps": args.steps}.items() if value is not None}
    params = dict(cfg.algo_params)
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error("--set requires PARAM=JSON_VALUE")
        params[key] = json.loads(value)
    cfg = replace(cfg, **changes, algo_params=params)
    result = run_experiment(cfg, args.out, args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
