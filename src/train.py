"""Один эксперимент = одна функция run_experiment(cfg).

Что сохраняется в results/<algo>/<group>/<name>/seed_<k>/ :
    config.json   конфиг + ФАКТИЧЕСКИЕ гиперпараметры модели + версии библиотек
    eval.csv      кривая обучения (оценка политики каждые eval_freq шагов + диагностика)
    summary.json  итог: финальная оценка, время обучения (его наличие = «запуск завершён»)
    model.zip     обученная модель
    policy.mp4/.gif   (только если record_video=True)

Запуск вручную из командной строки (из корня проекта):
    python -m src.train --algo ppo --steps 100000 --seed 0 --set learning_rate=0.001 --video
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
import time
from pathlib import Path
from typing import Any

import gymnasium
import stable_baselines3
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from .algos import ALGOS, build_policy_kwargs, effective_hyperparams
from .callbacks import EvalLogger
from .config import RESULTS_DIR, ExperimentConfig, make_config
from .envs import make_eval_env, make_train_env
from .video import record_episodes


def run_experiment(
    cfg: ExperimentConfig,
    results_root: Path | str = RESULTS_DIR,
    force: bool = False,
    record_video: bool = False,
) -> dict[str, Any]:
    run_dir = cfg.run_dir(results_root)
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    if summary_path.exists() and config_path.exists() and not force:
        stored = json.loads(config_path.read_text(encoding="utf-8")).get("config")
        if stored == json.loads(json.dumps(cfg.to_dict())):  # тот же конфиг -> уже посчитано, пропускаем
            return json.loads(summary_path.read_text(encoding="utf-8"))
        # конфиг изменился (например, другой бюджет шагов) -> пересчитываем, а не берём устаревший результат

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # для маленькой сети 1 поток на процесс быстрее и позволяет гонять запуски параллельно

    train_env = make_train_env(cfg)
    eval_env = make_eval_env(cfg)

    model = ALGOS[cfg.algo](
        "MlpPolicy",
        train_env,
        seed=cfg.seed,
        device="cpu",
        verbose=0,
        policy_kwargs=build_policy_kwargs(cfg.policy_kwargs_spec()),
        **cfg.algo_params,
    )

    meta = {
        "config": cfg.to_dict(),
        "effective_hyperparams": effective_hyperparams(model),
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger = EvalLogger(eval_env, cfg.eval_freq, cfg.n_eval_episodes)
    started = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=logger)
    train_time = time.time() - started

    logger.to_dataframe().to_csv(run_dir / "eval.csv", index=False)
    model.save(run_dir / "model")

    final_env = make_eval_env(cfg, seed_offset=20_000)
    final_mean, final_std = evaluate_policy(
        model, final_env, n_eval_episodes=cfg.final_eval_episodes, deterministic=True
    )
    sampled_mean, sampled_std = evaluate_policy(
        model, final_env, n_eval_episodes=cfg.final_eval_episodes, deterministic=False
    )
    summary = {
        "algo": cfg.algo,
        "group": cfg.group,
        "name": cfg.name,
        "seed": cfg.seed,
        "timesteps_done": int(model.num_timesteps),
        "final_eval_mean": float(final_mean),          # greedy (argmax), как в видео
        "final_eval_std": float(final_std),
        "final_eval_sampled_mean": float(sampled_mean),  # сэмплирование, как в кривых обучения
        "final_eval_sampled_std": float(sampled_std),
        "train_time_sec": round(train_time, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for env in (train_env, eval_env, final_env):
        env.close()

    if record_video:
        label = f"{cfg.algo.upper()} | {cfg.name} | seed {cfg.seed} | {cfg.total_timesteps // 1000}k steps"
        record_episodes(model, cfg.env_id, run_dir / "policy", label=label, n_episodes=3, seed=cfg.seed)

    return summary


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # строка, например "relu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить один эксперимент")
    parser.add_argument("--algo", default="ppo", choices=sorted(ALGOS))
    parser.add_argument("--group", default="manual")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="ключ=значение",
                        help="переопределить гиперпараметр, можно несколько раз: --set clip_range=0.1")
    parser.add_argument("--video", action="store_true", help="записать видео обученной политики")
    parser.add_argument("--force", action="store_true", help="перезаписать, даже если запуск уже есть")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key.strip()] = _parse_value(value.strip())

    cfg = make_config(args.algo, args.group, overrides, args.seed, total_timesteps=args.steps)
    print(f"Запуск: {cfg.run_dir()}")
    summary = run_experiment(cfg, force=args.force, record_video=args.video)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
