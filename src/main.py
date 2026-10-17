"""Демо: «до обучения» и «после обучения» на одном PPO с гиперпараметрами по умолчанию.

    python -m src.main

Что произойдёт:
  1. запишется видео случайной политики -> videos/random_policy.mp4/.gif
  2. обучится PPO (100 000 шагов, ~1-2 минуты) -> results/ppo/demo/default/seed_0/
     (config.json, eval.csv, summary.json, model.zip, policy.mp4/.gif)
  3. напечатается итог. Кривую обучения смотри: python -m src.analyze
"""
from __future__ import annotations

import json

from .config import ROOT, make_config
from .train import run_experiment
from .video import record_episodes


def main() -> None:
    print("1/2  Видео случайной политики...")
    record_episodes(None, "CartPole-v1", ROOT / "videos" / "random_policy",
                    label="RANDOM policy (no training)", n_episodes=3)

    print("2/2  Обучение PPO (дефолтные гиперпараметры) + видео...")
    cfg = make_config("ppo", "demo", {}, seed=0, total_timesteps=100_000)
    summary = run_experiment(cfg, record_video=True)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Результаты: {cfg.run_dir()}")


if __name__ == "__main__":
    main()
